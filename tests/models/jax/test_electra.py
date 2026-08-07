# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the JAX-native ELECTRA embedding model.

Numerical parity is checked against `transformers.ElectraModel` (native HF
support, float32, no remote code) — cosine similarity of per-token hidden
states and mean-pooled embeddings must exceed 0.999.
"""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig, set_current_vllm_config
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.electra import ElectraForPreTraining

MODEL_ID = "google/electra-small-discriminator"


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Electra model."""

    def __init__(self, model: str):
        self.model_config = ModelConfig(model, runner="pooling")
        self.model_config.dtype = jnp.float32
        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.cache_config = MagicMock(cache_dtype="auto")
        self.quant_config = None
        self.parallel_config = None


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, 1, 1, 1))
    m = Mesh(device_mesh, axis_names=('data', 'attn_dp', 'expert', 'model'))
    with jax.set_mesh(m):
        yield m


@pytest.fixture
def rng() -> PRNGKey:
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def local_model_dir(tmp_path_factory):
    """Download once and re-save as safetensors: the HF repo ships only
    pytorch_model.bin, while tpu-inference's loader reads *.safetensors."""
    transformers = pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("electra_st")
    try:
        model = transformers.AutoModelForPreTraining.from_pretrained(MODEL_ID)
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:  # e.g. no network
        pytest.skip(f"Could not download {MODEL_ID}: {e}")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    return str(out)


@pytest.fixture(scope="module")
def mock_vllm_config(local_model_dir):
    # Register the arch with vLLM's registry first (normally done by the
    # vllm.general_plugins entrypoint).
    from tpu_inference.models.vllm.experimental import register_models
    register_models()
    try:
        return MockVllmConfig(local_model_dir)
    except Exception as e:
        pytest.skip(f"Could not build ModelConfig for {MODEL_ID}: {e}")


def _make_attention_metadata(seq_lens_list):
    total = int(sum(seq_lens_list))
    positions = jnp.concatenate(
        [jnp.arange(n, dtype=jnp.int32) for n in seq_lens_list])
    seq_lens = jnp.array(seq_lens_list, dtype=jnp.int32)
    query_start_loc = jnp.array(np.cumsum([0] + list(seq_lens_list)),
                                dtype=jnp.int32)
    return AttentionMetadata(
        input_positions=positions,
        block_tables=jnp.zeros((len(seq_lens_list), ), dtype=jnp.int32),
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=jnp.array([0, 0, len(seq_lens_list)],
                                       dtype=jnp.int32),
    ), total


class TestElectra:

    def test_model_init(self, mock_vllm_config, rng, mesh):
        with jax.set_mesh(mesh):
            model = ElectraForPreTraining(mock_vllm_config, rng, mesh)

        hf_config = mock_vllm_config.model_config.hf_config
        embedding_size = hf_config.embedding_size
        layers = model.model.encoder.layer
        assert len(layers) == hf_config.num_hidden_layers

        attn = getattr(layers[0].attention, "self")
        assert attn.num_heads == hf_config.num_attention_heads
        assert attn.head_dim == hf_config.hidden_size // \
            hf_config.num_attention_heads
        assert attn.query.weight.shape == (hf_config.hidden_size,
                                           attn.num_heads, attn.head_dim)
        assert attn.query.bias.shape == (attn.num_heads, attn.head_dim)

        layer = layers[0]
        assert layer.intermediate.dense.weight.shape == (
            hf_config.hidden_size, hf_config.intermediate_size)
        assert layer.output.dense.weight.shape == (
            hf_config.intermediate_size, hf_config.hidden_size)

        emb = model.model.embeddings
        assert emb.word_embeddings.weight.shape == (hf_config.vocab_size,
                                                    embedding_size)
        assert emb.position_embeddings.weight.shape == (
            hf_config.max_position_embeddings, embedding_size)
        # Factorized embeddings: 128 -> 256 projection must exist.
        assert model.model.embeddings_project is not None
        assert model.model.embeddings_project.weight.shape == (
            embedding_size, hf_config.hidden_size)

    def test_forward_and_hf_parity(self, mock_vllm_config, local_model_dir,
                                   rng, mesh):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        # --- Reference: native transformers ElectraModel (fp32), loaded
        # from the same converted checkpoint the JAX model will load ---
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            local_model_dir)
        ref_model = transformers.ElectraModel.from_pretrained(
            local_model_dir, torch_dtype=torch.float32)
        ref_model.eval()

        sentences = [
            "How is the weather today?",
            "ELECTRA discriminators run on TPU v6e with vLLM.",
        ]
        encoded = [tokenizer(s, return_tensors="pt") for s in sentences]
        ref_hidden = []
        with torch.no_grad():
            for e in encoded:
                out = ref_model(input_ids=e["input_ids"],
                                attention_mask=e["attention_mask"])
                # last_hidden_state: [1, T, hidden]
                ref_hidden.append(
                    out.last_hidden_state[0].numpy().astype(np.float32))

        # --- JAX model, same tokens as one concatenated ragged batch ---
        with jax.set_mesh(mesh):
            model = ElectraForPreTraining(mock_vllm_config, rng, mesh)
        with jax.set_mesh(mesh), set_current_vllm_config(mock_vllm_config):
            loader = get_model_loader(LoadConfig(load_format="hf"))
            loader.load_weights(model, mock_vllm_config.model_config)

        seq_lens = [int(e["input_ids"].shape[1]) for e in encoded]
        input_ids = jnp.array(
            np.concatenate([e["input_ids"][0].numpy() for e in encoded]),
            dtype=jnp.int32)
        metadata, total = _make_attention_metadata(seq_lens)

        kv_caches, hidden, aux, _ = model(kv_caches=[],
                                          input_ids=input_ids,
                                          attention_metadata=metadata)
        assert hidden.shape == (total,
                                mock_vllm_config.model_config.hf_config.
                                hidden_size)
        assert kv_caches == []
        assert aux == []

        hidden = np.asarray(hidden, dtype=np.float32)
        offset = 0
        for ref, n in zip(ref_hidden, seq_lens):
            got = hidden[offset:offset + n]
            offset += n
            # Per-token cosine similarity.
            cos = np.sum(ref * got, axis=-1) / (
                np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1))
            assert cos.min() > 0.999, f"per-token cosine too low: {cos.min()}"
            # Mean-pooled embedding cosine similarity.
            ref_emb = ref.mean(axis=0)
            got_emb = got.mean(axis=0)
            emb_cos = np.dot(ref_emb, got_emb) / (np.linalg.norm(ref_emb) *
                                                  np.linalg.norm(got_emb))
            assert emb_cos > 0.999, f"pooled cosine too low: {emb_cos}"
