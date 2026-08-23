# Copyright 2025 Google LLC
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

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.granite import GraniteForCausalLM
from tpu_inference.runner.kv_cache import create_kv_caches


class MockVllmConfig:

    def __init__(self, model: str, kv_cache_dtype: str):
        self.model_config = ModelConfig(model)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.speculative_config = None
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)


@pytest.fixture(scope="module")
def mesh():
    """Creates a mesh with 1 device."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m


@pytest.fixture
def mock_model_inputs():
    num_tokens = 8
    num_reqs = 1
    max_num_blocks_per_req = 4
    input_ids = jnp.ones((num_tokens, ), dtype=jnp.int32)
    positions = jnp.ones((num_tokens, ), dtype=jnp.int32)
    block_tables = jnp.zeros((num_reqs, max_num_blocks_per_req),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.ones((num_reqs, ), dtype=jnp.int32)
    query_start_loc = jnp.ones((num_reqs + 1, ), dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 0], dtype=jnp.int32)

    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    indices_do_sample = jnp.ones((num_reqs, ), dtype=jnp.int32)

    return (input_ids, attention_metadata, indices_do_sample)


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    mock_pp = MagicMock(is_first_rank=True,
                        is_last_rank=True,
                        rank_in_group=0,
                        world_size=1)
    with patch("tpu_inference.models.jax.granite.get_pp_group",
               return_value=mock_pp), patch(
                   "tpu_inference.layers.jax.pp_utils.get_pp_group",
                   return_value=mock_pp):
        yield


class TestGraniteForCausalLM:
    """Tests for the GraniteForCausalLM model class."""

    @pytest.mark.parametrize("mock_vllm_config", [
        MockVllmConfig("ibm-granite/granite-3.3-2b-instruct", "auto"),
        MockVllmConfig("ibm-granite/granite-3.3-2b-instruct", "fp8")
    ])
    def test_granite(self, mock_vllm_config, rng, mesh, mock_model_inputs):
        """Tests model init and forward for the Granite architecture."""

        # Test model init
        with jax.set_mesh(mesh):
            model = GraniteForCausalLM(mock_vllm_config, rng, mesh)

        model_config = mock_vllm_config.model_config
        hf_config = model_config.hf_config

        assert model.mesh.shape == {
            "data": 1,
            "attn_dp": 1,
            "expert": 1,
            "model": 1
        }

        layers = model.model.layers
        assert len(layers) == hf_config.num_hidden_layers
        assert isinstance(model.rng, nnx.Rngs)
        assert (model.model.lm_head == model.model.embed.embedding).all()

        # Granite-specific: the config must carry the four scaling multipliers
        # and the model must surface them for the forward pass.
        assert hasattr(hf_config, "attention_multiplier")
        assert hasattr(hf_config, "embedding_multiplier")
        assert hasattr(hf_config, "residual_multiplier")
        assert hasattr(hf_config, "logits_scaling")
        assert model.model.embedding_multiplier == hf_config.embedding_multiplier
        assert layers[0].residual_multiplier == hf_config.residual_multiplier

        attn = layers[0].self_attn
        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)
        intermediate_size = hf_config.intermediate_size

        # Granite replaces the default 1/sqrt(head_dim) softmax scale with its
        # own attention multiplier.
        assert attn.attention_multiplier == hf_config.attention_multiplier

        assert attn.hidden_size == hidden_size
        assert attn.num_heads == num_heads
        assert attn.num_kv_heads == num_kv_heads
        assert attn.head_dim_original == head_dim
        assert attn.q_proj.kernel.shape == (hidden_size, num_heads,
                                            attn.head_dim)
        assert attn.k_proj.kernel.shape == (hidden_size, num_kv_heads,
                                            attn.head_dim)
        assert attn.v_proj.kernel.shape == (hidden_size, num_kv_heads,
                                            attn.head_dim)
        assert attn.o_proj.kernel.shape == (num_heads, attn.head_dim,
                                            hidden_size)

        mlp = layers[0].mlp
        assert mlp.gate_proj.kernel.shape == (hidden_size, intermediate_size)
        assert mlp.up_proj.kernel.shape == (hidden_size, intermediate_size)
        assert mlp.down_proj.kernel.shape == (intermediate_size, hidden_size)

        # Test model load
        model.load_weights(rng)

        # Test model forward
        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=32,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            mesh=mesh,
            layer_names=["layer"] * hf_config.num_hidden_layers,
            cache_dtype=jnp.float8_e4m3fn
            if mock_vllm_config.cache_config.cache_dtype == "fp8" else
            jnp.bfloat16)
        # 1 seq with 8 tokens
        input_ids, attention_metadata, indices_do_sample = mock_model_inputs
        kv_caches, hidden_states, aux_hidden_states, _ = model(
            kv_caches, input_ids, attention_metadata, None, None, None, None,
            None, True, True)
        assert hidden_states.shape == (8, hidden_size)
        assert len(aux_hidden_states) == 0

        hidden_states = hidden_states[indices_do_sample]
        assert hidden_states.shape == (1, hidden_size)

        logits = model.compute_logits(hidden_states)
        assert logits.shape == (1, hf_config.vocab_size)

    def test_logits_scaling_applied(self, mock_vllm_config, rng, mesh):
        """logits must be divided by logits_scaling."""
        cfg = MockVllmConfig("ibm-granite/granite-3.3-2b-instruct", "auto")
        with jax.set_mesh(mesh):
            model = GraniteForCausalLM(cfg, rng, mesh)
        hf_config = cfg.model_config.hf_config

        hidden_size = hf_config.hidden_size
        vocab = hf_config.vocab_size
        hidden = jnp.ones((1, hidden_size), dtype=jnp.bfloat16)

        logits = model.compute_logits(hidden)

        # Recompute the unscaled logits directly and compare.
        lm_head = model.model.lm_head.value
        raw = jnp.dot(hidden, lm_head.T)
        expected = raw / hf_config.logits_scaling
        np.testing.assert_allclose(np.asarray(logits, dtype=np.float32),
                                   np.asarray(expected, dtype=np.float32),
                                   rtol=1e-2,
                                   atol=1e-2)
        assert logits.shape == (1, vocab)
