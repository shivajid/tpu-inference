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

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import tpu_inference.env_override  # noqa: F401
from tpu_inference import envs

# Engine-first XLA load: import Raiden's engine extension here -- the earliest
# tpu_inference entry point, before any submodule (e.g. platforms/tpu_platform.py)
# runs `import jax` and loads jaxlib's libjax_common.so. This ensures Raiden's
# embedded XLA runtime comes up before jaxlib's so the two XLA copies don't collide
# in static initializers.
try:
    import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
except ModuleNotFoundError:
    # Expected in processes without tpu_raiden on the path (e.g. the benchmark
    # client). Only the EngineCore workers need the engine-first load; skip quietly.
    pass
except Exception as _raiden_exc:  # pragma: no cover - best-effort preload
    import sys as _sys
    print(f"[tpu_raiden] engine preload failed: {_raiden_exc}",
          file=_sys.stderr)

from tpu_inference import tpu_info as ti
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

if "proxy" in envs.JAX_PLATFORMS:
    logger.info("Running vLLM on TPU via Pathways proxy.")
    # Must run pathwaysutils.initialize() before any JAX operations
    try:
        import traceback

        import pathwaysutils
        import vllm
        from vllm.platforms import (resolve_current_platform_cls_qualname,
                                    resolve_obj_by_qualname)
        pathwaysutils.initialize()
        logger.info("Module pathwaysutils is imported.")

        # Pathways requires eager resolution of vllm.current_platform instead of
        # lazy resolution in the normal code path. Since this part involves
        # global topology discovery across multiple hosts, the platform
        # resolution must happen before other components are loaded.
        logger.info("Eagerly resolving vLLM current_platform for Pathways.")
        platform_cls_qualname = resolve_current_platform_cls_qualname()
        resolved_platform_instance = resolve_obj_by_qualname(
            platform_cls_qualname)()
        vllm.platforms._current_platform = resolved_platform_instance
        vllm.platforms._init_trace = "".join(traceback.format_stack())
        logger.info(
            f"vLLM platform resolved to: {resolved_platform_instance.__class__.__name__}"
        )

    except Exception as e:
        logger.error(
            f"Error occurred while importing pathwaysutils or logging TPU info: {e}"
        )
else:
    # Either running on TPU or CPU
    try:
        logger.info(f"TPU info: node_name={ti.get_node_name()} | "
                    f"tpu_type={ti.get_tpu_type()} | "
                    f"worker_id={ti.get_node_worker_id()} | "
                    f"num_chips={ti.get_num_chips()} | "
                    f"num_cores_per_chip={ti.get_num_cores_per_chip()}")
    except Exception as e:
        logger.error(
            f"Error occurred while logging TPU info: {e}. Are you running on CPU?"
        )


def _register_vllm_compat_archs():
    """Register TPU-only architectures with vLLM's ModelRegistry.

    The `vllm.general_plugins` entrypoint (register_layers) fires before
    engine-config creation in the offline LLM path, but NOT in the API-server
    process — which still validates ModelConfig against the registry. This
    module is imported in every vLLM process via platform resolution (before
    config creation), so registering here covers the API server too. Lazy
    string references keep the import cost negligible.
    """
    try:
        from vllm import ModelRegistry
        compat_archs = {
            "JinaBertForMaskedLM":
            "tpu_inference.models.vllm.jina_bert_compat:JinaBertForMaskedLM",
            "JinaBertModel":
            "tpu_inference.models.vllm.jina_bert_compat:JinaBertForMaskedLM",
            "ElectraForPreTraining":
            "tpu_inference.models.vllm.electra_compat:ElectraForPreTraining",
            "ElectraModel":
            "tpu_inference.models.vllm.electra_compat:ElectraForPreTraining",
        }
        existing = set(ModelRegistry.get_supported_archs())
        for arch, qualname in compat_archs.items():
            if arch not in existing:
                ModelRegistry.register_model(arch, qualname)
    except Exception as e:  # pragma: no cover - best effort, plugin path
        logger.debug(f"Skipping vLLM compat arch registration: {e}")


_register_vllm_compat_archs()
