from datetime import timedelta
from functools import partial
from contextlib import contextmanager
import os
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh


_HYBRID_DEVICE_MESH = None
_HYBRID_DEVICE_MESH_SHAPE = None


def get_hybrid_device_mesh():
    """Return one cached (replicate, shard) mesh for multi-node HSDP."""
    global _HYBRID_DEVICE_MESH, _HYBRID_DEVICE_MESH_SHAPE
    if not dist.is_initialized():
        raise RuntimeError("distributed process group must be initialized first")
    world_size = dist.get_world_size()
    if _HYBRID_DEVICE_MESH is not None:
        if "LOCAL_WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "HYBRID_SHARD requires LOCAL_WORLD_SIZE; launch with torchrun"
            )
        current_shape = (
            world_size // int(os.environ["LOCAL_WORLD_SIZE"]),
            int(os.environ["LOCAL_WORLD_SIZE"]),
        )
        if _HYBRID_DEVICE_MESH_SHAPE != current_shape:
            raise RuntimeError(
                "Distributed topology changed after the hybrid device mesh "
                f"was created: {_HYBRID_DEVICE_MESH_SHAPE} -> {current_shape}"
            )
        return _HYBRID_DEVICE_MESH
    if "LOCAL_WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "HYBRID_SHARD requires LOCAL_WORLD_SIZE; launch with torchrun"
        )
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    reported_local_world_sizes = [None] * world_size
    dist.all_gather_object(reported_local_world_sizes, local_world_size)
    if len(set(reported_local_world_sizes)) != 1:
        raise RuntimeError(
            "LOCAL_WORLD_SIZE differs across ranks: "
            f"{reported_local_world_sizes}"
        )
    if local_world_size <= 0 or world_size % local_world_size:
        raise RuntimeError(
            "HYBRID_SHARD requires world_size to be divisible by "
            f"LOCAL_WORLD_SIZE, got {world_size} and {local_world_size}"
        )
    mesh_shape = (world_size // local_world_size, local_world_size)
    _HYBRID_DEVICE_MESH = init_device_mesh(
        "cuda",
        mesh_shape,
        mesh_dim_names=("replicate", "shard"),
    )
    _HYBRID_DEVICE_MESH_SHAPE = mesh_shape
    return _HYBRID_DEVICE_MESH


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    has_meta_parameters = any(
        parameter.is_meta for parameter in module.parameters()
    )
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    device_mesh = None
    if sharding_strategy in (
        ShardingStrategy.HYBRID_SHARD,
        ShardingStrategy._HYBRID_SHARD_ZERO2,
    ):
        device_mesh = get_hybrid_device_mesh()

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_mesh=device_mesh,
        param_init_fn=(
            (
                lambda submodule: submodule.to_empty(
                    device=torch.cuda.current_device(), recurse=False
                )
            )
            if has_meta_parameters
            else None
        ),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    """
    Initialize the distributed training environment.
    When launched via torchrun, environment variables are already set by torchrun,
    so we use a concise initialization and let PyTorch handle rendezvous automatically.
    """
    # Concise initialization (ref. Bogel), letting torchrun handle MASTER_ADDR/PORT
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


class FSDPTrainState(Stateful):
    """DCP adapter for sharded generator/critic and optimizer state."""

    def __init__(
        self,
        generator,
        generator_optimizer,
        critic,
        critic_optimizer,
        bidirectional_cd_ema=None,
    ):
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.bidirectional_cd_ema = bidirectional_cd_ema
        self.options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            strict=True,
        )

    def state_dict(self):
        generator_state, generator_optim_state = get_state_dict(
            self.generator,
            self.generator_optimizer,
            options=self.options,
        )
        critic_state, critic_optim_state = get_state_dict(
            self.critic,
            self.critic_optimizer,
            options=self.options,
        )
        state = {
            "generator": generator_state,
            "generator_optimizer": generator_optim_state,
            "critic": critic_state,
            "critic_optimizer": critic_optim_state,
        }
        if self.bidirectional_cd_ema is not None:
            state["bidirectional_cd_ema"] = get_model_state_dict(
                self.bidirectional_cd_ema, options=self.options
            )
        return state

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.generator,
            self.generator_optimizer,
            model_state_dict=state_dict["generator"],
            optim_state_dict=state_dict["generator_optimizer"],
            options=self.options,
        )
        set_state_dict(
            self.critic,
            self.critic_optimizer,
            model_state_dict=state_dict["critic"],
            optim_state_dict=state_dict["critic_optimizer"],
            options=self.options,
        )
        if self.bidirectional_cd_ema is not None:
            if "bidirectional_cd_ema" not in state_dict:
                raise RuntimeError(
                    "Checkpoint lacks the required bidirectional_cd_ema state"
                )
            set_model_state_dict(
                self.bidirectional_cd_ema,
                state_dict["bidirectional_cd_ema"],
                options=self.options,
            )


class FSDPModelState(Stateful):
    """DCP adapter for model-only sharded state (used by EMA)."""

    def __init__(self, model):
        self.model = model
        self.options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            strict=True,
        )

    def state_dict(self):
        return get_model_state_dict(self.model, options=self.options)

    def load_state_dict(self, state_dict):
        set_model_state_dict(
            self.model,
            state_dict,
            options=self.options,
        )


class EMA_FSDP:
    """EMA over each rank's local FSDP parameter shards.

    This avoids full-parameter all-gathers and avoids keeping one complete CPU
    model per rank. DCP serialization uses a temporary local parameter swap so
    the existing FSDP model emits a normal reshardable state dict.
    """
    @staticmethod
    def _clean_name(name: str) -> str:
        """Remove prefixes introduced by FSDP / activation-checkpoint / torch.compile to keep keys consistent with the original model"""
        return (
            name.replace("_fsdp_wrapped_module.", "")
                .replace("_checkpoint_wrapped_module.", "")
                .replace("_orig_mod.", "")
        )

    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        decay: float = 0.999,
        device: str | torch.device = "cpu",
    ):
        self.decay = decay
        self.device = torch.device(device)
        self.shadow = {}
        self._init_shadow(fsdp_module)

    def _named_trainable_parameters(self, fsdp_module):
        for name, parameter in fsdp_module.named_parameters():
            if parameter.requires_grad:
                yield self._clean_name(name), parameter

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        self.shadow = {
            name: parameter.detach().to(
                device=self.device, dtype=torch.float32, copy=True
            )
            for name, parameter in self._named_trainable_parameters(fsdp_module)
        }

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        for name, parameter in self._named_trainable_parameters(fsdp_module):
            source = parameter.detach().to(
                device=self.device, dtype=torch.float32
            )
            if name not in self.shadow or self.shadow[name].shape != source.shape:
                raise RuntimeError(
                    f"EMA shard mismatch for {name}: checkpoint/model topology changed"
                )
            self.shadow[name].mul_(d).add_(source, alpha=1. - d)

    @torch.no_grad()
    def capture(self, fsdp_module):
        """Replace EMA shards with the model's current local shards."""
        self._init_shadow(fsdp_module)

    @contextmanager
    def preserve_model_parameters(self, fsdp_module):
        """Restore live local shards after a temporary DCP EMA load/swap."""
        with torch.no_grad():
            backups = {
                name: parameter.detach().to(device="cpu", copy=True)
                for name, parameter in self._named_trainable_parameters(fsdp_module)
            }
            try:
                yield
            finally:
                for name, parameter in self._named_trainable_parameters(fsdp_module):
                    parameter.copy_(
                        backups[name].to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    )

    @contextmanager
    def apply_to(self, fsdp_module):
        """Temporarily expose EMA shards through the live FSDP module."""
        with torch.no_grad():
            with self.preserve_model_parameters(fsdp_module):
                for name, parameter in self._named_trainable_parameters(fsdp_module):
                    parameter.copy_(
                        self.shadow[name].to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    )
                yield

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {
            self._clean_name(k): v.to(
                device=self.device, dtype=torch.float32, copy=True
            )
            for k, v in sd.items()
        }

    @torch.no_grad()
    def copy_to(self, fsdp_module):
        for name, parameter in self._named_trainable_parameters(fsdp_module):
            parameter.copy_(
                self.shadow[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
            )
