import gc
import hashlib
import logging
import json
import datetime
import sys
import shutil
import uuid
import random
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
import numpy as np

from utils.dataset import (
    PDPairDataset,
    ResumableDistributedSampler,
    SQLiteT2VVideoDataset,
    ShardingLMDBDataset,
    cycle,
)
from utils.dataset import TextDataset
from utils.distributed import (
    EMA_FSDP,
    FSDPModelState,
    FSDPTrainState,
    fsdp_wrap,
    launch_distributed_job,
)
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
    set_state_dict,
)
from omegaconf import OmegaConf
from model import DMD, TimestepForcingDMD
import torch
try:
    import wandb
except ModuleNotFoundError:
    wandb = None
import time
import os
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from pipeline import CausalInferencePipeline


def _write_video_mp4(filename, video_array, fps):
    """Write an RGB uint8 THWC tensor without torchvision/PyAV ABI coupling.

    torchvision 0.20 assigns the string ``"NONE"`` to ``VideoFrame.pict_type``;
    PyAV 17 requires an integer enum and raises during every validation write.
    Encoding directly through PyAV is both shorter and compatible with the
    runtime shipped on the training nodes.
    """
    import av

    video = (
        torch.as_tensor(video_array, dtype=torch.uint8)
        .detach()
        .cpu()
        .contiguous()
        .numpy()
    )
    if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] == 0:
        raise ValueError(
            "video_array must be a non-empty uint8 tensor with shape [T,H,W,3], "
            f"got {video.shape}"
        )
    rate = Fraction(str(fps)).limit_denominator(1000)
    if rate <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    with av.open(filename, mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = int(video.shape[2])
        stream.height = int(video.shape[1])
        stream.pix_fmt = "yuv420p"
        for image in video:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class TeeLogger:
    """Tee stdout to both the console and a log file for persistent logging.

    Usage: sys.stdout = TeeLogger(log_file_path)
    After this, all print() output will be written to both the console and the file.
    """
    def __init__(self, log_path, mode='a'):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, mode, buffering=1)  # line-buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def fileno(self):
        return self.terminal.fileno()

    def isatty(self):
        # PyTorch Dynamo/Fx checks this while formatting compiler diagnostics.
        return self.terminal.isatty()

    def __getattr__(self, name):
        # Preserve the rest of the TextIO interface (encoding, errors,
        # writable, etc.) for libraries that inspect sys.stdout/sys.stderr.
        return getattr(self.terminal, name)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        if not getattr(config, "logdir", ""):
            raise ValueError("--logdir must be a non-empty path")
        self.output_path = os.path.abspath(os.path.expanduser(config.logdir))
        config.logdir = self.output_path

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if not self.disable_wandb and wandb is None:
            raise RuntimeError(
                "Weights & Biases logging was requested but wandb is not "
                "installed; install it or pass --disable-wandb"
            )
        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        configured_checkpoint_path = getattr(config, "checkpoint_dir", None)
        configured_eval_path = getattr(config, "eval_dir", None)
        self.checkpoint_path = os.path.abspath(os.path.expanduser(
            configured_checkpoint_path
            or os.path.join(self.output_path, "checkpoints")
        ))
        self.eval_path = os.path.abspath(os.path.expanduser(
            configured_eval_path
            or os.path.join(self.output_path, "evaluation")
        ))
        legacy_interval = int(getattr(config, "log_iters", 100))
        self.log_interval = int(getattr(config, "log_interval", 1))
        self.checkpoint_interval = int(
            getattr(config, "checkpoint_interval", legacy_interval)
        )
        self.eval_interval = int(getattr(config, "eval_interval", legacy_interval))
        self.max_checkpoints_to_keep = int(
            getattr(config, "max_checkpoints_to_keep", 5)
        )
        for name, value in (
            ("log_interval", self.log_interval),
            ("checkpoint_interval", self.checkpoint_interval),
            ("eval_interval", self.eval_interval),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.max_checkpoints_to_keep < 0:
            raise ValueError(
                "max_checkpoints_to_keep must be >= 0, got "
                f"{self.max_checkpoints_to_keep}"
            )

        # Initialize log file (before TensorBoard, to start recording as early as possible)
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            log_file_path = os.path.join(self.output_path, "train.log")
            sys.stdout = TeeLogger(log_file_path, mode='a')
            sys.stderr = TeeLogger(os.path.join(self.output_path, "train_stderr.log"), mode='a')
            print(f"\n{'='*60}")
            print(f"Log session started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"All stdout -> {log_file_path}")
            print(f"{'='*60}")

        # Initialize TensorBoard
        if self.is_main_process:
            tb_log_dir = os.path.join(config.logdir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"TensorBoard log directory: {tb_log_dir}")
        else:
            self.tb_writer = None

        # Step 2: Initialize the model and optimizer
        requested_bidirectional_cd = bool(
            getattr(config, "bidirectional_cd_enabled", False)
        )
        if requested_bidirectional_cd:
            if config.distribution_loss != "timestep_forcing_dmd":
                raise ValueError(
                    "bidirectional_cd_enabled requires "
                    "distribution_loss=timestep_forcing_dmd"
                )
            cd_num_timesteps = int(
                getattr(config, "bidirectional_cd_num_timesteps", 48)
            )
            cd_shift = float(
                getattr(config, "bidirectional_cd_timestep_shift", 5.0)
            )
            cd_loss_weight = float(
                getattr(config, "bidirectional_cd_loss_weight", 1.0)
            )
            cd_ema_decay = float(
                getattr(config, "bidirectional_cd_ema_decay", 0.99)
            )
            cd_sft_weight = float(
                getattr(config, "bidirectional_cd_sft_weight", 0.0)
            )
            cd_guidance = float(
                getattr(config, "bidirectional_cd_guidance_scale", 3.0)
            )
            if cd_num_timesteps < 2:
                raise ValueError("bidirectional_cd_num_timesteps must be >= 2")
            if not np.isfinite(cd_shift) or cd_shift <= 0:
                raise ValueError(
                    "bidirectional_cd_timestep_shift must be finite and > 0"
                )
            if not np.isfinite(cd_guidance) or cd_guidance < 0:
                raise ValueError(
                    "bidirectional_cd_guidance_scale must be finite and >= 0"
                )
            if not np.isfinite(cd_loss_weight) or cd_loss_weight < 0:
                raise ValueError(
                    "bidirectional_cd_loss_weight must be finite and >= 0"
                )
            if not 0.0 <= cd_ema_decay < 1.0:
                raise ValueError(
                    "bidirectional_cd_ema_decay must be in [0, 1)"
                )
            if not np.isfinite(cd_sft_weight) or cd_sft_weight < 0:
                raise ValueError(
                    "bidirectional_cd_sft_weight must be finite and >= 0"
                )
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "timestep_forcing_dmd":
            self.model = TimestepForcingDMD(config, device=self.device)
        else:
            raise ValueError(f"Invalid distribution_loss: {config.distribution_loss}")
        self.bidirectional_cd_enabled = requested_bidirectional_cd

        # Count model parameters
        if self.is_main_process:
            def _count_params(m):
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                return total, trainable
            gen_total, gen_train = _count_params(self.model.generator)
            fs_total, fs_train = _count_params(self.model.fake_score)
            rs_total, _ = _count_params(self.model.real_score)
            print(f"Model Parameter Summary:")
            print(f"  Generator : {gen_total/1e6:.1f}M total, {gen_train/1e6:.1f}M trainable")
            print(f"  Fake Score: {fs_total/1e6:.1f}M total, {fs_train/1e6:.1f}M trainable")
            print(f"  Real Score: {rs_total/1e6:.1f}M total (frozen)")
            if self.bidirectional_cd_enabled:
                cd_total, _ = _count_params(self.model.bidirectional_cd_ema)
                print(
                    f"  Bi-CD EMA : {cd_total/1e6:.1f}M total (frozen target)"
                )

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )
        if self.bidirectional_cd_enabled:
            self.model.bidirectional_cd_ema = fsdp_wrap(
                self.model.bidirectional_cd_ema,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
            )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        configured_rollout = int(getattr(config, "num_rollout_frames", 0))
        effective_rollout = (
            configured_rollout
            if configured_rollout > 0
            else int(getattr(config, "num_training_frames", 21))
        )
        needs_vae = (
            not config.no_visualize
            or config.load_raw_video
            or effective_rollout > int(getattr(config, "num_training_frames", 21))
            or bool(getattr(config, "i2v", False))
            or self.bidirectional_cd_enabled
        )
        if needs_vae:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.bidirectional_cd_enabled:
            dataset_factory = lambda: SQLiteT2VVideoDataset(
                index_path=config.data_path,
                num_frames=int(config.num_frames),
                height=int(config.height),
                width=int(config.width),
                seed=int(config.seed),
                target_fps=float(
                    getattr(config, "data_target_fps", 0.0)
                ),
                prompt_cache_size=int(
                    getattr(config, "data_prompt_cache_size", 128)
                ),
                decode_video=True,
                align_prompt_segments=bool(
                    getattr(config, "data_align_prompt_segments", True)
                ),
                exclude_metadata_patterns=list(
                    getattr(config, "data_exclude_metadata_patterns", [])
                ),
            )
        elif self.config.i2v:
            dataset_factory = lambda: ShardingLMDBDataset(
                config.data_path, max_pair=int(1e8)
            )
        else:
            dataset_factory = lambda: TextDataset(config.data_path)
        dataset = self._build_dataset_collectively(
            dataset_factory, "training dataset"
        )
        sampler = ResumableDistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            seed=int(config.seed),
        )
        num_workers = int(
            getattr(
                config,
                "data_num_workers",
                0 if not config.load_raw_video else 2,
            )
        )
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=bool(
                getattr(config, "data_pin_memory", self.bidirectional_cd_enabled)
            ),
            persistent_workers=num_workers > 0,
        )
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = int(
                getattr(config, "data_prefetch_factor", 1)
            )
        dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        if len(dataloader) == 0:
            raise RuntimeError(
                f"Training dataset ({len(dataset)} items) is too small for "
                f"world_size={self.world_size}, batch_size={config.batch_size}, drop_last=True"
            )
        self._generator_data_loader = dataloader
        self.generator_batches_consumed = 0
        self.critic_batches_consumed = 0
        self.pd_batches_consumed = 0
        self.generator_update_count = 0
        self.bidirectional_cd_ema_update_count = 0
        self.dataloader = cycle(self._generator_data_loader)
        self.critic_dataloader = self.dataloader
        self._critic_data_loader = None

        if self.bidirectional_cd_enabled:
            # DMD critic training only uses prompts. Keep its input stream
            # independent so it never decodes/pins real video frames that the
            # critic path discards.
            critic_dataset = self._build_dataset_collectively(
                lambda: SQLiteT2VVideoDataset(
                    index_path=config.data_path,
                    num_frames=int(config.num_frames),
                    height=int(config.height),
                    width=int(config.width),
                    seed=int(config.seed) + 1,
                    target_fps=float(
                        getattr(config, "data_target_fps", 0.0)
                    ),
                    prompt_cache_size=int(
                        getattr(config, "data_prompt_cache_size", 128)
                    ),
                    decode_video=False,
                    align_prompt_segments=bool(
                        getattr(config, "data_align_prompt_segments", True)
                    ),
                    exclude_metadata_patterns=list(
                        getattr(config, "data_exclude_metadata_patterns", [])
                    ),
                ),
                "critic prompt dataset",
            )
            critic_sampler = ResumableDistributedSampler(
                critic_dataset,
                shuffle=True,
                drop_last=True,
                seed=int(config.seed) + 1,
            )
            # This loader only returns prompt strings, so subprocess workers do
            # not buy us meaningful throughput.  More importantly, its iterator
            # is first created after the generator has initialized CUDA/NCCL and
            # torch.compile; forking workers from that multithreaded process is
            # unsafe and can segfault.  Keep it synchronous by default while
            # leaving an explicit opt-in for spawn-safe environments.
            critic_num_workers = int(
                getattr(config, "critic_data_num_workers", 0)
            )
            critic_loader_kwargs = dict(
                dataset=critic_dataset,
                batch_size=config.batch_size,
                sampler=critic_sampler,
                drop_last=True,
                num_workers=critic_num_workers,
                pin_memory=False,
                persistent_workers=critic_num_workers > 0,
            )
            if critic_num_workers > 0:
                critic_loader_kwargs["prefetch_factor"] = int(
                    getattr(
                        config,
                        "critic_data_prefetch_factor",
                        getattr(config, "data_prefetch_factor", 1),
                    )
                )
            critic_dataloader = torch.utils.data.DataLoader(
                **critic_loader_kwargs
            )
            if len(critic_dataloader) == 0:
                raise RuntimeError(
                    "Critic prompt dataset is too small for the configured "
                    "world size and batch size"
                )
            self.critic_dataloader = cycle(critic_dataloader)
            self._critic_data_loader = critic_dataloader

        # Step 3b: Initialize PD dataloader (if Progressive Distillation is enabled)
        self.pd_enabled = getattr(config, 'pd_enabled', False)
        if self.pd_enabled and self.bidirectional_cd_enabled:
            raise ValueError(
                "pd_enabled and bidirectional_cd_enabled are mutually exclusive"
            )
        self.pd_dataloader = None
        self._pd_data_loader = None
        if self.pd_enabled:
            pd_data_path = getattr(config, 'pd_data_path', '')
            def build_pd_dataset():
                if not pd_data_path or not os.path.isdir(pd_data_path):
                    raise FileNotFoundError(
                        "pd_enabled=true but pd_data_path="
                        f"{pd_data_path!r} is not a valid directory"
                    )
                pd_dataset = PDPairDataset(
                    pd_data_path,
                    require_manifest=bool(
                        getattr(config, "pd_require_manifest", True)
                    ),
                )
                if pd_dataset.manifest is not None:
                    actual_sigmas = np.asarray(
                        pd_dataset.manifest["aligned_sigmas"], dtype=np.float64
                    )
                    expected_sigmas = np.asarray(
                        self.model.pd_aligned_sigmas, dtype=np.float64
                    )
                    if (
                        actual_sigmas.shape != expected_sigmas.shape
                        or not np.allclose(
                            actual_sigmas,
                            expected_sigmas,
                            rtol=0.0,
                            atol=1.0e-6,
                        )
                    ):
                        raise ValueError(
                            "PD ODE pairs are not aligned with the student "
                            f"schedule: manifest={actual_sigmas.tolist()}, "
                            f"expected={expected_sigmas.tolist()}"
                        )
                return pd_dataset

            pd_dataset = self._build_dataset_collectively(
                build_pd_dataset, "PD dataset"
            )
            pd_sampler = ResumableDistributedSampler(
                pd_dataset,
                shuffle=True,
                drop_last=True,
                seed=int(config.seed) + 2,
            )
            pd_dataloader = torch.utils.data.DataLoader(
                pd_dataset,
                batch_size=config.batch_size,
                sampler=pd_sampler,
                drop_last=True,
                num_workers=0)
            if len(pd_dataloader) == 0:
                raise RuntimeError(
                    f"PD dataset ({len(pd_dataset)} pairs) is too small for "
                    f"world_size={self.world_size}, batch_size={config.batch_size}, drop_last=True"
                )
            self._pd_data_loader = pd_dataloader
            self.pd_dataloader = cycle(self._pd_data_loader)
            if dist.get_rank() == 0:
                print(f"PD DATASET SIZE {len(pd_dataset)} (from {pd_data_path})")
                print(f"PD loss weight: {getattr(config, 'pd_loss_weight', 1.0)}")
                print(f"PD num steps: {getattr(config, 'pd_num_steps', 0)}")
                print(f"PD loss type: {getattr(config, 'pd_loss_type', 'euler_endpoint')}")

        ##############################################################################################################
        # 6. Keep one local EMA shard per rank. It is created before resume
        # so a sharded EMA checkpoint can be loaded without a full all-gather.
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (
            (ema_weight is not None)
            and (ema_weight > 0.0)
            and (config.ema_start_step <= 0)
        ):
            self.generator_ema = EMA_FSDP(
                self.model.generator,
                decay=ema_weight,
                device=getattr(config, "ema_device", "cpu"),
            )

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        
        # ============================================================
        # Weight loading logic (three-level priority):
        #   1. Resume checkpoint (fully restore training state)
        #   2. generator_ckpt (load only generator pretrained weights, start from step 0)
        #   3. Start from random initialization (not recommended, usually a config error)
        # ============================================================
        resume_path = self._resolve_resume_path(getattr(config, "resume_from", None))
        checkpoint_loaded = False

        if resume_path:
            if self.is_main_process:
                print(f"Resuming training from checkpoint: {resume_path}")
            if os.path.isdir(resume_path):
                self._load_sharded_checkpoint(resume_path)
            elif os.path.isfile(resume_path):
                self._load_legacy_checkpoint(resume_path)
            else:
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            checkpoint_loaded = True
            torch.cuda.empty_cache()

        # Fallback: load generator pretrained weights (not a full resume)
        if not checkpoint_loaded and getattr(config, "generator_ckpt", False):
            if self.is_main_process:
                print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = None
            load_error = None
            if self.is_main_process:
                try:
                    state_dict = torch.load(
                        config.generator_ckpt,
                        map_location="cpu",
                        weights_only=False,
                    )
                    if (
                        "generator" in state_dict
                        and isinstance(state_dict["generator"], dict)
                        and state_dict["generator"]
                    ):
                        state_dict = state_dict["generator"]
                    elif (
                        "model" in state_dict
                        and isinstance(state_dict["model"], dict)
                        and state_dict["model"]
                    ):
                        state_dict = state_dict["model"]
                    elif (
                        "generator_ema" in state_dict
                        and isinstance(state_dict["generator_ema"], dict)
                        and state_dict["generator_ema"]
                    ):
                        state_dict = state_dict["generator_ema"]
                    if not isinstance(state_dict, dict):
                        raise TypeError(
                            "generator_ckpt must contain a model state dictionary"
                        )
                except Exception as exc:
                    load_error = f"{type(exc).__name__}: {exc}"
            payload = [load_error]
            dist.broadcast_object_list(payload, src=0)
            if payload[0] is not None:
                raise RuntimeError(payload[0])
            set_model_state_dict(
                self.model.generator,
                state_dict if self.is_main_process else {},
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                    strict=True,
                    broadcast_from_rank0=True,
                ),
            )
            if self.bidirectional_cd_enabled:
                set_model_state_dict(
                    self.model.bidirectional_cd_ema,
                    state_dict if self.is_main_process else {},
                    options=StateDictOptions(
                        full_state_dict=True,
                        cpu_offload=True,
                        strict=True,
                        broadcast_from_rank0=True,
                    ),
                )
                self._assert_bidirectional_cd_target_matches_student()
            if self.generator_ema is not None:
                self.generator_ema.capture(self.model.generator)
            checkpoint_loaded = True
            if self.is_main_process:
                print(f"  - Generator weights loaded, training from step 0")
        
        if not checkpoint_loaded and self.is_main_process:
            print("WARNING: No checkpoint or generator_ckpt loaded! Training from random initialization.")

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        # Compute gradient accumulation steps
        # total_batch_size = batch_size * world_size * gradient_accumulation_steps
        effective_batch_per_step = config.batch_size * self.world_size
        if hasattr(config, 'gradient_accumulation_steps') and config.gradient_accumulation_steps > 0:
            self.gradient_accumulation_steps = config.gradient_accumulation_steps
        elif hasattr(config, 'total_batch_size') and config.total_batch_size > effective_batch_per_step:
            if config.total_batch_size % effective_batch_per_step:
                raise ValueError(
                    f"total_batch_size ({config.total_batch_size}) must be "
                    f"divisible by batch_size*world_size ({effective_batch_per_step})"
                )
            self.gradient_accumulation_steps = config.total_batch_size // effective_batch_per_step
        else:
            self.gradient_accumulation_steps = 1
        
        if self.is_main_process:
            print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
            print(f"Effective batch size: {config.batch_size * self.world_size * self.gradient_accumulation_steps}")

        # ============================================================
        # Save full config and training metadata to output directory
        # ============================================================
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            # Save merged full config (YAML format for readability)
            config_save_path = os.path.join(self.output_path, "config.yaml")
            OmegaConf.save(config, config_save_path)
            print(f"Full config saved to {config_save_path}")
            
            # Save training metadata (JSON format for programmatic access)
            train_meta = {
                "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "start_step": self.step,
                "world_size": self.world_size,
                "batch_size_per_gpu": config.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "effective_batch_size": config.batch_size * self.world_size * self.gradient_accumulation_steps,
                "distribution_loss": config.distribution_loss,
                "lr_generator": config.lr,
                "lr_critic": getattr(config, 'lr_critic', config.lr),
                "denoising_step_list": config.denoising_step_list,
                "num_training_frames": getattr(config, 'num_training_frames', 21),
                "num_rollout_frames": getattr(config, 'num_rollout_frames', 0),
                "fixed_rollout_frames": getattr(config, 'fixed_rollout_frames', 0),
                "num_frame_per_block": getattr(config, 'num_frame_per_block', 1),
                "mixed_precision": config.mixed_precision,
                "sharding_strategy": config.sharding_strategy,
                "resume_from": getattr(config, 'resume_from', None),
                "generator_ckpt": getattr(config, 'generator_ckpt', None),
                "pd_enabled": getattr(config, 'pd_enabled', False),
                "pd_num_steps": getattr(config, 'pd_num_steps', 0),
                "denoising_order": getattr(config, 'denoising_order', 'timestep_first'),
            }
            meta_save_path = os.path.join(self.output_path, "train_meta.json")
            with open(meta_save_path, "w") as f:
                json.dump(train_meta, f, indent=2, default=str)
            
            # Print training configuration summary
            print("\n" + "=" * 60)
            print("Training Configuration Summary")
            print("=" * 60)
            print(f"  Distribution Loss : {config.distribution_loss}")
            print(f"  Denoising Steps   : {config.denoising_step_list}")
            print(f"  Denoising Order   : {getattr(config, 'denoising_order', 'timestep_first')}")
            print(f"  World Size        : {self.world_size}")
            print(f"  Batch / GPU       : {config.batch_size}")
            print(f"  Grad Accum Steps  : {self.gradient_accumulation_steps}")
            print(f"  Effective Batch   : {config.batch_size * self.world_size * self.gradient_accumulation_steps}")
            print(f"  LR (generator)    : {config.lr}")
            print(f"  LR (critic)       : {getattr(config, 'lr_critic', config.lr)}")
            print(f"  EMA Weight        : {config.ema_weight} (start at step {config.ema_start_step})")
            print(f"  Mixed Precision   : {config.mixed_precision}")
            print(f"  Num Frames        : {getattr(config, 'num_training_frames', 21)}")
            _rf = getattr(config, 'num_rollout_frames', 0)
            _effective_rollout = _rf if _rf > 0 else getattr(config, 'num_training_frames', 21)
            print(f"  Rollout Frames    : {_effective_rollout}")
            _fixed_rollout = int(getattr(config, 'fixed_rollout_frames', 0))
            print(
                "  Fixed Rollout     : "
                + (str(_fixed_rollout) if _fixed_rollout > 0 else "disabled")
            )
            print(f"  Frames / Block    : {getattr(config, 'num_frame_per_block', 1)}")
            print(f"  Log Interval      : {self.log_interval} steps")
            print(f"  Checkpoint Interval: {self.checkpoint_interval} steps")
            print(f"  Eval Interval     : {self.eval_interval} steps")
            print(f"  Gen/Critic Ratio  : 1:{config.dfake_gen_update_ratio}")
            if getattr(config, 'pd_enabled', False):
                print(f"  PD Enabled        : True")
                print(f"  PD Num Steps      : {getattr(config, 'pd_num_steps', 0)}")
                print(f"  PD Loss Type      : {getattr(config, 'pd_loss_type', 'euler_endpoint')}")
                print(f"  PD Loss Weight    : {getattr(config, 'pd_loss_weight', 1.0)}")
                print(f"  PD Compute Mode   : {getattr(config, 'pd_compute_mode', 'single_forward')}")
            print(f"  Starting Step     : {self.step}")
            print(f"  Output Dir        : {self.output_path}")
            print(f"  Checkpoint Dir    : {self.checkpoint_path}")
            print(f"  Evaluation Dir    : {self.eval_path}")
            print("=" * 60 + "\n")

    @staticmethod
    def _clean_fsdp_name(name: str) -> str:
        return (
            name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )

    def _named_parameter_map(self, module):
        return {
            self._clean_fsdp_name(name): parameter
            for name, parameter in module.named_parameters()
        }

    def _bidirectional_cd_checkpoint_config(self):
        if not self.bidirectional_cd_enabled:
            return None
        return {
            "num_timesteps": int(
                getattr(self.config, "bidirectional_cd_num_timesteps", 48)
            ),
            "timestep_shift": float(
                getattr(self.config, "bidirectional_cd_timestep_shift", 5.0)
            ),
            "guidance_scale": float(
                getattr(self.config, "bidirectional_cd_guidance_scale", 3.0)
            ),
            "loss_weight": float(
                getattr(self.config, "bidirectional_cd_loss_weight", 1.0)
            ),
            "ema_decay": float(
                getattr(self.config, "bidirectional_cd_ema_decay", 0.99)
            ),
            "sft_weight": float(
                getattr(self.config, "bidirectional_cd_sft_weight", 0.0)
            ),
        }

    def _training_semantics_config(self):
        """Canonical config subset that must not change across resume."""
        config = OmegaConf.to_container(self.config, resolve=True)
        mutable_keys = {
            "checkpoint_dir",
            "checkpoint_interval",
            "config_name",
            "allow_data_restart_on_reshard",
            "disable_wandb",
            "ema_device",
            "eval_dir",
            "eval_fps",
            "eval_interval",
            "eval_num_frames",
            "eval_num_prompts",
            "generator_ckpt",
            "log_interval",
            "logdir",
            "max_checkpoints_to_keep",
            "max_steps",
            "no_save",
            "no_visualize",
            "resume_from",
            "wandb_entity",
            "wandb_host",
            "wandb_key",
            "wandb_project",
            "wandb_save_dir",
        }
        # These settings are covered by the dataset records digest stored in
        # data_stream_identity. Keeping them out of the config fingerprint also
        # permits legacy checkpoints to resume after filters became explicit.
        data_identity_covered_keys = {
            "data_exclude_metadata_patterns",
        }
        semantics = {
            key: value
            for key, value in config.items()
            if key not in mutable_keys
            and key not in data_identity_covered_keys
        }
        data_path = semantics.get("data_path")
        if data_path:
            real_path = os.path.realpath(os.path.expanduser(str(data_path)))
            stat_result = os.stat(real_path)
            semantics["data_path_identity"] = {
                "realpath": real_path,
                "size": int(stat_result.st_size),
                "mtime_ns": int(stat_result.st_mtime_ns),
                "inode": int(stat_result.st_ino),
            }
        return semantics

    def _training_semantics_fingerprint(self):
        payload = json.dumps(
            self._training_semantics_config(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _loader_identity(loader):
        if loader is None:
            return None
        dataset = loader.dataset
        return {
            "dataset_length": int(len(dataset)),
            "records_digest": getattr(dataset, "records_digest", None),
            "batches_per_epoch": int(len(loader)),
        }

    def _data_stream_identity(self):
        return {
            "generator": self._loader_identity(
                self._generator_data_loader
            ),
            "critic": self._loader_identity(self._critic_data_loader),
            "pd": self._loader_identity(self._pd_data_loader),
        }

    def _reset_data_iterators(self):
        if self._critic_data_loader is None:
            shared_batches_consumed = (
                self.generator_batches_consumed
                + self.critic_batches_consumed
            )
            self.dataloader = cycle(
                self._generator_data_loader,
                start_batch=shared_batches_consumed,
            )
            self.critic_dataloader = self.dataloader
        else:
            self.dataloader = cycle(
                self._generator_data_loader,
                start_batch=self.generator_batches_consumed,
            )
            self.critic_dataloader = cycle(
                self._critic_data_loader,
                start_batch=self.critic_batches_consumed,
            )
        if self._pd_data_loader is not None:
            self.pd_dataloader = cycle(
                self._pd_data_loader,
                start_batch=self.pd_batches_consumed,
            )

    def _raise_collective_errors(self, local_error, label):
        """Raise one identical error after every rank reports local status."""
        local_bad = torch.tensor(
            int(local_error is not None),
            device=self.device,
            dtype=torch.int32,
        )
        dist.all_reduce(local_bad, op=dist.ReduceOp.SUM)
        if local_bad.item() == 0:
            return
        errors = [None] * self.world_size
        dist.all_gather_object(errors, local_error)
        errors = [error for error in errors if error is not None]
        if errors:
            raise RuntimeError(f"{label}: " + "; ".join(errors))

    def _bidirectional_cd_structure_error(self):
        student = self._named_parameter_map(self.model.generator)
        target = self._named_parameter_map(self.model.bidirectional_cd_ema)
        if student.keys() != target.keys():
            missing = sorted(student.keys() - target.keys())[:5]
            extra = sorted(target.keys() - student.keys())[:5]
            return f"parameter names differ: missing={missing}, extra={extra}"
        for name, source in student.items():
            if source.shape != target[name].shape:
                return (
                    f"EMA shard shape mismatch for {name}: "
                    f"{source.shape} != {target[name].shape}"
                )
        student_buffers = {
            self._clean_fsdp_name(name): value
            for name, value in self.model.generator.named_buffers()
        }
        target_buffers = {
            self._clean_fsdp_name(name): value
            for name, value in self.model.bidirectional_cd_ema.named_buffers()
        }
        if student_buffers.keys() != target_buffers.keys():
            return "buffer names differ"
        for name, source in student_buffers.items():
            if source.shape != target_buffers[name].shape:
                return f"EMA buffer shape mismatch for {name}"
        return None

    @torch.no_grad()
    def _assert_bidirectional_cd_target_matches_student(self):
        if not self.bidirectional_cd_enabled:
            return
        local_error = self._bidirectional_cd_structure_error()
        if local_error is not None:
            local_error = f"rank {dist.get_rank()}: {local_error}"
        self._raise_collective_errors(
            local_error, "Bi-CD student/EMA structure validation failed"
        )
        student = self._named_parameter_map(self.model.generator)
        target = self._named_parameter_map(self.model.bidirectional_cd_ema)
        local_max = torch.zeros((), device=self.device, dtype=torch.float32)
        for name, source in student.items():
            destination = target[name]
            if source.numel():
                local_max = torch.maximum(
                    local_max,
                    (source.detach().float() - destination.detach().float())
                    .abs()
                    .max(),
                )
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        if local_max.item() != 0.0:
            raise RuntimeError(
                "Bi-CD student and EMA target were not initialized from the "
                f"same checkpoint (max diff {local_max.item()})"
            )

    @torch.no_grad()
    def _update_bidirectional_cd_ema(self):
        if not self.bidirectional_cd_enabled:
            return
        decay = float(getattr(self.config, "bidirectional_cd_ema_decay", 0.99))
        if not 0.0 <= decay < 1.0:
            raise ValueError("bidirectional_cd_ema_decay must be in [0, 1)")
        local_error = self._bidirectional_cd_structure_error()
        if local_error is not None:
            local_error = f"rank {dist.get_rank()}: {local_error}"
        self._raise_collective_errors(
            local_error, "Bi-CD EMA update structure validation failed"
        )
        student = self._named_parameter_map(self.model.generator)
        target = self._named_parameter_map(self.model.bidirectional_cd_ema)
        for name, source in student.items():
            destination = target[name]
            destination.mul_(decay).add_(
                source.detach().to(dtype=destination.dtype),
                alpha=1.0 - decay,
            )
        student_buffers = {
            self._clean_fsdp_name(name): value
            for name, value in self.model.generator.named_buffers()
        }
        target_buffers = {
            self._clean_fsdp_name(name): value
            for name, value in self.model.bidirectional_cd_ema.named_buffers()
        }
        for name, source in student_buffers.items():
            destination = target_buffers[name]
            destination.copy_(source)

    def _build_dataset_collectively(self, factory, label):
        """Build a rank-local dataset, then fail identically on every rank."""
        dataset = None
        local_error = None
        try:
            dataset = factory()
        except Exception as exc:
            local_error = (
                f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
            )
        self._raise_collective_errors(
            local_error, f"Failed to initialize {label}"
        )

        identity = (
            len(dataset),
            getattr(dataset, "records_digest", None),
        )
        identities = [None] * self.world_size
        dist.all_gather_object(identities, identity)
        if len(set(identities)) != 1:
            raise RuntimeError(
                f"{label} differs across ranks: identities={identities}"
            )
        return dataset

    def _next_batch_collectively(self, iterator, label, counter_name):
        """Fetch a batch on every rank and turn worker failures into one exit."""
        batch = None
        local_error = None
        try:
            batch = next(iterator)
        except Exception as exc:
            local_error = (
                f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
            )
        self._raise_collective_errors(local_error, f"Failed to fetch {label}")
        setattr(self, counter_name, int(getattr(self, counter_name)) + 1)
        return batch

    @staticmethod
    def _is_fatal_cuda_error(exc):
        return isinstance(exc, torch.cuda.OutOfMemoryError) or (
            isinstance(exc, RuntimeError)
            and any(
                marker in str(exc).lower()
                for marker in (
                    "cuda error",
                    "device-side assert",
                    "illegal memory access",
                    "nccl error",
                )
            )
        )

    def _encode_real_video_collectively(self, batch):
        """Encode real RGB clips and validate the VAE result on all ranks."""
        clean_latent = None
        local_error = None
        pixels = None
        try:
            if "video" not in batch:
                raise KeyError("Bi-CD training batch is missing real video frames")
            # Remove the pinned CPU tensor from the batch as soon as the async
            # copy is issued; no later part of the step consumes pixels.
            video = batch.pop("video")
            pixels = video.to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            del video
            pixels = pixels.div_(127.5).sub_(1.0)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(pixels).to(
                    dtype=self.dtype
                )
            del pixels
            pixels = None
            expected_shape = tuple(self.config.image_or_video_shape[1:])
            if tuple(clean_latent.shape[1:]) != expected_shape:
                raise RuntimeError(
                    "Real-video VAE latent shape mismatch: "
                    f"{tuple(clean_latent.shape[1:])} != {expected_shape}"
                )
            if not bool(torch.isfinite(clean_latent).all().item()):
                raise FloatingPointError(
                    "Real-video VAE encoding contains NaN or Inf"
                )
            clean_latent = clean_latent.detach()
        except Exception as exc:
            if self._is_fatal_cuda_error(exc):
                # A fatal CUDA failure can poison this rank's context.  Do not
                # enter another NCCL collective: let torchrun terminate peers
                # through async error handling instead of hanging all ranks.
                pixels = None
                clean_latent = None
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"rank {dist.get_rank()} fatal CUDA failure during VAE encode: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            local_error = (
                f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
            )
            clean_latent = None
            pixels = None
        self._raise_collective_errors(
            local_error, "Failed to encode the real-video Bi-CD batch"
        )
        return clean_latent

    def _assert_finite_collective(self, value, label):
        """Abort an optimizer update on every rank if any rank is non-finite."""
        tensor = torch.as_tensor(value, device=self.device).detach()
        local_bad = torch.tensor(
            int(not bool(torch.isfinite(tensor).all().item())),
            device=self.device,
            dtype=torch.int32,
        )
        dist.all_reduce(local_bad, op=dist.ReduceOp.SUM)
        if local_bad.item():
            raise FloatingPointError(
                f"{label} is non-finite on {local_bad.item()} rank(s); "
                "optimizer step was not applied"
            )

    @contextmanager
    def _preserve_training_rng(self):
        """Keep evaluation randomness out of the training RNG stream."""
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.device),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
        try:
            yield
        finally:
            torch.set_rng_state(rng_state["torch"])
            torch.cuda.set_rng_state(rng_state["cuda"], device=self.device)
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])

    def _prune_old_checkpoints(self, current_dir):
        """Keep only the newest successful checkpoints (0 disables pruning)."""
        keep = self.max_checkpoints_to_keep
        if keep == 0:
            return
        root = Path(self.checkpoint_path)
        checkpoints = sorted(
            path
            for path in root.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]")
            if path.is_dir() and (path / "_SUCCESS").is_file()
        )
        for checkpoint in checkpoints[:-keep]:
            if os.path.abspath(checkpoint) == os.path.abspath(current_dir):
                continue
            shutil.rmtree(checkpoint)
            print(f"Pruned old checkpoint: {checkpoint}")

    def _resolve_resume_path(self, requested_path):
        """Resolve resume paths on rank 0 and broadcast one shared decision."""
        resolved = None
        if self.is_main_process:
            if requested_path == "auto":
                latest_link = os.path.join(self.checkpoint_path, "latest")
                if os.path.exists(latest_link):
                    resolved = os.path.realpath(latest_link)
                    print(f"Auto-detected latest checkpoint: {resolved}")
                else:
                    print(
                        f"No latest checkpoint found at {latest_link}; "
                        "falling back to generator_ckpt"
                    )
            elif requested_path:
                resolved = os.path.abspath(os.path.expanduser(str(requested_path)))

        payload = [resolved]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _load_sharded_checkpoint(self, checkpoint_dir: str):
        checkpoint_dir = os.path.realpath(checkpoint_dir)
        metadata = None
        validation_error = None
        if self.is_main_process:
            try:
                success_path = os.path.join(checkpoint_dir, "_SUCCESS")
                metadata_path = os.path.join(checkpoint_dir, "metadata.json")
                if not os.path.isfile(success_path):
                    raise RuntimeError(
                        f"Incomplete checkpoint (missing _SUCCESS): {checkpoint_dir}"
                    )
                if not os.path.isfile(metadata_path):
                    raise RuntimeError(
                        f"Incomplete checkpoint (missing metadata.json): {checkpoint_dir}"
                    )
                with open(metadata_path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
                if metadata.get("schema_version") != 2:
                    raise RuntimeError(
                        f"Unsupported checkpoint schema: "
                        f"{metadata.get('schema_version')}"
                    )
                if (
                    self.bidirectional_cd_enabled
                    and not metadata.get("has_bidirectional_cd_ema", False)
                ):
                    raise RuntimeError(
                        "Checkpoint predates bidirectional CD and lacks its "
                        "EMA target; resume would change the CD objective"
                    )
                if self.bidirectional_cd_enabled:
                    saved_cd_config = metadata.get("bidirectional_cd_config")
                    current_cd_config = (
                        self._bidirectional_cd_checkpoint_config()
                    )
                    if saved_cd_config != current_cd_config:
                        raise RuntimeError(
                            "Bi-CD configuration differs from the checkpoint: "
                            f"saved={saved_cd_config}, current={current_cd_config}"
                        )
                saved_fingerprint = metadata.get(
                    "training_semantics_fingerprint"
                )
                current_fingerprint = self._training_semantics_fingerprint()
                if saved_fingerprint != current_fingerprint:
                    saved_config = metadata.get(
                        "training_semantics_config", {}
                    )
                    current_config = self._training_semantics_config()
                    changed_keys = sorted(
                        key
                        for key in set(saved_config) | set(current_config)
                        if saved_config.get(key) != current_config.get(key)
                    )
                    raise RuntimeError(
                        "Training semantics differ from the checkpoint; "
                        f"changed keys={changed_keys[:20]}, "
                        f"saved fingerprint={saved_fingerprint}, "
                        f"current fingerprint={current_fingerprint}"
                    )
                saved_world_size = int(
                    metadata.get("world_size", self.world_size)
                )
                if (
                    saved_world_size != self.world_size
                    and not bool(
                        getattr(
                            self.config,
                            "allow_data_restart_on_reshard",
                            False,
                        )
                    )
                ):
                    raise RuntimeError(
                        "Checkpoint world size changed from "
                        f"{saved_world_size} to {self.world_size}; exact "
                        "DistributedSampler continuation is impossible. Set "
                        "allow_data_restart_on_reshard=true to explicitly "
                        "restart the data streams while resharding model state."
                    )
                saved_data_identity = metadata.get("data_stream_identity")
                current_data_identity = self._data_stream_identity()
                if saved_world_size == self.world_size:
                    data_identity_matches = (
                        saved_data_identity == current_data_identity
                    )
                else:
                    def without_batches(identity):
                        if not isinstance(identity, dict):
                            return identity
                        return {
                            name: (
                                None
                                if value is None
                                else {
                                    key: item
                                    for key, item in value.items()
                                    if key != "batches_per_epoch"
                                }
                            )
                            for name, value in identity.items()
                        }

                    data_identity_matches = (
                        without_batches(saved_data_identity)
                        == without_batches(current_data_identity)
                    )
                if not data_identity_matches:
                    raise RuntimeError(
                        "Training data streams differ from the checkpoint: "
                        f"saved={saved_data_identity}, "
                        f"current={current_data_identity}"
                    )
                for state_dir in ("train_state",):
                    if not os.path.isdir(os.path.join(checkpoint_dir, state_dir)):
                        raise RuntimeError(
                            f"Incomplete checkpoint (missing {state_dir}): "
                            f"{checkpoint_dir}"
                        )
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
        payload = [metadata, validation_error]
        dist.broadcast_object_list(payload, src=0)
        metadata, validation_error = payload
        if validation_error is not None:
            raise RuntimeError(validation_error)

        visibility_error = None
        visible_entries = None
        visible_metadata_digest = None
        try:
            required_paths = [
                os.path.join(checkpoint_dir, "_SUCCESS"),
                os.path.join(checkpoint_dir, "metadata.json"),
                os.path.join(checkpoint_dir, "train_state"),
            ]
            if metadata.get("has_ema", False):
                required_paths.append(
                    os.path.join(checkpoint_dir, "ema_state")
                )
            missing_paths = [
                path for path in required_paths if not os.path.exists(path)
            ]
            if missing_paths:
                raise FileNotFoundError(
                    f"checkpoint paths are not visible: {missing_paths}"
                )
            # Force one metadata directory read from every node before any rank
            # enters the collective DCP load.
            visible_entries = sorted(
                os.listdir(os.path.join(checkpoint_dir, "train_state"))
            )
            if not visible_entries:
                raise RuntimeError("train_state directory is empty")
            with open(
                os.path.join(checkpoint_dir, "metadata.json"), "rb"
            ) as handle:
                visible_metadata_digest = hashlib.sha256(
                    handle.read()
                ).hexdigest()
        except Exception as exc:
            visibility_error = (
                f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
            )
        self._raise_collective_errors(
            visibility_error, "Checkpoint visibility validation failed"
        )
        visibility_views = [None] * self.world_size
        dist.all_gather_object(
            visibility_views,
            (visible_entries, visible_metadata_digest),
        )
        if len({repr(view) for view in visibility_views}) != 1:
            raise RuntimeError(
                "Checkpoint directory contents differ across ranks: "
                f"{visibility_views}"
            )

        train_state = FSDPTrainState(
            self.model.generator,
            self.generator_optimizer,
            self.model.fake_score,
            self.critic_optimizer,
            (
                self.model.bidirectional_cd_ema
                if self.bidirectional_cd_enabled
                else None
            ),
        )
        dcp.load(
            {"train_state": train_state},
            checkpoint_id=os.path.join(checkpoint_dir, "train_state"),
        )
        self.step = int(metadata["step"])

        has_ema = bool(metadata.get("has_ema", False))
        if self.step >= self.config.ema_start_step and self.config.ema_weight > 0:
            if not has_ema:
                raise RuntimeError(
                    f"Checkpoint step {self.step} must contain EMA state but does not"
                )
            if self.generator_ema is None:
                self.generator_ema = EMA_FSDP(
                    self.model.generator,
                    decay=self.config.ema_weight,
                    device=getattr(self.config, "ema_device", "cpu"),
                )
            with self.generator_ema.preserve_model_parameters(self.model.generator):
                dcp.load(
                    {"ema": FSDPModelState(self.model.generator)},
                    checkpoint_id=os.path.join(checkpoint_dir, "ema_state"),
                )
                self.generator_ema.capture(self.model.generator)

        rng_path = os.path.join(
            checkpoint_dir, "rng", f"rank_{dist.get_rank():05d}.pt"
        )
        saved_world_size = int(metadata.get("world_size", self.world_size))
        self.generator_update_count = int(
            metadata.get("generator_update_count", 0)
        )
        self.bidirectional_cd_ema_update_count = int(
            metadata.get("bidirectional_cd_ema_update_count", 0)
        )
        if saved_world_size == self.world_size:
            data_cursors = metadata.get("data_cursors")
            if not isinstance(data_cursors, dict):
                raise RuntimeError(
                    "Checkpoint lacks exact data-stream cursors"
                )
            self.generator_batches_consumed = int(
                data_cursors.get("generator_batches", 0)
            )
            self.critic_batches_consumed = int(
                data_cursors.get("critic_batches", 0)
            )
            self.pd_batches_consumed = int(
                data_cursors.get("pd_batches", 0)
            )
        else:
            # DistributedSampler ownership changes under resharding, so a
            # per-rank cursor cannot represent the same global sample stream.
            self.generator_batches_consumed = 0
            self.critic_batches_consumed = 0
            self.pd_batches_consumed = 0
        self._reset_data_iterators()
        if saved_world_size == self.world_size:
            rng_error = None
            try:
                if not os.path.isfile(rng_path):
                    raise RuntimeError(
                        "Checkpoint is missing RNG state for rank "
                        f"{dist.get_rank()}: {rng_path}"
                    )
                rng_state = torch.load(
                    rng_path, map_location="cpu", weights_only=False
                )
                required_rng_keys = {"torch", "cuda", "python", "numpy"}
                missing_rng_keys = required_rng_keys.difference(rng_state)
                if missing_rng_keys:
                    raise RuntimeError(
                        f"RNG state is missing keys: {sorted(missing_rng_keys)}"
                    )
                torch.set_rng_state(rng_state["torch"])
                torch.cuda.set_rng_state(rng_state["cuda"], device=self.device)
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
            except Exception as exc:
                rng_error = (
                    f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
                )
            rng_errors = [None] * self.world_size
            dist.all_gather_object(rng_errors, rng_error)
            rng_errors = [error for error in rng_errors if error is not None]
            if rng_errors:
                raise RuntimeError(
                    "Failed to restore RNG state: " + "; ".join(rng_errors)
                )
        elif self.is_main_process:
            print(
                "Checkpoint was resharded from "
                f"world_size={saved_world_size} to {self.world_size}; retaining "
                "the deterministic per-rank RNG state initialized for this launch."
            )

        if self.is_main_process:
            print(f"Successfully resumed sharded checkpoint at step {self.step}")

    def _load_legacy_checkpoint(self, checkpoint_path: str):
        """Strict migration path for older single-file checkpoints."""
        if self.bidirectional_cd_enabled:
            raise RuntimeError(
                "Legacy checkpoints do not contain the sharded Bi-CD EMA "
                "target. Start Bi-CD from generator_ckpt or a new DCP checkpoint."
            )
        checkpoint = None
        validation_error = None
        metadata = None
        if self.is_main_process:
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                required = {
                    "generator",
                    "critic",
                    "step",
                    "generator_optimizer",
                    "critic_optimizer",
                }
                missing = sorted(required.difference(checkpoint))
                if missing:
                    raise RuntimeError(
                        "Legacy resume checkpoint is incomplete; "
                        f"missing keys: {missing}"
                    )
                metadata = {
                    "step": int(checkpoint["step"]),
                    "has_ema": "generator_ema" in checkpoint,
                }
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"

        payload = [metadata, validation_error]
        dist.broadcast_object_list(payload, src=0)
        metadata, validation_error = payload
        if validation_error is not None:
            raise RuntimeError(validation_error)

        full_options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            strict=True,
            broadcast_from_rank0=True,
        )
        set_state_dict(
            self.model.generator,
            self.generator_optimizer,
            model_state_dict=(checkpoint["generator"] if self.is_main_process else {}),
            optim_state_dict=(
                checkpoint["generator_optimizer"] if self.is_main_process else {}
            ),
            options=full_options,
        )
        set_state_dict(
            self.model.fake_score,
            self.critic_optimizer,
            model_state_dict=(checkpoint["critic"] if self.is_main_process else {}),
            optim_state_dict=(
                checkpoint["critic_optimizer"] if self.is_main_process else {}
            ),
            options=full_options,
        )
        self.step = int(metadata["step"])

        if self.step >= self.config.ema_start_step and self.config.ema_weight > 0:
            if not metadata["has_ema"]:
                raise RuntimeError(
                    f"Legacy checkpoint step {self.step} is missing generator_ema"
                )
            if self.generator_ema is None:
                self.generator_ema = EMA_FSDP(
                    self.model.generator,
                    decay=self.config.ema_weight,
                    device=getattr(self.config, "ema_device", "cpu"),
                )
            with self.generator_ema.preserve_model_parameters(self.model.generator):
                set_model_state_dict(
                    self.model.generator,
                    checkpoint["generator_ema"] if self.is_main_process else {},
                    options=full_options,
                )
                self.generator_ema.capture(self.model.generator)

        del checkpoint
        if self.is_main_process:
            print(f"Successfully migrated legacy checkpoint at step {self.step}")

    def save(self):
        """Atomically write a reshardable DCP checkpoint on every rank."""
        final_dir = os.path.join(
            self.checkpoint_path, f"step_{self.step:08d}"
        )
        temp_dir = None
        setup_error = None
        if self.is_main_process:
            try:
                os.makedirs(self.checkpoint_path, exist_ok=True)
                if os.path.lexists(final_dir):
                    raise FileExistsError(
                        f"Refusing to overwrite existing checkpoint: {final_dir}"
                    )
                temp_dir = os.path.join(
                    self.checkpoint_path,
                    f".tmp_step_{self.step:08d}_{uuid.uuid4().hex}",
                )
                os.makedirs(os.path.join(temp_dir, "train_state"))
                os.makedirs(os.path.join(temp_dir, "rng"))
                if self.generator_ema is not None:
                    os.makedirs(os.path.join(temp_dir, "ema_state"))
            except Exception as exc:
                setup_error = f"{type(exc).__name__}: {exc}"
                if temp_dir and os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir)

        payload = [temp_dir, setup_error]
        dist.broadcast_object_list(payload, src=0)
        temp_dir, setup_error = payload
        if setup_error is not None:
            raise RuntimeError(setup_error)

        # DCP writes one shard per rank and therefore requires a genuinely
        # shared checkpoint directory. Probe visibility and write access before
        # entering any DCP collective, so a node-local path fails consistently.
        probe_error = None
        probe_path = os.path.join(
            temp_dir, f".shared_probe_rank_{dist.get_rank():05d}"
        )
        try:
            if not os.path.isdir(temp_dir):
                raise FileNotFoundError(
                    f"checkpoint temp directory is not visible: {temp_dir}"
                )
            Path(probe_path).write_text(
                str(dist.get_rank()), encoding="utf-8"
            )
        except Exception as exc:
            probe_error = (
                f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
            )
        probe_errors = [None] * self.world_size
        dist.all_gather_object(probe_errors, probe_error)
        probe_errors = [error for error in probe_errors if error is not None]

        visibility_error = None
        if self.is_main_process and not probe_errors:
            missing_probes = [
                rank
                for rank in range(self.world_size)
                if not os.path.isfile(
                    os.path.join(temp_dir, f".shared_probe_rank_{rank:05d}")
                )
            ]
            if missing_probes:
                visibility_error = (
                    "checkpoint_dir is not shared across ranks; missing probes "
                    f"from ranks {missing_probes}"
                )
        payload = [visibility_error]
        dist.broadcast_object_list(payload, src=0)
        visibility_error = payload[0]
        if probe_errors or visibility_error is not None:
            if self.is_main_process and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
            details = "; ".join(probe_errors)
            if visibility_error is not None:
                details = "; ".join(filter(None, (details, visibility_error)))
            raise RuntimeError(
                "Sharded checkpoint shared-filesystem probe failed: " + details
            )
        dist.barrier()
        cleanup_error = None
        if self.is_main_process:
            try:
                for rank in range(self.world_size):
                    os.unlink(
                        os.path.join(
                            temp_dir, f".shared_probe_rank_{rank:05d}"
                        )
                    )
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
        payload = [cleanup_error]
        dist.broadcast_object_list(payload, src=0)
        if payload[0] is not None:
            raise RuntimeError(
                "Failed to clean shared-filesystem probes: " + payload[0]
            )

        if self.is_main_process:
            print(
                f"Saving sharded checkpoint for step {self.step} to {final_dir}"
            )

        try:
            train_state = FSDPTrainState(
                self.model.generator,
                self.generator_optimizer,
                self.model.fake_score,
                self.critic_optimizer,
                (
                    self.model.bidirectional_cd_ema
                    if self.bidirectional_cd_enabled
                    else None
                ),
            )
            dcp.save(
                {"train_state": train_state},
                checkpoint_id=os.path.join(temp_dir, "train_state"),
            )

            if self.generator_ema is not None:
                with self.generator_ema.apply_to(self.model.generator):
                    dcp.save(
                        {"ema": FSDPModelState(self.model.generator)},
                        checkpoint_id=os.path.join(temp_dir, "ema_state"),
                    )

            rng_state = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            }
            rng_error = None
            try:
                torch.save(
                    rng_state,
                    os.path.join(
                        temp_dir, "rng", f"rank_{dist.get_rank():05d}.pt"
                    ),
                )
            except Exception as exc:
                rng_error = (
                    f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
                )
            rng_errors = [None] * self.world_size
            dist.all_gather_object(rng_errors, rng_error)
            rng_errors = [error for error in rng_errors if error is not None]
            if rng_errors:
                raise RuntimeError(
                    "Failed to save RNG state: " + "; ".join(rng_errors)
                )
            dist.barrier()

            finalize_error = None
            if self.is_main_process:
                try:
                    metadata = {
                        "schema_version": 2,
                        "step": int(self.step),
                        "world_size": int(self.world_size),
                        "has_ema": self.generator_ema is not None,
                        "has_bidirectional_cd_ema": (
                            self.bidirectional_cd_enabled
                        ),
                        "bidirectional_cd_config": (
                            self._bidirectional_cd_checkpoint_config()
                        ),
                        "training_semantics_fingerprint": (
                            self._training_semantics_fingerprint()
                        ),
                        "training_semantics_config": (
                            self._training_semantics_config()
                        ),
                        "data_cursors": {
                            "generator_batches": int(
                                self.generator_batches_consumed
                            ),
                            "critic_batches": int(
                                self.critic_batches_consumed
                            ),
                            "pd_batches": int(self.pd_batches_consumed),
                        },
                        "data_stream_identity": (
                            self._data_stream_identity()
                        ),
                        "generator_update_count": int(
                            self.generator_update_count
                        ),
                        "bidirectional_cd_ema_update_count": int(
                            self.bidirectional_cd_ema_update_count
                        ),
                        "format": "torch.distributed.checkpoint",
                        "created_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    }
                    with open(
                        os.path.join(temp_dir, "metadata.json"),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(metadata, handle, indent=2)
                        handle.flush()
                        os.fsync(handle.fileno())
                    Path(os.path.join(temp_dir, "_SUCCESS")).touch()
                    os.replace(temp_dir, final_dir)

                    latest_link = os.path.join(self.checkpoint_path, "latest")
                    temporary_link = os.path.join(
                        self.checkpoint_path,
                        f".tmp_latest_{uuid.uuid4().hex}",
                    )
                    os.symlink(os.path.basename(final_dir), temporary_link)
                    os.replace(temporary_link, latest_link)
                    self._prune_old_checkpoints(final_dir)
                except Exception as exc:
                    finalize_error = f"{type(exc).__name__}: {exc}"

            payload = [finalize_error]
            dist.broadcast_object_list(payload, src=0)
            finalize_error = payload[0]
            if finalize_error is not None:
                raise RuntimeError(finalize_error)
            dist.barrier()
        except Exception:
            # Only remove the unique, unpublished temporary directory. A final
            # checkpoint carrying _SUCCESS is never deleted here.
            if self.is_main_process and temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
            raise

        if self.is_main_process:
            print(f"Sharded checkpoint committed: {final_dir}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            cached_unconditional = getattr(self, "unconditional_dict", None)
            if (
                cached_unconditional is None
                or next(iter(cached_unconditional.values())).shape[0] != batch_size
            ):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            # Progressive Distillation loss (if enabled)
            pd_loss = torch.tensor(0.0, device=self.device)
            pd_log_dict = {}
            if self.pd_enabled and self.pd_dataloader is not None and hasattr(self.model, 'progressive_distillation_loss'):
                pd_batch = self._next_batch_collectively(
                    self.pd_dataloader,
                    "progressive-distillation batch",
                    "pd_batches_consumed",
                )
                pd_prompts = pd_batch["prompts"]
                # Teacher ODE chords are extrapolated to x0. Keep source points
                # in fp32; bf16 quantization is strongly amplified by 1/delta_sigma.
                pd_ode_latent = pd_batch["ode_latent"].to(
                    device=self.device, dtype=torch.float32
                )
                
                # Encode PD prompts text embeddings
                with torch.no_grad():
                    pd_conditional_dict = self.model.text_encoder(
                        text_prompts=pd_prompts)
                
                pd_loss, pd_log_dict = self.model.progressive_distillation_loss(
                    ode_latent=pd_ode_latent,
                    conditional_dict=pd_conditional_dict,
                )
                pd_loss = pd_loss * getattr(self.config, 'pd_loss_weight', 1.0)

            bidirectional_cd_loss = torch.tensor(
                0.0, device=self.device
            )
            bidirectional_cd_log_dict = {}
            if self.bidirectional_cd_enabled:
                # Backpropagate DMD first so its long-rollout graph is released
                # before the teacher/EMA/student Bi-CD forwards. Gradients still
                # add into the same generator optimizer step.
                self._assert_finite_collective(
                    generator_loss, "DMD generator loss"
                )
                (
                    generator_loss / self.gradient_accumulation_steps
                ).backward()

                clean_latent = self._encode_real_video_collectively(batch)
                (
                    bidirectional_cd_loss,
                    bidirectional_cd_log_dict,
                ) = self.model.bidirectional_consistency_loss(
                    clean_latent=clean_latent,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                )
                bidirectional_cd_loss = (
                    bidirectional_cd_loss
                    * float(
                        getattr(
                            self.config,
                            "bidirectional_cd_loss_weight",
                            1.0,
                        )
                    )
                )
                self._assert_finite_collective(
                    bidirectional_cd_loss, "bidirectional CD loss"
                )
                (
                    bidirectional_cd_loss
                    / self.gradient_accumulation_steps
                ).backward()

            # DMD remains the rollout/deployment objective. Bi-CD replaces
            # forward-KL as the real-data bidirectional regularizer.
            total_loss = (
                generator_loss + pd_loss + bidirectional_cd_loss
            )
            self._assert_finite_collective(total_loss, "generator loss")

            if not self.bidirectional_cd_enabled:
                # Scale loss for gradient accumulation. The Bi-CD path above
                # uses two equivalent backwards to cap peak activation memory.
                scaled_loss = total_loss / self.gradient_accumulation_steps
                scaled_loss.backward()

            # Log the unscaled loss for monitoring
            generator_log_dict.update({"generator_loss": generator_loss})
            if self.pd_enabled:
                generator_log_dict.update({
                    "pd_loss": pd_loss.detach(),
                    "pd_loss_weighted": pd_loss.detach(),
                    "total_generator_loss": total_loss.detach(),
                })
                generator_log_dict.update(pd_log_dict)
            if self.bidirectional_cd_enabled:
                generator_log_dict.update({
                    "bidirectional_cd_loss": (
                        bidirectional_cd_loss.detach()
                    ),
                    "total_generator_loss": total_loss.detach(),
                })
                generator_log_dict.update(
                    bidirectional_cd_log_dict
                )

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )
        self._assert_finite_collective(critic_loss, "critic loss")

        # Scale loss for gradient accumulation
        scaled_loss = critic_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        # Log the unscaled loss for monitoring
        critic_log_dict.update({"critic_loss": critic_loss})

        return critic_log_dict

    def generate_video(self, prompts, image=None, num_output_frames=None):
        """Generate evaluation videos and always restore shared model state."""
        batch_size = len(prompts)
        num_training_frames = int(
            getattr(self.config, "num_training_frames", 21)
        )
        if num_output_frames is None:
            configured_frames = int(
                getattr(self.config, "eval_num_frames", num_training_frames)
            )
            num_output_frames = configured_frames or num_training_frames

        num_frame_per_block = int(
            getattr(self.config, "num_frame_per_block", 3)
        )
        independent_first_frame = bool(
            getattr(self.config, "independent_first_frame", False)
        )
        divisible_frames = (
            num_output_frames - 1 if independent_first_frame else num_output_frames
        )
        remainder = divisible_frames % num_frame_per_block
        if remainder:
            num_output_frames += num_frame_per_block - remainder

        latent_shape = list(self.config.image_or_video_shape)
        latent_channels, latent_height, latent_width = latent_shape[2:]
        pipeline = None
        block_attrs = []
        was_training = self.model.generator.training
        self.model.generator.eval()

        try:
            pipeline = CausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae,
            )
            generator_wrapper = getattr(
                pipeline.generator, "module", pipeline.generator
            )
            backbone = generator_wrapper.model
            patch_size = tuple(backbone.patch_size)
            frame_seq_length = (
                latent_height
                * latent_width
                // (patch_size[1] * patch_size[2])
            )
            local_attn_window = num_training_frames
            for block in backbone.blocks:
                block_attrs.append((
                    block.self_attn,
                    block.self_attn.local_attn_size,
                    block.self_attn.max_attention_size,
                ))
                block.self_attn.local_attn_size = local_attn_window
                block.self_attn.max_attention_size = (
                    local_attn_window * frame_seq_length
                )

            pipeline.local_attn_size = local_attn_window
            pipeline.num_frame_per_block = num_frame_per_block
            pipeline.use_ode_trajectory = False
            pipeline.kv_cache1 = None

            if image is not None:
                image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(
                    device=self.device, dtype=self.dtype
                )
                initial_latent = pipeline.vae.encode_to_latent(image).to(
                    device=self.device, dtype=self.dtype
                )
                initial_latent = initial_latent.repeat(
                    batch_size, 1, 1, 1, 1
                )
                noise_frames = num_output_frames - 1
            else:
                initial_latent = None
                noise_frames = num_output_frames

            sampled_noise = torch.randn(
                [
                    batch_size,
                    noise_frames,
                    latent_channels,
                    latent_height,
                    latent_width,
                ],
                device=self.device,
                dtype=self.dtype,
            )
            if self.config.distribution_loss == "timestep_forcing_dmd":
                video, _ = pipeline.inference_hybrid(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    initial_latent=initial_latent,
                    frame_first_steps=0,
                )
            else:
                video, _ = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    initial_latent=initial_latent,
                )

            return (
                rearrange(video, "b t c h w -> b t h w c")
                .cpu()
                .numpy()
                * 255.0
            )
        finally:
            if pipeline is not None:
                for self_attn, local_size, max_size in block_attrs:
                    self_attn.local_attn_size = local_size
                    self_attn.max_attention_size = max_size
                try:
                    pipeline.vae.model.clear_cache()
                except Exception:
                    pass
                pipeline.kv_cache1 = None
                if hasattr(pipeline, "crossattn_cache"):
                    pipeline.crossattn_cache = None
                del pipeline
            if was_training:
                self.model.generator.train()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self, prompts):
        """Run distributed FSDP evaluation and write artifacts on rank 0."""
        num_prompts = max(1, int(getattr(self.config, "eval_num_prompts", 1)))
        prompts = prompts[:num_prompts]
        setup_error = None
        if self.is_main_process:
            try:
                os.makedirs(self.eval_path, exist_ok=True)
                print(
                    f"Running evaluation at step {self.step} with "
                    f"{len(prompts)} prompt(s)"
                )
            except Exception as exc:
                setup_error = f"{type(exc).__name__}: {exc}"
        payload = [setup_error]
        dist.broadcast_object_list(payload, src=0)
        if payload[0] is not None:
            raise RuntimeError(f"Evaluation setup failed: {payload[0]}")

        # A failure inside an FSDP collective invalidates the process group.
        # Propagate it immediately so torchrun tears down the job; attempting a
        # follow-up error collective can otherwise wait for the PG timeout.
        if self.generator_ema is None:
            sample_video = self.generate_video(prompts=prompts)
        else:
            with self.generator_ema.apply_to(self.model.generator):
                sample_video = self.generate_video(prompts=prompts)

        write_error = None
        if self.is_main_process:
            try:
                for idx in range(sample_video.shape[0]):
                    video_path = os.path.join(
                        self.eval_path,
                        f"sample_step_{self.step:08d}_p{idx}.mp4",
                    )
                    _write_video_mp4(
                        video_path,
                        torch.from_numpy(sample_video[idx]).to(torch.uint8),
                        fps=int(getattr(self.config, "eval_fps", 16)),
                    )
                    if self.tb_writer is not None:
                        first_frame = (
                            torch.from_numpy(sample_video[idx, 0])
                            .permute(2, 0, 1)
                            / 255.0
                        )
                        self.tb_writer.add_image(
                            f"evaluation/prompt_{idx}_first_frame",
                            first_frame,
                            self.step,
                        )
                print(
                    f"Saved {sample_video.shape[0]} evaluation video(s) "
                    f"to {self.eval_path}"
                )
            except Exception as exc:
                write_error = f"{type(exc).__name__}: {exc}"
        payload = [write_error]
        dist.broadcast_object_list(payload, src=0)
        if payload[0] is not None:
            raise RuntimeError(f"Evaluation artifact write failed: {payload[0]}")

    def train(self):
        start_step = self.step
        train_start_time = time.time()
        
        if self.is_main_process:
            print(f"\nTraining started at step {start_step}")
        
        # Default sampling prompts
        default_sample_prompt_list = [
            "Two NBA legends, Michael Jordan and Kobe Bryant, competing intensely on a basketball court. Both players are mid-action, with Jordan leaping towards the hoop, showcasing his iconic hang time, and Kobe driving to the basket, displaying his agility and determination. They are both wearing their respective team uniforms from their prime years, Jordan in his Chicago Bulls jersey and Kobe in his Los Angeles Lakers attire. The court is vividly lit with spectators cheering loudly in the background. The focus is on the dynamic interaction between the two players, capturing the essence of competitive sportsmanship. Medium shot, emphasizing their physical engagement and the intensity of the moment.",
            "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it's tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.",
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
 
        ]

        max_steps = int(getattr(self.config, "max_steps", 0))
        while max_steps <= 0 or self.step < max_steps:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            # Train the generator (with gradient accumulation)
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for accum_step in range(self.gradient_accumulation_steps):
                    batch = self._next_batch_collectively(
                        self.dataloader,
                        "generator real-video batch",
                        "generator_batches_consumed",
                    )
                    # Use no_sync context manager to skip gradient sync during accumulation, sync only on last step
                    if (
                        accum_step < self.gradient_accumulation_steps - 1
                        and not bool(
                            getattr(
                                self.config,
                                "sync_each_microbatch",
                                False,
                            )
                        )
                    ):
                        with self.model.generator.no_sync():
                            extra = self.fwdbwd_one_step(batch, True)
                    else:
                        extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)
                # Perform clip_grad_norm after gradient accumulation is complete
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator)
                self._assert_finite_collective(
                    generator_grad_norm, "generator gradient norm"
                )
                generator_log_dict = merge_dict_list(extras_list)
                generator_log_dict["generator_grad_norm"] = generator_grad_norm
                self.generator_optimizer.step()
                self.generator_update_count += 1
                self._update_bidirectional_cd_ema()
                if self.bidirectional_cd_enabled:
                    self.bidirectional_cd_ema_update_count += 1
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic (with gradient accumulation)
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            for accum_step in range(self.gradient_accumulation_steps):
                batch = self._next_batch_collectively(
                    self.critic_dataloader,
                    "critic prompt batch",
                    "critic_batches_consumed",
                )
                # Use no_sync context manager to skip gradient sync during accumulation, sync only on last step
                if (
                    accum_step < self.gradient_accumulation_steps - 1
                    and not bool(
                        getattr(
                            self.config,
                            "sync_each_microbatch",
                            False,
                        )
                    )
                ):
                    with self.model.fake_score.no_sync():
                        extra = self.fwdbwd_one_step(batch, False)
                else:
                    extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
            # Perform clip_grad_norm after gradient accumulation is complete
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            self._assert_finite_collective(
                critic_grad_norm, "critic gradient norm"
            )
            critic_log_dict = merge_dict_list(extras_list)
            critic_log_dict["critic_grad_norm"] = critic_grad_norm
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(
                    self.model.generator,
                    decay=self.config.ema_weight,
                    device=getattr(self.config, "ema_device", "cpu"),
                )

            should_checkpoint = (
                not self.config.no_save
                and self.checkpoint_interval > 0
                and self.step % self.checkpoint_interval == 0
            )
            if should_checkpoint:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            should_evaluate = (
                not self.config.no_visualize
                and self.eval_interval > 0
                and self.step % self.eval_interval == 0
            )
            if should_evaluate:
                with self._preserve_training_rng():
                    self.evaluate(default_sample_prompt_list)
                torch.cuda.empty_cache()

            # Logging
            should_log = (
                self.log_interval > 0
                and self.step % self.log_interval == 0
            )
            if self.is_main_process and should_log:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    generator_loss_mean = generator_log_dict["generator_loss"].mean().item()
                    generator_grad_norm_mean = generator_log_dict["generator_grad_norm"].mean().item()
                    dmdtrain_gradient_norm_mean = generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                    
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_loss_mean,
                            "generator_grad_norm": generator_grad_norm_mean,
                            "dmdtrain_gradient_norm": dmdtrain_gradient_norm_mean
                        }
                    )
                    
                    # TensorBoard: log generator metrics
                    self.tb_writer.add_scalar("loss/generator", generator_loss_mean, self.step)
                    self.tb_writer.add_scalar("grad_norm/generator", generator_grad_norm_mean, self.step)
                    self.tb_writer.add_scalar("grad_norm/dmdtrain_gradient", dmdtrain_gradient_norm_mean, self.step)
                    
                    # TensorBoard: log PD metrics (if enabled)
                    if self.pd_enabled and "pd_loss" in generator_log_dict:
                        pd_loss_mean = generator_log_dict["pd_loss"].mean().item()
                        self.tb_writer.add_scalar("loss/pd", pd_loss_mean, self.step)
                        wandb_loss_dict["pd_loss"] = pd_loss_mean
                        if "total_generator_loss" in generator_log_dict:
                            total_gen_loss_mean = generator_log_dict["total_generator_loss"].mean().item()
                            self.tb_writer.add_scalar("loss/total_generator", total_gen_loss_mean, self.step)
                            wandb_loss_dict["total_generator_loss"] = total_gen_loss_mean
                    
                    # TensorBoard: log per-chunk generator loss (if multiple accumulation steps)
                    if self.gradient_accumulation_steps > 1:
                        for i, loss_val in enumerate(generator_log_dict["generator_loss"].tolist()):
                            self.tb_writer.add_scalar(f"loss_per_chunk/generator_chunk_{i}", loss_val, self.step)
                    
                    # TensorBoard: log other useful metrics from generator_log_dict
                    for key, value in generator_log_dict.items():
                        if key not in ["generator_loss", "generator_grad_norm", "dmdtrain_gradient_norm"]:
                            if isinstance(value, torch.Tensor):
                                self.tb_writer.add_scalar(f"generator/{key}", value.mean().item(), self.step)
                            elif isinstance(value, (int, float)):
                                self.tb_writer.add_scalar(f"generator/{key}", value, self.step)

                critic_loss_mean = critic_log_dict["critic_loss"].mean().item()
                critic_grad_norm_mean = critic_log_dict["critic_grad_norm"].mean().item()
                
                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_loss_mean,
                        "critic_grad_norm": critic_grad_norm_mean
                    }
                )
                
                # TensorBoard: log critic metrics
                self.tb_writer.add_scalar("loss/critic", critic_loss_mean, self.step)
                self.tb_writer.add_scalar("grad_norm/critic", critic_grad_norm_mean, self.step)
                
                # TensorBoard: log per-chunk critic loss (if multiple accumulation steps)
                if self.gradient_accumulation_steps > 1:
                    for i, loss_val in enumerate(critic_log_dict["critic_loss"].tolist()):
                        self.tb_writer.add_scalar(f"loss_per_chunk/critic_chunk_{i}", loss_val, self.step)
                
                # TensorBoard: log other useful metrics from critic_log_dict
                for key, value in critic_log_dict.items():
                    if key not in ["critic_loss", "critic_grad_norm"]:
                        if isinstance(value, torch.Tensor):
                            self.tb_writer.add_scalar(f"critic/{key}", value.mean().item(), self.step)
                        elif isinstance(value, (int, float)):
                            self.tb_writer.add_scalar(f"critic/{key}", value, self.step)

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                
                # Print training progress at the independently configured cadence.
                current_time = time.time()
                elapsed = current_time - train_start_time
                steps_done = self.step - start_step
                avg_time = elapsed / steps_done if steps_done > 0 else 0
                
                log_parts = [f"Step {self.step}"]
                if TRAIN_GENERATOR:
                    log_parts.append(f"G_loss={generator_loss_mean:.4f}")
                    log_parts.append(f"G_grad={generator_grad_norm_mean:.2f}")
                    if "rollout_actual_frames" in generator_log_dict:
                        actual_frames = generator_log_dict[
                            "rollout_actual_frames"
                        ].float().mean().item()
                        did_reencode = generator_log_dict[
                            "rollout_did_reencode"
                        ].float().mean().item()
                        active_frames = generator_log_dict[
                            "rollout_active_gradient_frames"
                        ].float().mean().item()
                        log_parts.append(f"rollout={actual_frames:.0f}")
                        log_parts.append(f"reencode={did_reencode:.0f}")
                        log_parts.append(f"active_frames={active_frames:.0f}")
                    if self.pd_enabled and "pd_loss" in generator_log_dict:
                        log_parts.append(f"PD_loss={pd_loss_mean:.4f}")
                log_parts.append(f"C_loss={critic_loss_mean:.4f}")
                log_parts.append(f"C_grad={critic_grad_norm_mean:.2f}")
                log_parts.append(f"{avg_time:.1f}s/step")
                log_parts.append(f"elapsed={datetime.timedelta(seconds=int(elapsed))}")
                
                print(" | ".join(log_parts))

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process and should_log:
                current_time = time.time()
                if self.previous_time is None:
                    steps_since_time_log = max(1, self.step - start_step)
                    iter_time = (current_time - train_start_time) / steps_since_time_log
                else:
                    iter_time = (current_time - self.previous_time) / self.log_interval
                self.tb_writer.add_scalar("time/iteration", iter_time, self.step)
                self.tb_writer.add_scalar(
                    "lr/generator",
                    self.generator_optimizer.param_groups[0]['lr'],
                    self.step,
                )
                self.tb_writer.add_scalar(
                    "lr/critic",
                    self.critic_optimizer.param_groups[0]['lr'],
                    self.step,
                )
                if not self.disable_wandb:
                    wandb.log({"per iteration time": iter_time}, step=self.step)
                self.previous_time = current_time

        if self.is_main_process:
            self.tb_writer.flush()
            print(f"Training completed at configured max_steps={max_steps}")
