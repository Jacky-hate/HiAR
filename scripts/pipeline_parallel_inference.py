"""
Multi-GPU Pipeline Parallel Inference for HiAR.

Uses diagonal scheduling across GPUs, with one denoising step per stage.
The default mode (DAC) splits each step into Denoise-AddNoise (DA) and
CacheUpdate (C), overlapping C with the next stage's DA via point-to-point
communication.  This yields 1.1-1.7x wall-clock speedup depending on
sequence length.

Usage:
    torchrun --nproc_per_node=4 scripts/pipeline_parallel_inference.py \
        --config_path configs/hiar.yaml \
        --checkpoint_path ckpts/hiar.pt \
        --prompt "A cat watching rain" \
        --num_output_frames 81

Options:
    --no_dac        Disable DAC optimisation (use barrier+broadcast baseline)
    --profile       Emit per-rank D/A/C timing JSON files
    --profile_output PREFIX   Path prefix for profile JSON files

The number of GPUs must equal the number of denoising steps (default: 4).
"""

import argparse
import json as _json
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from utils.misc import set_seed


def _clean_state_dict_keys(state_dict):
    return {
        k.replace("_fsdp_wrapped_module.", "")
         .replace("_checkpoint_wrapped_module.", "")
         .replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }


def _extract_generator_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    for key in ("generator_ema", "generator", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint

    available_keys = ", ".join(map(str, checkpoint.keys()))
    raise KeyError(
        "Unable to locate generator weights in checkpoint. "
        f"Available top-level keys: {available_keys}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU pipeline parallel inference for HiAR")
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None,
                    help="Text file with one prompt per line")
    p.add_argument("--output_path", type=str,
                    default="outputs/pipeline_output.mp4")
    p.add_argument("--num_output_frames", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise_seed", type=int, default=123,
                    help="Shared seed for SDE noise consistency across GPUs")
    p.add_argument("--vae_decode_chunk_size", type=int, default=8,
                    help="Number of latent frames to decode per VAE chunk when saving video")
    p.add_argument("--no_dac", action="store_true",
                    help="Disable DAC optimisation; use barrier+broadcast")
    p.add_argument("--profile", action="store_true",
                    help="Profile D/A/C component timings via CUDA events")
    p.add_argument("--profile_output", type=str, default=None,
                    help="Path prefix for per-rank JSON profile files")
    # Legacy alias kept for backward compatibility
    p.add_argument("--dac_optimize", action="store_true",
                    help=argparse.SUPPRESS)
    return p.parse_args()


def _get_latent_shape(config):
    image_or_video_shape = getattr(config, "image_or_video_shape", None)
    if image_or_video_shape is None or len(image_or_video_shape) != 5:
        raise ValueError(
            "Config must provide image_or_video_shape=[B, F, C, H, W] for pipeline inference"
        )
    return tuple(int(v) for v in image_or_video_shape[2:])


def _align_num_output_frames(requested_num_output_frames, pipeline):
    nfpb = pipeline.num_frame_per_block
    if pipeline.independent_first_frame:
        remainder = (requested_num_output_frames - 1) % nfpb
    else:
        remainder = requested_num_output_frames % nfpb
    if remainder == 0:
        return requested_num_output_frames

    aligned_num_output_frames = requested_num_output_frames + (nfpb - remainder)
    print(
        f"num_output_frames ({requested_num_output_frames}) is not divisible by "
        f"num_frame_per_block ({nfpb}); rounding up to {aligned_num_output_frames}"
    )
    return aligned_num_output_frames


def _prepare_runtime_spec(pipeline, config):
    pipeline.latent_shape = _get_latent_shape(config)
    _, height, width = pipeline.latent_shape
    pipeline._update_runtime_cache_spec(height, width)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_pipeline(args, config, device):
    """Build pipeline and load checkpoint.

    Uses ``mmap=True`` so each rank avoids a full copy of the checkpoint in
    RAM.  Ranks load sequentially so the OS page-cache is warm for later
    ranks, bounding peak RSS.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    pipeline = CausalInferencePipeline(config, device=device)

    for r in range(world_size):
        if r == rank:
            checkpoint = torch.load(
                args.checkpoint_path, map_location="cpu", mmap=True)
            raw_sd = _extract_generator_state_dict(checkpoint)
            clean_sd = _clean_state_dict_keys(raw_sd)
            pipeline.generator.load_state_dict(clean_sd)
            del checkpoint, raw_sd, clean_sd
            gc.collect()
        if dist.is_initialized():
            dist.barrier()

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline


# ---------------------------------------------------------------------------
# Attention windowing for long sequences
# ---------------------------------------------------------------------------
def _setup_local_attention(pipeline, num_output_frames):
    if num_output_frames > 21:
        if pipeline.frame_seq_length is None:
            raise ValueError("frame_seq_length must be initialized before enabling local attention")
        la = 21
        pipeline.local_attn_size = la
        pipeline.generator.model.local_attn_size = la
        for blk in pipeline.generator.model.blocks:
            blk.self_attn.local_attn_size = la
            blk.self_attn.max_attention_size = la * pipeline.frame_seq_length




# ---------------------------------------------------------------------------
# KV / cross-attention cache initialisation
# ---------------------------------------------------------------------------
def _init_caches(pipeline, batch_size, device):
    if pipeline.frame_seq_length is None:
        raise ValueError("frame_seq_length must be initialized before cache allocation")

    if pipeline.local_attn_size != -1:
        kv_sz = pipeline.local_attn_size * pipeline.frame_seq_length
    else:
        kv_sz = 32760

    kv_cache, crossattn_cache = [], []
    for _ in range(pipeline.num_transformer_blocks):
        kv_cache.append({
            "k": torch.zeros(
                [batch_size, kv_sz, pipeline.num_attention_heads, pipeline.attention_head_dim],
                dtype=torch.bfloat16, device=device),
            "v": torch.zeros(
                [batch_size, kv_sz, pipeline.num_attention_heads, pipeline.attention_head_dim],
                dtype=torch.bfloat16, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long,
                                              device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long,
                                             device=device),
        })
        crossattn_cache.append({
            "k": torch.zeros(
                [batch_size, pipeline.text_context_len, pipeline.num_attention_heads, pipeline.attention_head_dim],
                dtype=torch.bfloat16, device=device),
            "v": torch.zeros(
                [batch_size, pipeline.text_context_len, pipeline.num_attention_heads, pipeline.attention_head_dim],
                dtype=torch.bfloat16, device=device),
            "is_init": False,
        })
    return kv_cache, crossattn_cache


# ---------------------------------------------------------------------------
# Baseline: barrier + broadcast
# ---------------------------------------------------------------------------
def _run_baseline(pipeline, args, prompt, device, rank, my_stage, num_stages,
                  num_blocks, num_frames, noise_bank, profile):
    batch_size = 1
    nfpb = pipeline.num_frame_per_block
    num_channels, height, width = pipeline.latent_shape

    noise = torch.randn(
        [batch_size, num_frames, num_channels, height, width],
        device=device, dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(args.seed))

    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
    kv_cache, crossattn_cache = _init_caches(pipeline, batch_size, device)

    block_start = [i * nfpb for i in range(num_blocks)]
    noisy_inputs = [noise[:, sf:sf + nfpb].contiguous()
                    for sf in block_start]
    output = torch.zeros_like(noise)
    prof_entries = []

    t0 = time.time()
    for diag in range(num_stages + num_blocks - 1):
        bidx = diag - my_stage
        has_work = 0 <= bidx < num_blocks

        if profile and has_work:
            ev = {k: torch.cuda.Event(enable_timing=True)
                  for k in ("ds", "de", "as", "ae", "cs", "ce")}

        if has_work:
            sf = block_start[bidx]
            nf = nfpb
            ts = pipeline.denoising_step_list[my_stage]
            last = my_stage == num_stages - 1
            timestep = (torch.ones([batch_size, nf], device=device,
                                   dtype=torch.int64) * ts)

            if profile: ev["ds"].record()
            _, denoised = pipeline.generator(
                noisy_image_or_video=noisy_inputs[bidx],
                conditional_dict=conditional_dict, timestep=timestep,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=sf * pipeline.frame_seq_length)
            if profile: ev["de"].record()

            if profile: ev["as"].record()
            if not last:
                nxt = pipeline.denoising_step_list[my_stage + 1]
                sde = noise_bank[(my_stage, bidx)]
                noisy_inputs[bidx] = pipeline.scheduler.add_noise(
                    denoised.flatten(0, 1), sde.flatten(0, 1),
                    nxt * torch.ones([batch_size * nf], device=device,
                                     dtype=torch.long)
                ).unflatten(0, denoised.shape[:2])
                ci = (denoised if pipeline.always_clean_context
                      else noisy_inputs[bidx])
                ct = (pipeline.args.context_noise
                      if pipeline.always_clean_context else nxt)
            else:
                output[:, sf:sf + nf] = denoised
                ci, ct = denoised, pipeline.args.context_noise
            if profile: ev["ae"].record()

            ctx_ts = (torch.ones([batch_size, nf], device=device,
                                 dtype=torch.int64) * ct)
            if profile: ev["cs"].record()
            pipeline.generator(
                noisy_image_or_video=ci,
                conditional_dict=conditional_dict, timestep=ctx_ts,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=sf * pipeline.frame_seq_length)
            if profile: ev["ce"].record()

        torch.cuda.synchronize()
        dist.barrier()
        for stage in range(num_stages):
            b = diag - stage
            if 0 <= b < num_blocks:
                dist.broadcast(noisy_inputs[b], src=stage)

        if profile and has_work:
            prof_entries.append({"diag": diag, "bidx": bidx, "ev": ev})

    torch.cuda.synchronize()
    dist.broadcast(output, src=num_stages - 1)
    return output, time.time() - t0, prof_entries


# ---------------------------------------------------------------------------
# DAC: point-to-point, DA/C split  (default)
# ---------------------------------------------------------------------------
def _run_dac(pipeline, args, prompt, device, rank, my_stage, num_stages,
             num_blocks, num_frames, noise_bank, profile):
    batch_size = 1
    nfpb = pipeline.num_frame_per_block
    num_channels, height, width = pipeline.latent_shape

    noise = torch.randn(
        [batch_size, num_frames, num_channels, height, width],
        device=device, dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(args.seed))

    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
    kv_cache, crossattn_cache = _init_caches(pipeline, batch_size, device)

    block_start = [i * nfpb for i in range(num_blocks)]
    noisy_inputs = [noise[:, sf:sf + nfpb].contiguous()
                    for sf in block_start]
    output = torch.zeros_like(noise)
    prof_entries = []
    last = my_stage == num_stages - 1

    t0 = time.time()
    for bidx in range(num_blocks):
        sf = block_start[bidx]
        nf = nfpb

        if profile:
            ev = {k: torch.cuda.Event(enable_timing=True)
                  for k in ("ds", "de", "as", "ae", "cs", "ce",
                            "recv_s", "recv_e", "send_s", "send_e")}

        # Receive from previous stage
        if my_stage > 0:
            if profile: ev["recv_s"].record()
            dist.recv(noisy_inputs[bidx], src=my_stage - 1)
            if profile: ev["recv_e"].record()

        ts = pipeline.denoising_step_list[my_stage]
        timestep = (torch.ones([batch_size, nf], device=device,
                               dtype=torch.int64) * ts)

        # [D] Denoise
        if profile: ev["ds"].record()
        _, denoised = pipeline.generator(
            noisy_image_or_video=noisy_inputs[bidx],
            conditional_dict=conditional_dict, timestep=timestep,
            kv_cache=kv_cache, crossattn_cache=crossattn_cache,
            current_start=sf * pipeline.frame_seq_length)
        if profile: ev["de"].record()

        # [A] AddNoise
        if profile: ev["as"].record()
        if not last:
            nxt = pipeline.denoising_step_list[my_stage + 1]
            sde = noise_bank[(my_stage, bidx)]
            noisy_inputs[bidx] = pipeline.scheduler.add_noise(
                denoised.flatten(0, 1), sde.flatten(0, 1),
                nxt * torch.ones([batch_size * nf], device=device,
                                 dtype=torch.long)
            ).unflatten(0, denoised.shape[:2])
            ci = (denoised if pipeline.always_clean_context
                  else noisy_inputs[bidx])
            ct = (pipeline.args.context_noise
                  if pipeline.always_clean_context else nxt)
        else:
            output[:, sf:sf + nf] = denoised
            ci, ct = denoised, pipeline.args.context_noise
        if profile: ev["ae"].record()

        # Send to next stage immediately after DA (before C)
        if not last:
            if profile: ev["send_s"].record()
            dist.send(noisy_inputs[bidx], dst=my_stage + 1)
            if profile: ev["send_e"].record()

        # [C] Cache update — overlaps with next stage's DA
        ctx_ts = (torch.ones([batch_size, nf], device=device,
                             dtype=torch.int64) * ct)
        if profile: ev["cs"].record()
        if bidx < num_blocks - 1:
            pipeline.generator(
                noisy_image_or_video=ci,
                conditional_dict=conditional_dict, timestep=ctx_ts,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=sf * pipeline.frame_seq_length)
        if profile: ev["ce"].record()

        if profile:
            prof_entries.append(
                {"diag": my_stage + bidx, "bidx": bidx, "ev": ev})

    torch.cuda.synchronize()
    dist.broadcast(output, src=num_stages - 1)
    return output, time.time() - t0, prof_entries


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------
def _resolve_profile(entries, rank, my_stage, mode):
    """Convert CUDA events to milliseconds."""
    torch.cuda.synchronize()
    result = {"mode": mode, "rank": rank, "stage": my_stage, "diagonals": []}
    for e in entries:
        ev = e["ev"]
        def _elapsed(s, e):
            try:
                return ev[s].elapsed_time(ev[e])
            except (RuntimeError, ValueError):
                return 0.0
        row = {"diag": e["diag"], "block_idx": e["bidx"],
               "d_ms": _elapsed("ds", "de"),
               "a_ms": _elapsed("as", "ae"),
               "c_ms": _elapsed("cs", "ce")}
        if "recv_s" in ev:
            row["recv_ms"] = _elapsed("recv_s", "recv_e")
        if "send_s" in ev:
            row["send_ms"] = _elapsed("send_s", "send_e")
        result["diagonals"].append(row)
    return result


def _save_output_video_streaming(pipeline, latent, opath, chunk_num_frames, fps=16):
    if chunk_num_frames <= 0:
        raise ValueError(f"vae_decode_chunk_size must be positive, got {chunk_num_frames}")

    total_latent_frames = latent.shape[1]
    total_chunks = (total_latent_frames + chunk_num_frames - 1) // chunk_num_frames
    saved_frames = 0

    os.makedirs(os.path.dirname(opath) or ".", exist_ok=True)
    writer = imageio.get_writer(opath, fps=fps, quality=8)
    try:
        for chunk_idx, pixel_chunk in enumerate(
            pipeline.vae.iter_decode_to_pixel_chunks(
                latent, chunk_num_frames=chunk_num_frames
            ),
            start=1,
        ):
            pixel_chunk = (pixel_chunk * 0.5 + 0.5).clamp_(0, 1)
            frame_chunk = rearrange(pixel_chunk[0], "t c h w -> t h w c")
            frame_chunk = (255.0 * frame_chunk).clamp_(0, 255).to(torch.uint8).cpu().numpy()
            for frame in frame_chunk:
                writer.append_data(frame)
            saved_frames += frame_chunk.shape[0]
            if chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % 20 == 0:
                print(
                    f"VAE decode/save chunk {chunk_idx}/{total_chunks}, "
                    f"written {saved_frames} pixel frames"
                )
            del pixel_chunk, frame_chunk
    finally:
        writer.close()
        pipeline.vae.model.clear_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.set_grad_enabled(False)
        set_seed(args.seed)

        config = OmegaConf.load(args.config_path)
        default_config = OmegaConf.load("configs/default_config.yaml")
        config = OmegaConf.merge(default_config, config)

        num_stages = len(config.denoising_step_list)
        assert world_size == num_stages, (
            f"GPUs ({world_size}) must equal denoising steps ({num_stages})")
        my_stage = rank

        # DAC is default; --no_dac disables it
        use_dac = not args.no_dac or args.dac_optimize

        if rank == 0:
            tag = "baseline" if not use_dac else "DAC"
            print(f"Pipeline parallel [{tag}]: "
                  f"{num_stages} stages on {world_size} GPUs")

        pipeline = _load_pipeline(args, config, device)
        args.num_output_frames = _align_num_output_frames(args.num_output_frames, pipeline)
        _prepare_runtime_spec(pipeline, config)
        _setup_local_attention(pipeline, args.num_output_frames)

        if args.prompt:
            prompts = [args.prompt]
        elif args.data_path:
            with open(args.data_path) as f:
                prompts = [l.strip() for l in f if l.strip()]
        else:
            raise ValueError("--prompt or --data_path required")

        defer_rank0_save = len(prompts) == 1
        deferred_save_payload = None

        dist.barrier()

        for pidx, prompt in enumerate(prompts):
            if rank == 0:
                print(f"\n[{pidx + 1}/{len(prompts)}] {prompt[:80]}...")

            num_frames = args.num_output_frames
            nfpb = pipeline.num_frame_per_block
            num_blocks = num_frames // nfpb

            noise_bank = {}
            gen = torch.Generator(device=device).manual_seed(args.noise_seed)
            num_channels, height, width = pipeline.latent_shape
            bshape = [1, nfpb, num_channels, height, width]
            for s in range(num_stages):
                for b in range(num_blocks):
                    noise_bank[(s, b)] = torch.randn(
                        bshape, device=device, dtype=torch.bfloat16,
                        generator=gen)

            run_fn = _run_dac if use_dac else _run_baseline
            output, elapsed, prof_entries = run_fn(
                pipeline, args, prompt, device, rank, my_stage, num_stages,
                num_blocks, num_frames, noise_bank, args.profile)

            if rank == 0:
                print(f"Inference time: {elapsed:.2f}s "
                      f"({num_blocks} blocks, {num_stages} stages)")

            # Profiling summary
            if args.profile and prof_entries:
                mode = "dac" if use_dac else "baseline"
                prof = _resolve_profile(prof_entries, rank, my_stage, mode)
                prof["num_stages"] = num_stages
                prof["num_blocks"] = num_blocks
                prof["num_frames"] = num_frames
                prof["total_time_s"] = elapsed

                if args.profile_output:
                    out = f"{args.profile_output}_{mode}_rank{rank}.json"
                    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                    with open(out, "w") as f:
                        _json.dump(prof, f, indent=2)

                if rank == 0:
                    d_t = [r["d_ms"] for r in prof["diagonals"]]
                    a_t = [r["a_ms"] for r in prof["diagonals"]]
                    c_t = [r["c_ms"] for r in prof["diagonals"]]
                    print(f"\n  Profile [{mode}] (rank 0 / stage 0):")
                    print(f"    D (Denoise) : avg {np.mean(d_t):7.1f} ms, "
                          f"total {np.sum(d_t):8.0f} ms")
                    print(f"    A (AddNoise): avg {np.mean(a_t):7.1f} ms, "
                          f"total {np.sum(a_t):8.0f} ms")
                    print(f"    C (Cache)   : avg {np.mean(c_t):7.1f} ms, "
                          f"total {np.sum(c_t):8.0f} ms")
                    dac_sum = [d + a + c for d, a, c in zip(d_t, a_t, c_t)]
                    print(f"    DA+C / step : avg {np.mean(dac_sum):7.1f} ms")

            # Save video (rank 0 only)
            if rank == 0:
                opath = args.output_path
                if len(prompts) > 1:
                    base, ext = os.path.splitext(opath)
                    opath = f"{base}_{pidx}{ext}"

                if defer_rank0_save:
                    deferred_save_payload = (output, opath)
                else:
                    _save_output_video_streaming(
                        pipeline,
                        output,
                        opath,
                        chunk_num_frames=args.vae_decode_chunk_size,
                    )
                    print(f"Saved: {opath}")

            del noise_bank, prof_entries, output
            gc.collect()

            if not defer_rank0_save:
                dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0 and deferred_save_payload is not None:
        output, opath = deferred_save_payload
        if hasattr(pipeline, "generator"):
            delattr(pipeline, "generator")
        if hasattr(pipeline, "text_encoder"):
            delattr(pipeline, "text_encoder")
        gc.collect()
        if output.is_cuda:
            torch.cuda.empty_cache()

        _save_output_video_streaming(
            pipeline,
            output,
            opath,
            chunk_num_frames=args.vae_decode_chunk_size,
        )
        print(f"Saved: {opath}")
        del output
        gc.collect()


if __name__ == "__main__":
    main()
