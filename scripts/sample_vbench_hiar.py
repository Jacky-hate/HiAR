#!/usr/bin/env python3
"""Resumable one-sample-per-GPU VBench inference worker for HiAR."""

import argparse
import json
import os
import random
import re
import time
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from utils.memory import get_cuda_free_memory_gb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-output-frames", type=int, default=21)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--indices",
        default=None,
        help="Optional comma-separated global prompt indices for smoke tests",
    )
    return parser.parse_args()


def load_prompts(path):
    payload = json.loads(Path(path).read_text())
    prompts = payload.get("prompts", payload) if isinstance(payload, dict) else payload
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt JSON must contain a non-empty list")
    required = {"prompt_id", "prompt_en", "gen_prompt"}
    for index, item in enumerate(prompts):
        missing = required.difference(item)
        if missing:
            raise ValueError(f"prompt {index} is missing keys: {sorted(missing)}")
    ids = [str(item["prompt_id"]) for item in prompts]
    names = [item["prompt_en"] for item in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("prompt_id values are not unique")
    if len(set(names)) != len(names):
        raise ValueError("prompt_en values are not unique; filenames would collide")
    return prompts


def sanitize_filename(text):
    text = str(text).replace("/", "-").replace("\\", "-")
    text = re.sub(r"[\r\n\x00]+", " ", text).strip()
    return text


def set_prompt_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_jsonl(path, payload):
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def existing_completed_indices(path):
    completed = set()
    if not path.is_file():
        return completed
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("status") in {"complete", "already_complete"}:
            completed.add(int(item["prompt_index"]))
    return completed


def main():
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"invalid rank {args.rank} for world_size={args.world_size}")
    output_dir = Path(args.output_dir).resolve()
    videos_dir = output_dir / "videos"
    manifests_dir = output_dir / "manifests"
    logs_dir = output_dir / "logs"
    done_dir = output_dir / "rank_done"
    for directory in (videos_dir, manifests_dir, logs_dir, done_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / f"rank_{args.rank:02d}.jsonl"
    prompts = load_prompts(args.prompt_json)
    if args.indices:
        indices = [int(piece) for piece in args.indices.split(",")]
    else:
        indices = list(range(args.rank, len(prompts), args.world_size))
    if any(index < 0 or index >= len(prompts) for index in indices):
        raise IndexError(f"requested index outside [0, {len(prompts)})")

    print(
        f"WORKER_START rank={args.rank} world_size={args.world_size} "
        f"prompts_total={len(prompts)} assigned={len(indices)} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    free_gib = get_cuda_free_memory_gb(device)
    if free_gib < 40:
        raise RuntimeError(
            f"rank {args.rank} has only {free_gib:.1f} GiB free; refusing low-memory sampling"
        )

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    num_frame_per_block = int(config.get("num_frame_per_block", 1))
    if args.num_output_frames % num_frame_per_block:
        raise ValueError(
            f"num_output_frames={args.num_output_frames} must be divisible by "
            f"num_frame_per_block={num_frame_per_block}"
        )
    init_start = time.monotonic()
    pipeline = CausalInferencePipeline(config, device=device)
    if args.num_output_frames > 21:
        local_attn_size = 21
        pipeline.local_attn_size = local_attn_size
        pipeline.generator.model.local_attn_size = local_attn_size
        for block in pipeline.generator.model.blocks:
            block.self_attn.local_attn_size = local_attn_size
            block.self_attn.max_attention_size = local_attn_size * 1560
        print(
            f"LOCAL_ATTENTION rank={args.rank} window_latent_frames={local_attn_size} "
            f"output_latent_frames={args.num_output_frames}",
            flush=True,
        )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    if "generator_ema" not in checkpoint:
        raise KeyError("exported checkpoint does not contain generator_ema")
    raw_state = checkpoint["generator_ema"]
    clean_state = {
        key.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", ""): value
        for key, value in raw_state.items()
    }
    incompatible = pipeline.generator.load_state_dict(clean_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict EMA load failed: {incompatible}")
    del checkpoint, raw_state, clean_state

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    torch.cuda.synchronize()
    print(
        f"MODEL_READY rank={args.rank} init_sec={time.monotonic() - init_start:.2f} "
        f"free_gib={get_cuda_free_memory_gb(device):.2f}",
        flush=True,
    )

    completed_before = existing_completed_indices(manifest_path)
    failures = []
    worker_start = time.monotonic()
    completed_this_run = 0
    for ordinal, prompt_index in enumerate(indices):
        item = prompts[prompt_index]
        final_path = videos_dir / f"{sanitize_filename(item['prompt_en'])}-0.mp4"
        prompt_seed = args.seed + prompt_index
        if final_path.is_file() and final_path.stat().st_size > 1024:
            if prompt_index not in completed_before:
                append_jsonl(
                    manifest_path,
                    {
                        "status": "already_complete",
                        "rank": args.rank,
                        "prompt_index": prompt_index,
                        "prompt_id": str(item["prompt_id"]),
                        "prompt_en": item["prompt_en"],
                        "gen_prompt": item["gen_prompt"],
                        "dimension": item.get("dimension", []),
                        "auxiliary_info": item.get("auxiliary_info"),
                        "sample_index": 0,
                        "seed": prompt_seed,
                        "num_output_latent_frames": args.num_output_frames,
                        "num_frames": (args.num_output_frames - 1) * 4 + 1,
                        "duration_sec": ((args.num_output_frames - 1) * 4 + 1)
                        / args.fps,
                        "video_path": str(final_path),
                        "video_bytes": final_path.stat().st_size,
                    },
                )
            print(
                f"SKIP rank={args.rank} prompt_index={prompt_index} path={final_path}",
                flush=True,
            )
            continue

        success = False
        for attempt in range(args.retries + 1):
            sample_start = time.monotonic()
            temp_path = final_path.with_name(
                f".{final_path.stem}.rank{args.rank:02d}.pid{os.getpid()}.tmp.mp4"
            )
            try:
                set_prompt_seed(prompt_seed)
                noise = torch.randn(
                    [1, args.num_output_frames, 16, 60, 104],
                    device=device,
                    dtype=torch.bfloat16,
                )
                with torch.inference_mode():
                    video, _ = pipeline.inference_hybrid(
                        noise=noise,
                        text_prompts=[item["gen_prompt"]],
                        return_latents=True,
                        low_memory=False,
                        frame_first_steps=0,
                    )
                video = rearrange(video, "b t c h w -> b t h w c")[0]
                video_array = (
                    video.mul(255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                )
                del video, noise
                pipeline.vae.model.clear_cache()
                imageio.mimwrite(
                    temp_path,
                    video_array,
                    fps=args.fps,
                    quality=8,
                    macro_block_size=None,
                )
                del video_array
                if not temp_path.is_file() or temp_path.stat().st_size <= 1024:
                    raise RuntimeError(f"encoded video is empty: {temp_path}")
                os.replace(temp_path, final_path)
                elapsed = time.monotonic() - sample_start
                completed_this_run += 1
                append_jsonl(
                    manifest_path,
                    {
                        "status": "complete",
                        "rank": args.rank,
                        "prompt_index": prompt_index,
                        "prompt_id": str(item["prompt_id"]),
                        "prompt_en": item["prompt_en"],
                        "gen_prompt": item["gen_prompt"],
                        "dimension": item.get("dimension", []),
                        "auxiliary_info": item.get("auxiliary_info"),
                        "sample_index": 0,
                        "seed": prompt_seed,
                        "fps": args.fps,
                        "num_output_latent_frames": args.num_output_frames,
                        "num_frames": (args.num_output_frames - 1) * 4 + 1,
                        "duration_sec": ((args.num_output_frames - 1) * 4 + 1)
                        / args.fps,
                        "height": 480,
                        "width": 832,
                        "elapsed_sec": elapsed,
                        "video_path": str(final_path),
                        "video_bytes": final_path.stat().st_size,
                    },
                )
                total_done = ordinal + 1
                rate = completed_this_run / max(time.monotonic() - worker_start, 1e-6)
                remaining = len(indices) - total_done
                eta_sec = remaining / rate if rate > 0 else -1
                print(
                    f"COMPLETE rank={args.rank} prompt_index={prompt_index} "
                    f"ordinal={total_done}/{len(indices)} elapsed_sec={elapsed:.2f} "
                    f"worker_eta_sec={eta_sec:.1f} bytes={final_path.stat().st_size}",
                    flush=True,
                )
                success = True
                break
            except Exception as error:
                if temp_path.exists():
                    temp_path.unlink()
                pipeline.vae.model.clear_cache()
                torch.cuda.empty_cache()
                print(
                    f"ATTEMPT_FAILED rank={args.rank} prompt_index={prompt_index} "
                    f"attempt={attempt + 1}/{args.retries + 1} error={error}",
                    flush=True,
                )
                traceback.print_exc()
        if not success:
            failure = {
                "status": "failed",
                "rank": args.rank,
                "prompt_index": prompt_index,
                "prompt_id": str(item["prompt_id"]),
                "prompt_en": item["prompt_en"],
                "gen_prompt": item["gen_prompt"],
                "seed": prompt_seed,
            }
            failures.append(failure)
            append_jsonl(manifest_path, failure)

    done_payload = {
        "rank": args.rank,
        "assigned": len(indices),
        "failures": len(failures),
        "completed_this_run": completed_this_run,
        "elapsed_sec": time.monotonic() - worker_start,
    }
    done_path = done_dir / f"rank_{args.rank:02d}.json"
    temp_done = done_path.with_name(f".{done_path.name}.tmp.{os.getpid()}")
    temp_done.write_text(json.dumps(done_payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp_done, done_path)
    print(f"WORKER_DONE {json.dumps(done_payload, sort_keys=True)}", flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
