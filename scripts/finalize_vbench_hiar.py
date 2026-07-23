#!/usr/bin/env python3
"""Validate all VBench outputs and build merged evaluation metadata."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-world-size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    source = json.loads(Path(args.prompt_json).read_text())
    prompts = source.get("prompts", source) if isinstance(source, dict) else source
    manifests = {}
    failures = []
    for manifest_path in sorted((output_dir / "manifests").glob("rank_*.jsonl")):
        for line in manifest_path.read_text().splitlines():
            item = json.loads(line)
            index = int(item["prompt_index"])
            if item.get("status") in {"complete", "already_complete"}:
                manifests[index] = item
            elif item.get("status") == "failed":
                failures.append(item)

    missing = []
    for index, prompt in enumerate(prompts):
        row = manifests.get(index)
        if row is None:
            missing.append(index)
            continue
        video_path = Path(row["video_path"])
        if not video_path.is_file() or video_path.stat().st_size <= 1024:
            missing.append(index)

    eval_info = [
        {
            "prompt_en": prompt["prompt_en"],
            "dimension": prompt.get("dimension", []),
            "video_list": [str(Path(manifests[index]["video_path"]).resolve())]
            if index in manifests
            else [],
            **(
                {"auxiliary_info": prompt["auxiliary_info"]}
                if "auxiliary_info" in prompt
                else {}
            ),
        }
        for index, prompt in enumerate(prompts)
    ]
    (output_dir / "vbench_eval_info.json").write_text(
        json.dumps(eval_info, ensure_ascii=False, indent=2) + "\n"
    )
    summary = {
        "expected_prompts": len(prompts),
        "completed_prompts": len(prompts) - len(missing),
        "missing_prompt_indices": missing,
        "recorded_failures": failures,
        "done_ranks": len(list((output_dir / "rank_done").glob("rank_*.json"))),
        "expected_world_size": args.expected_world_size,
    }
    (output_dir / "sampling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if missing or summary["done_ranks"] != args.expected_world_size:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
