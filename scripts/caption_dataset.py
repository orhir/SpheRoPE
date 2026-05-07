# Copyright 2026 The SpheRoPE authors.
#
# This file is part of SpheRoPE and is licensed under the
# Creative Commons Attribution-NonCommercial 4.0 International License
# (CC BY-NC 4.0). See the LICENSE file at the repository root.

"""
Caption ERP 360° panoramas using Qwen3-VL, then summarize to a CLIP-friendly
short caption with Qwen3.  Produces a single JSON file with both fields per
image.

Typical use (one line) — captions every .jpg under <data-root>/<subdir>/HR:

    python caption_dataset.py \\
        --data-root /path/to/odisr \\
        --subdirs training/HR testing/HR \\
        --output odisr_captions.json

Output schema (list of dicts):

    [
        {
            "dataset": "training_HR",   # derived from --subdirs entry
            "filename": "000.jpg",
            "path": "/abs/path/to/000.jpg",
            "caption":       "<long, detailed description ~80 words>",
            "short_caption": "<concise CLIP-friendly <50 word caption>"
        },
        ...
    ]

Resumable: if the output file already contains entries, they are skipped.

The script runs two phases sequentially on the same process:
    1. Vision captioning with Qwen3-VL (default: Qwen/Qwen3-VL-32B-Instruct)
    2. Text-only summarization with Qwen3 (default: Qwen/Qwen3-8B)

For a 3000-image dataset on a single A100 80GB, phase 1 dominates runtime
(~2.5 h @ batch 4) and phase 2 adds ~10 min.  Multi-GPU is supported via
``--num-workers`` which shards the image list across N model replicas.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

LONG_CAPTION_PROMPT = (
    "This is a 360° equirectangular panoramic (ERP) photograph with a 2:1 "
    "aspect ratio capturing the full surrounding environment. Write a "
    "detailed descriptive caption covering: the overall scene type and "
    "setting (indoor/outdoor, specific place type); key objects, furniture, "
    "architectural elements and their spatial arrangement; materials, "
    "textures, colors and surface qualities; lighting conditions "
    "(natural/artificial, direction, intensity, shadows); atmosphere and "
    "mood; and any notable details like decorations, signage, vegetation, "
    "or people. Write a single flowing paragraph of 3-5 sentences. Be "
    "specific, not generic."
)

SHORT_CAPTION_SYSTEM = (
    "You are a caption shortener. Given a detailed panoramic image "
    "description, produce a single concise caption under 50 words that "
    "preserves the key scene type, main objects, and atmosphere. Output "
    "ONLY the shortened caption, nothing else."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def collect_images(data_root: str, subdirs: list[str]) -> list[dict]:
    """Scan ``<data_root>/<subdir>/*.jpg`` for each subdir and return a list
    of task dicts::

        {"dataset": "<subdir-with-slashes->_>", "filename": "...", "path": "..."}

    The ``dataset`` key is the subdir path with ``/`` replaced by ``_`` so it
    can be used as a stable identifier in the output JSON.
    """
    tasks: list[dict] = []
    for subdir in subdirs:
        d = os.path.join(data_root, subdir)
        if not os.path.isdir(d):
            print(f"  [warn] skipping missing directory: {d}")
            continue
        dataset_name = subdir.replace("/", "_").replace("\\", "_")
        paths = sorted(glob.glob(os.path.join(d, "*.jpg")))
        paths += sorted(glob.glob(os.path.join(d, "*.png")))
        for p in paths:
            tasks.append({
                "dataset": dataset_name,
                "filename": os.path.basename(p),
                "path": p,
            })
        print(f"  {dataset_name}: {len(paths)} images")
    return tasks


def load_existing(output_path: str) -> tuple[list[dict], set]:
    """Load a partial output JSON and return (entries, set-of-(dataset,filename) keys)."""
    if not os.path.exists(output_path):
        return [], set()
    with open(output_path) as f:
        data = json.load(f)
    done_keys = {(e["dataset"], e["filename"]) for e in data}
    return data, done_keys


def save_results(output_path: str, results: list[dict]) -> None:
    """Atomic-ish save: write to a tmp file and rename."""
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Phase 1: vision captioning (Qwen3-VL)
# ---------------------------------------------------------------------------

def _vl_worker(
    rank: int,
    gpu_ids: list[int],
    tasks_chunk: list[dict],
    model_name: str,
    max_tokens: int,
    batch_size: int,
    result_queue: "mp.Queue",
) -> None:
    """Worker process: load the VL model onto ``gpu_ids`` and caption a chunk.

    Each task dict is augmented with a ``"caption"`` key, then forwarded via
    ``result_queue``.  ``None`` is pushed on the queue when the worker is done.
    """
    device = f"cuda:{gpu_ids[0]}"
    max_memory = {g: "75GiB" for g in gpu_ids}

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"

    for i in range(0, len(tasks_chunk), batch_size):
        batch = tasks_chunk[i : i + batch_size]

        batch_messages = []
        for task in batch:
            img = Image.open(task["path"]).convert("RGB")
            batch_messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": LONG_CAPTION_PROMPT},
                ],
            }])

        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        captions = processor.batch_decode(
            [o[len(inp):] for inp, o in zip(inputs.input_ids, out)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for task, cap in zip(batch, captions):
            result_queue.put({**task, "caption": cap.strip()})

        done = min(i + batch_size, len(tasks_chunk))
        print(f"  [VL worker {rank} | GPUs {gpu_ids}] {done}/{len(tasks_chunk)}")

    result_queue.put(None)


def run_long_captioning(
    tasks: list[dict],
    existing: list[dict],
    output_path: str,
    model_name: str,
    batch_size: int,
    max_tokens: int,
    num_workers: int,
    save_every: int = 20,
) -> list[dict]:
    """Drive the Qwen3-VL captioning phase across ``num_workers`` processes."""
    done_keys = {(e["dataset"], e["filename"]) for e in existing}
    remaining = [t for t in tasks if (t["dataset"], t["filename"]) not in done_keys]
    print(
        f"\n[Phase 1: long captioning]  total={len(tasks)}  "
        f"done={len(done_keys)}  remaining={len(remaining)}"
    )
    if not remaining:
        return existing

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA device available — captioning requires a GPU.")

    num_workers = max(1, min(num_workers, n_gpus))
    gpus_per_worker = max(1, n_gpus // num_workers)
    gpu_assignments = [
        list(range(i * gpus_per_worker, (i + 1) * gpus_per_worker))
        for i in range(num_workers)
    ]
    # Interleave chunks for balanced work even when prefixes are correlated.
    chunks = [remaining[i::num_workers] for i in range(num_workers)]

    print(
        f"  workers={num_workers}  gpus_per_worker={gpus_per_worker}  "
        f"assignment={gpu_assignments}  chunk_sizes={[len(c) for c in chunks]}"
    )

    results = list(existing)

    if num_workers == 1:
        # Single-worker path: run inline, no multiprocessing overhead.
        queue: list[dict] = []

        class _InlineQueue:  # minimal stand-in for mp.Queue
            def put(self, x):
                queue.append(x)

        _vl_worker(
            rank=0,
            gpu_ids=gpu_assignments[0],
            tasks_chunk=chunks[0],
            model_name=model_name,
            max_tokens=max_tokens,
            batch_size=batch_size,
            result_queue=_InlineQueue(),
        )
        for item in queue:
            if item is None:
                continue
            results.append(item)
            if len(results) % save_every == 0:
                save_results(output_path, results)
    else:
        mp.set_start_method("spawn", force=True)
        result_queue: mp.Queue = mp.Queue()
        procs = []
        for rank in range(num_workers):
            proc = mp.Process(
                target=_vl_worker,
                args=(rank, gpu_assignments[rank], chunks[rank],
                      model_name, max_tokens, batch_size, result_queue),
            )
            proc.start()
            procs.append(proc)

        done_workers = 0
        while done_workers < num_workers:
            item = result_queue.get()
            if item is None:
                done_workers += 1
                continue
            results.append(item)
            if len(results) % save_every == 0:
                save_results(output_path, results)

        for proc in procs:
            proc.join()

    save_results(output_path, results)
    print(f"  Phase 1 complete: {len(results)} entries → {output_path}")
    return results


# ---------------------------------------------------------------------------
# Phase 2: short-caption summarization (text-only Qwen3)
# ---------------------------------------------------------------------------

def run_short_captioning(
    entries: list[dict],
    output_path: str,
    model_name: str,
    batch_size: int,
    max_tokens: int,
) -> None:
    """Add a ``short_caption`` field to every entry that lacks one, in place."""
    remaining = [e for e in entries if "short_caption" not in e or not e["short_caption"]]
    print(
        f"\n[Phase 2: short captioning]  total={len(entries)}  "
        f"remaining={len(remaining)}"
    )
    if not remaining:
        return

    print(f"  loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        messages_batch = [
            [
                {"role": "system", "content": SHORT_CAPTION_SYSTEM},
                {"role": "user", "content": e["caption"]},
            ]
            for e in batch
        ]

        texts = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for m in messages_batch
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, temperature=None, top_p=None,
            )

        for j, entry in enumerate(batch):
            new_tokens = out[j][inputs.input_ids.shape[1]:]
            short = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            entry["short_caption"] = short

        save_results(output_path, entries)
        print(
            f"  [Phase 2] {i + len(batch)}/{len(remaining)}  "
            f"{batch[-1]['dataset']}/{batch[-1]['filename']}: "
            f"{batch[-1]['short_caption'][:80]}..."
        )

    print(f"  Phase 2 complete: {len(entries)} entries → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Caption an ERP panorama dataset with Qwen3-VL (long) + Qwen3 (short).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True,
                   help="Root directory containing the image subdirectories.")
    p.add_argument("--subdirs", nargs="+", default=["training/HR", "testing/HR"],
                   help="Image subdirectories (relative to --data-root) to caption.")
    p.add_argument("--output", required=True,
                   help="Output JSON path.  Existing entries are resumed.")

    p.add_argument("--vl-model", default="Qwen/Qwen3-VL-32B-Instruct",
                   help="HuggingFace id of the vision-language model (phase 1).")
    p.add_argument("--text-model", default="Qwen/Qwen3-8B",
                   help="HuggingFace id of the text-only LLM (phase 2).")

    p.add_argument("--vl-batch-size", type=int, default=4)
    p.add_argument("--vl-max-tokens", type=int, default=300)
    p.add_argument("--text-batch-size", type=int, default=32)
    p.add_argument("--text-max-tokens", type=int, default=80)

    p.add_argument("--num-workers", type=int, default=1,
                   help="Number of parallel VL workers (one model replica per "
                        "worker).  Requires --num-workers <= number of CUDA "
                        "devices.  Each worker uses floor(n_gpus/num_workers) GPUs.")

    p.add_argument("--skip-long", action="store_true",
                   help="Skip phase 1 (use an existing captioned JSON and only "
                        "run phase 2).")
    p.add_argument("--skip-short", action="store_true",
                   help="Skip phase 2 (only produce long captions).")

    args = p.parse_args()

    output_path = str(Path(args.output).expanduser().resolve())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Scanning images under {args.data_root}:")
    tasks = collect_images(args.data_root, args.subdirs)
    if not tasks:
        raise SystemExit("No images found — check --data-root and --subdirs.")

    existing, _ = load_existing(output_path)
    print(f"Existing entries in {output_path}: {len(existing)}")

    if not args.skip_long:
        existing = run_long_captioning(
            tasks=tasks,
            existing=existing,
            output_path=output_path,
            model_name=args.vl_model,
            batch_size=args.vl_batch_size,
            max_tokens=args.vl_max_tokens,
            num_workers=args.num_workers,
        )
    else:
        print("[Phase 1] skipped (--skip-long)")

    if not args.skip_short:
        # Reload in case the phase-1 save wrote something new
        existing, _ = load_existing(output_path)
        run_short_captioning(
            entries=existing,
            output_path=output_path,
            model_name=args.text_model,
            batch_size=args.text_batch_size,
            max_tokens=args.text_max_tokens,
        )
    else:
        print("[Phase 2] skipped (--skip-short)")

    print("\nDone.")


if __name__ == "__main__":
    main()
