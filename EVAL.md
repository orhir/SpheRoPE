# Evaluation — dataset preparation and captioning

This doc explains how to reproduce the quantitative evaluation used in the
paper:

1. Download the **ODISR** panorama dataset.
2. Caption each panorama with **Qwen3-VL** (long caption) and summarize it
   with **Qwen3** (short, CLIP-token-budget-friendly caption).
3. (Future) run the evaluation script against generated panoramas.

All commands below assume the current working directory is the repo root
and that you have a Python environment with this project installed
(`pip install -r requirements.txt` and `pip install -e .`).

## 1. Download the ODISR dataset

We use the **ODISR** dataset released with
[LAU-Net (CVPR 2021)](https://github.com/wangh-allen/LAU-Net).  Follow the
instructions on the LAU-Net repo to obtain the ODISR splits (the authors
distribute them via a Baidu / Google Drive link).

After download, organize the files like this:

```
/path/to/odisr/
├── training/
│   └── HR/
│       ├── 000.jpg
│       ├── 001.jpg
│       └── ...
├── validation/
│   └── HR/
│       └── ...
└── testing/
    └── HR/
        └── ...
```

Each `HR/` directory should contain the high-resolution equirectangular
panoramas (2:1 aspect ratio, JPEG).

## 2. Caption the dataset

We use a two-stage captioning pipeline:

1. **Long caption** — a detailed 3–5 sentence description produced by
   `Qwen/Qwen3-VL-32B-Instruct`.  Useful for FLUX.2 (which has a
   long-context text encoder and benefits from rich prompts).
2. **Short caption** — a <50-word summary produced by text-only
   `Qwen/Qwen3-8B`.  Required for CLIP-score computation and for FLUX.1
   / SD-family models whose text encoders truncate at 77 tokens.

Both steps are handled by `scripts/caption_dataset.py`:

```bash
python scripts/caption_dataset.py \
    --data-root /path/to/odisr \
    --subdirs training/HR testing/HR \
    --output /path/to/odisr/captions.json
```

The script is resumable — if interrupted, rerun the same command and it
picks up where it left off.

### Output format

A single JSON file with one entry per image:

```json
[
    {
        "dataset": "training_HR",
        "filename": "000.jpg",
        "path": "/path/to/odisr/training/HR/000.jpg",
        "caption": "This 360° equirectangular panoramic photograph captures ...",
        "short_caption": "Overcast evening at Santa Monica Pier, colorful amusement park structures ..."
    },
    ...
]
```

### Tuning

| Flag                | Default                         | Notes |
|---------------------|---------------------------------|-------|
| `--vl-model`        | `Qwen/Qwen3-VL-32B-Instruct`    | Vision-language model (phase 1). Requires ~70 GB VRAM for the 32B variant — swap to a smaller Qwen3-VL if OOM. |
| `--text-model`      | `Qwen/Qwen3-8B`                 | Text-only summarizer (phase 2). ~16 GB VRAM. |
| `--vl-batch-size`   | `4`                             | Per-GPU batch size for captioning. Lower if OOM. |
| `--text-batch-size` | `32`                            | Text summarization batch size. |
| `--num-workers`     | `1`                             | Parallel VL workers. Use `N` = number of model replicas you can fit; each worker gets `floor(n_gpus / num_workers)` GPUs. On an 8×A100-80G node with 4 workers (2 GPUs each) the 32B model runs ~3× faster than single-worker. |
| `--skip-long`       | off                             | Only run short-captioning (requires an existing output with `caption` fields). |
| `--skip-short`      | off                             | Only run long-captioning. |

### Runtime expectations

On a single A100-80G with `--num-workers 1 --vl-batch-size 4`:

| Stage            | ~speed              | 3000-image dataset |
|------------------|---------------------|---------------------|
| Phase 1 (Qwen3-VL-32B) | ~4 img/s      | ~13 min             |
| Phase 2 (Qwen3-8B)     | ~40 img/s     | ~1.5 min            |

Scaling across 8 GPUs with `--num-workers 4` yields roughly 3× on phase 1.

## 3. Evaluation (WIP)

An end-to-end evaluation script (FID / IS / CLIP-score on ODISR
training+testing) is not part of this public release yet.  The generated
panorama directory structure we use internally is
`<gen_root>/<dataset>/<filename>.png`, matching the captions JSON
`dataset` and `filename` fields — so any FID/CLIP pipeline can pair
generated images with ground truth by key-matching on
`(dataset, filename)`.
