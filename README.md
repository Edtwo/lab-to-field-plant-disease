# Lab-to-Field on a Budget

This repo is a seeded, edge-aware benchmark of plant-disease classifiers under the PlantVillage → PlantDoc shift. It accompanies a HARVEST-Vision @ WACV 2027 workshop submission.

Every number in the paper comes from `results/`. `scripts/summarize.py` builds every table and figure from those files.

## Environment (as used)
- Windows 11, 1× RTX 5080 16 GB, 32 GB RAM, Python 3.14.7, CUDA 12.8 PyTorch build.
- `pip install -r requirements.txt` (pinned). Install PyTorch from the cu128 index:
  `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128`

> **Windows note:** keep the venv and the data in short paths (we used `C:\Users\<you>\.venvs\hv27` and `C:\Users\<you>\hv27_data`). PlantVillage file names are long, and onnxruntime fails to load its DLL from long paths.

## Reproduce
```bash
export HV_CACHE=/path/to/hv27_data/cache        # where the cached arrays go
# 1) data (about 3.2 GB download; see data/README.md for sources and licenses)
python scripts/download_data.py --raw /path/to/hv27_data/raw
# 2) cache 224x224 arrays + near-duplicate check (about 2 min)
python scripts/prepare_data.py --raw /path/to/hv27_data/raw --out $HV_CACHE
# 3) smoke test (about 5 min)
python src/train.py --model mnv3 --data pv --seed 0 --smoke
# 4) full matrix: 62 jobs, sequential, resumable (skips finished runs)
python scripts/run_matrix.py --blocks A,B,C,D,E
# 5) tables, figures, claim check -> results/summary.md, results/tables, results/figures
python scripts/summarize.py
# 6) paper
cd paper && latexmk -pdf main.tex
```

Single runs: `python src/train.py --model {mnv3,effb0,dinov2s_lp,dinov2s_ft} --data {pv,pd_k5,pd_all} --seed {0,1,2} [--init results/runs/<run>/model.pt]`.

## Layout
- `configs/classmap.yaml`: the 28-class PlantDoc↔PlantVillage label map (released as part of the protocol).
- `src/`: training/eval (`train.py`), CLIP zero-shot (`clip_zeroshot.py`), edge profiling (`profile_edge.py`).
- `scripts/`: data download/prep, matrix runner, summarizer, arXiv BibTeX generator.
- `results/runs/<run>/`: `config.json`, `metrics.json`, `log.txt` and test logits for every run. Model weights (`model.pt`) are gitignored.
- `notes/`: landscape, ideas, plan (with pre-registered claims), self-review.
- `refs/verified.bib`: every reference, each with a `% url:` line showing how it was verified.

## Determinism
Seeds fix the data order, the k-shot subset (`kshot_indices`) and the initialization of new heads. `cudnn.benchmark` is on and training uses bf16 autocast, so reruns match statistically but are not bit-exact.
