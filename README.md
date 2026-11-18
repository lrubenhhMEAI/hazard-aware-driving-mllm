# Hazard-Aware Captioning & Causal Explanation for Autonomous Driving

**A multi-agent multimodal LLM pipeline grounded in a road-safety knowledge graph**

MSc Data Science & Artificial Intelligence dissertation · University of Hull, UK · 2026

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-HuggingFace-orange) ![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--7B-purple)

## Overview
Can an AI system look at a dashcam frame, identify the hazard, explain *why* it is dangerous and recommend what the driver should do, with every recommendation traceable to a real traffic rule?

This project builds a **four-agent pipeline** on top of **Qwen2.5-VL-7B-Instruct** (no fine-tuning) and grounds its decisions in a **Safety Knowledge Graph (SKG)** built from the UN Vienna Convention on Road Traffic (1968).

## Architecture
```
Dashcam frame(s)
     │
     ▼
1. Perception Agent ──────── objects, distance, traffic-light state, influence on ego path (JSON)
     │
     ▼
2. Hazard Reasoning Agent ── picks the main hazard → retrieves its subgraph from the SKG
     │
     ▼
3. Causal Explanation Agent ─ causal chain + traffic rule + one recommended action
     │
     ▼
4. Caption + Validator ────── DRAMA-style caption, consistency check (pass / fail)
```

**Safety Knowledge Graph:** 15 hazard nodes across 4 classes (vehicle, infrastructure, pedestrian, cyclist). Each node links to its condition, the Convention article that governs it, and the prescribed action.

## Dataset
- **DRAMA** (Malla et al., 2023): 2-second dashcam clips recorded in Tokyo
- Balanced evaluation subset of **520 scenarios** (130 per hazard class)
- 6-label action taxonomy covering **97.4%** of the annotated suggestions

## Results

| Condition | Exact match | Directional match |
|---|---|---|
| **A: Full pipeline + SKG** | **33.5%** | **76.5%** |
| B: Same reasoning stages, no knowledge | 27.3% | 56.9% |
| C: Direct inference baseline | 27.3% | 61.3% |

- The knowledge graph improves action recommendation by **+6.2 pts** (exact, p = 0.022) and **+15.2 pts** (directional, p < 0.0001, McNemar test)
- Splitting the reasoning into stages adds nothing by itself: **the gain comes from the external knowledge**
- Without the graph, the model falls back on non-committal actions ("be cautious" in >50% of cases)
- Perception ablation (1 frame vs 10 frames vs oracle bounding box): each input improves one aspect of perception, none improves the decision
- Error analysis: the bottleneck is turning the retrieved knowledge into a specific manoeuvre, not seeing the object
- Few-shot captioning raised BLEU-4 from 0.033 to 0.118 and ROUGE-L from 0.263 to 0.375

## Engineering decisions
- **Reproducibility:** greedy decoding (`do_sample=False`), pinned model revision hash, fixed sampling seed, deterministic scenario ordering
- **Fair comparison:** all conditions reuse the *same* perception output, so only the reasoning varies (paired design)
- **Robust long runs:** results are saved after every scenario; runs resume automatically and stop cleanly before a time budget runs out (HPC jobs of 4–16 h on an NVIDIA A40)
- **Comparable inputs:** visual tokens capped per image because the dataset mixes two camera resolutions
- **Structured outputs:** every agent answers in a fixed JSON schema, so each stage can be evaluated separately

## Evaluation
Exact & directional match · McNemar exact test · BLEU-4 · METEOR · ROUGE-L · BERTScore · MLLM-as-a-judge · class-reweighted estimates

## Repository structure
```
├── notebooks/
│   ├── 01_dataset.ipynb      # data quality checks, balanced subset, taxonomy coverage
│   ├── 02_pipeline.ipynb     # agents and all experimental runs (GPU)
│   └── 03_evaluation.ipynb   # metrics, statistical tests and figures (no GPU)
├── src/
│   ├── skg.py                # Safety Knowledge Graph + action taxonomy
│   ├── datos.py              # DRAMA loading and stratified sampling
│   ├── pipeline.py           # the four agents and orchestration
│   └── evaluacion.py         # offline metrics
├── data/                     # DRAMA goes here (not included)
├── results/                  # run outputs (JSON)
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
```
1. Request the DRAMA dataset from its authors and place it under `data/drama/` (see `data/README.md`), or set the `DRAMA_ANNOTATIONS`, `DRAMA_FRAMES` and `RESULTS_DIR` environment variables.
2. Run the notebooks from the `notebooks/` folder in order: `01` → `02` (requires a GPU with ~24 GB+ VRAM) → `01` consolidation cell → `03`.

The offline evaluation can also run from the command line:
```bash
cd src && python evaluacion.py ../results/pool520_1f.json
```

## Tech stack
Python · PyTorch · Hugging Face Transformers · Qwen2.5-VL · Knowledge graphs · Prompt engineering · statsmodels · NLTK · Matplotlib

## Author
**Luis Rubén Hernández Huerta** · [LinkedIn](https://www.linkedin.com/in/rubén-hernández-889b69353)

Supervisor: Dr. Temitayo Matthew Fagbola
