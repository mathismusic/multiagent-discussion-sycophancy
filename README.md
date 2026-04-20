# Sycophancy-Aware Multi-Agent Debate

Code to go with the paper ``Too Polite to Disagree: Understanding Sycophancy Propagation in Multi-Agent Systems'' (SIGDIAL 2026).

## Requirements

- Python 3.10+
- PyTorch with CUDA
- Hugging Face `transformers`, `datasets`, `huggingface_hub`
- `pydantic`, `numpy`, `pandas`, `matplotlib`, `seaborn`

Set `HF_API_KEY` in your environment if needed for gated models.

Install dependencies:
```bash
pip install torch transformers datasets huggingface_hub pydantic numpy pandas matplotlib seaborn
```
https://github.com/0awesomeapples-dev/multiagent-discussion-sycophancy.git
## Models

The pipeline uses 6 models by default (short name -> HuggingFace ID):

| Short Name | Model |
|---|---|
| `llama3b`  | `meta-llama/Llama-3.2-3B-Instruct` |
| `llama8b`  | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen3b`   | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen7b`   | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen14b`  | `Qwen/Qwen2.5-14B-Instruct` |
| `qwen32b`  | `Qwen/Qwen2.5-32B-Instruct` |

## Quick Start

### Full pipeline (recommended)

Runs everything end-to-end: data generation, BSS computation, knowledge flags, all 9 debate experiments, and evaluation.

```bash
bash run_all_experiments.sh
```

**GPU requirement:** We use 8 A-series GPUs (experiments run in parallel pairs on GPUs 0-3 and 4-5-6-7). If using a different GPU setup, edit the `CUDA_VISIBLE_DEVICES` assignments in the script to match your setup.

**Experiments run:**

| Experiment | Score Mode | Description |
|---|---|---|
| `baseline` | None | No sycophancy information provided |
| `bss` | BSS | Static BSS-derived ranked labels |
| `dss` | DSS | Dynamic scores updated per round |
| `dbss` | DBSS | Post-debate BSS scores |
| `binary_bss` | Binary | Binary sycophantic/non-sycophantic labels |
| `accuracy_bss` | Accuracy | Accuracy-based scores (ablation) |
| `random_bss` | Random | Random scores (ablation) |
| `random_binary` | Random Binary | Random binary labels (ablation) |
| `warning_only` | Warning | Warning text only, no scores |

Results are saved to `logs/<experiment>/log.jsonl` with evaluations in `logs/<experiment>/eval/` and a cross-experiment comparison in `logs/comparison.csv`.

### With a different random seed

```bash
bash run_all_experiments_seeded.sh 123
```

Results go to `logs_seed123/`. Runs all 8 experiments.

### Novel-subject generalization experiment

Tests whether BSS scores computed on the original 5 MMLU subjects generalize to 15 new subjects.

```bash
bash run_novel_subject_expt.sh
```

Results go to `logs_new_subjects/`. Runs baseline, BSS, DSS, accuracy BSS, binary BSS, and DBSS.

## Running Individual Steps

### 1. Generate datasets

```bash
python generate_bss_dss_data.py \
  --bss_per_subject 50 --dss_per_subject 50 \
  --bss_out data_for_bss.csv --dss_out data_for_dss.csv
```

### 2. Compute BSS scores

```bash
python bss_calc.py --data_csv data_for_bss.csv
# Outputs: bss_scores_only.json (normalized), bss_scores_unnormalized.json
```

### 3. Pre-compute knowledge flags

```bash
python compute_knowledge_flags.py \
  --data_csv data_for_dss.csv \
  -m llama3b llama8b qwen3b qwen7b qwen14b qwen32b \
  -o scores/knowledge_flags.json
```

### 4. Run a single debate experiment

```bash
# Baseline (no scores)
python multiagent-debate.py -e baseline \
  -m llama3b llama8b qwen3b qwen7b qwen14b qwen32b \
  --data_csv data_for_dss.csv \
  --knowledge_flags_path scores/knowledge_flags.json

# With BSS scores
python multiagent-debate.py -e bss \
  -m llama3b llama8b qwen3b qwen7b qwen14b qwen32b \
  --data_csv data_for_dss.csv \
  --knowledge_flags_path scores/knowledge_flags.json \
  --use_bss_scores --bss_scores_path scores/bss_scores_final.json

# With DSS scores
python multiagent-debate.py -e dss \
  -m llama3b llama8b qwen3b qwen7b qwen14b qwen32b \
  --data_csv data_for_dss.csv \
  --knowledge_flags_path scores/knowledge_flags.json \
  --use_dss_scores --bss_scores_path scores/bss_scores_final.json \
  --alpha 0.2 --beta 0.0
```

### 5. Evaluate

```bash
# Single experiment
python evaluate.py -e baseline

# All experiments
python evaluate.py --all

# Custom log directory
python evaluate.py --all --log_dir logs_seed123
```

### 6. Plot results

```bash
# Edit LOG_DIR in the script to point to your results, then:
python plot_seed123.py
```

## Sycophancy Metrics

BSS is computed from four metrics (see `bss_calc.py`):

- **Sycophant With Knowledge (SWK):** Among samples where the model knows the correct answer, how often does it agree with the user's wrong stance?
- **Agreement Rate (AR):** How often does the model agree with the user's wrong stance (no knowledge gating)?
- **Confident Sycophancy (CS):** Average P("correct") when the user's stance is wrong (logprobs-based soft measure).
- **Picking Sides:** How often does the model side with the user regardless of correctness?

During debate, the three main metrics used are SWK, AR, and CS.
