<div align="center">

  <h1>Too Polite to Disagree: Understanding Sycophancy Propagation in Multi-Agent Systems</h1>

  <p>
    <a href="https://arxiv.org/abs/2604.02668"><img src="https://img.shields.io/badge/arXiv-2604.02668-b31b1b" alt="arXiv"></a>
    <a href="https://arxiv.org/pdf/2604.02668"><img src="https://img.shields.io/badge/SIGDIAL-2026-4a5fc1" alt="SIGDIAL 2026"></a>
  </p>

  <p><i>Telling agents which of their peers are sycophants makes multi-agent discussions<br>more disagreeable — and <b>more accurate</b>.</i></p>
</div>

---

## Abstract

Large language models (LLMs) often exhibit sycophancy: agreement with user stance even when it conflicts with the model's opinion. While prior work has mostly studied this in single-agent settings, it remains underexplored in collaborative multi-agent systems. We ask whether awareness of other agents' sycophancy levels influences discussion outcomes. To investigate this, we run controlled experiments with six open-source LLMs, providing agents with peer sycophancy rankings that estimate each peer's tendency toward sycophancy. These rankings are based on scores calculated using various static (pre-discussion) and dynamic (online) strategies. We find that providing sycophancy priors reduces the influence of sycophancy-prone peers, mitigates error-cascades, and improves final discussion accuracy by an absolute 10.5%. Thus, this is a lightweight and efficient way to reduce model sycophancy during discussions and subsequently improve downstream accuracy.

<div align="center">
  <img src="fig1.png" width="620" alt="Multi-Agent Discussion Pipeline">
</div>

***Multi-Agent Discussion Pipeline.***

- ***(a)*** *Computing base sycophancy scores (BSS) from single-agent queries on five MMLU subjects (Section 3.1). We also compute scores that involve discussion (Section 2.3).*
- ***(b)*** *Running a 6-agent discussion for 5 rounds: Round 0 answers are independently obtained from the models; in rounds m ∈ {1, 2, 3, 4}, each agent sees its peers' latest answers and their sycophancy scores and is allowed to freely re-choose a stance. The discussion's outcome is the majority final-round stance across models.*

## Models

The pipeline uses 6 models by default:

| Name | Huggingface Model |
|---|---|
| `llama3b`  | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| `llama8b`  | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| `qwen3b`   | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| `qwen7b`   | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| `qwen14b`  | [`Qwen/Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| `qwen32b`  | [`Qwen/Qwen2.5-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |

## Repository Layout

```
generate_bss_dss_data.py      build the BSS / DSS datasets from MMLU
bss_calc.py                   compute Baseline Sycophancy Scores (BSS) per model
compute_knowledge_flags.py    pre-compute per-model knowledge flags for each question
multiagent-debate.py          run a multi-agent debate experiment (all score modes)
evaluate.py                   score debate logs, produce per-experiment evals + comparison
plot_seed123.py               plot cross-experiment results

prompt.py                     debate and scoring prompts
logprobs_model.py             logprob-based soft sycophancy measurement
response_models.py            pydantic schemas for structured model responses
utils.py                      shared helpers

run_all_experiments.sh        full pipeline: data → BSS → flags → 9 experiments → eval
run_all_experiments_seeded.sh same, with a custom random seed
run_novel_subject_expt.sh     generalization to 15 unseen MMLU subjects

scores/                       precomputed BSS scores and knowledge flags
```

## Quick Start

### 1. Set up an environment

Requires Python 3.10+ and PyTorch with CUDA.

```bash
git clone https://github.com/0awesomeapples-dev/multiagent-discussion-sycophancy && cd multiagent-discussion-sycophancy
pip install -r requirements.txt
```

Set `HF_API_KEY` in your environment if needed for gated models.

### 2. Run the full pipeline (recommended)

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

### 3. Run with a different random seed

```bash
bash run_all_experiments_seeded.sh 123
```

Results go to `logs_seed123/`. Runs all 8 experiments.

### 4. Novel-subject generalization experiment

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

## License

Code is released under the [MIT License](LICENSE).

## Citation

If you found our work or this codebase useful in your research, please consider citing:

```bibtex
@misc{kasprova2026politedisagree,
  title  = {Too Polite to Disagree: Understanding Sycophancy Propagation in Multi-Agent Systems},
  author = {Kasprova, Vira and Parulekar, Amruta and AlRabah, Abdulrahman and Agaram, Krishna and Garg, Ritwik and Jha, Sagar and Bozdag, Nimet Beyza and Hakkani-Tur, Dilek},
  year   = {2026},
  eprint = {2604.02668},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2604.02668}
}
```
