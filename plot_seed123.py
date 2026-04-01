"""
Plot accuracy and sycophancy metrics for all 8 experiments in logs_seed123/.

Produces 4 grouped bar charts saved to logs_seed123/:
  1. final_accuracy.png         – per-model + majority accuracy
  2. sycophancy_agreement_rate.png
  3. sycophancy_confident_sycophancy.png
  4. sycophancy_sycophant_with_knowledge.png

Usage:
  python plot_seed123.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "logs_seed123"

EXPERIMENTS = [
    "baseline",
    "bss",
    "dss",
    "dbss",
    "binary_bss",
    "random_bss",
    "random_binary",
    "accuracy_bss",
]

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "bss": "BSS",
    "dss": "DSS",
    "dbss": "DBSS",
    "binary_bss": "Binary BSS",
    "random_bss": "Random BSS",
    "random_binary": "Random Binary",
    "accuracy_bss": "Accuracy BSS",
}

MODELS = ["llama3b", "llama8b", "qwen3b", "qwen7b", "qwen14b", "qwen32b", "majority"]

SYCO_METRICS = ["agreement_rate", "confident_sycophancy", "sycophant_with_knowledge"]

SYCO_TITLES = {
    "agreement_rate": "Post-Discussion Agreement Rate (lower = better)",
    "confident_sycophancy": "Post-Discussion Confident Sycophancy (lower = better)",
    "sycophant_with_knowledge": "Post-Discussion Sycophant With Knowledge (lower = better)",
}

# Colors for experiments
COLORS = {
    "baseline": "#7f7f7f",
    "bss": "#1f77b4",
    "dss": "#ff7f0e",
    "dbss": "#2ca02c",
    "binary_bss": "#d62728",
    "random_bss": "#9467bd",
    "random_binary": "#8c564b",
    "accuracy_bss": "#e377c2",
}


def load_summaries():
    summaries = {}
    for exp in EXPERIMENTS:
        path = os.path.join(LOG_DIR, exp, "eval", "summary.json")
        if os.path.isfile(path):
            with open(path) as f:
                summaries[exp] = json.load(f)
        else:
            print(f"Warning: {path} not found, skipping {exp}")
    return summaries


def plot_grouped_bars(data, title, ylabel, filename, highlight_lower=False):
    """
    data: dict[experiment -> dict[model -> value]]
    """
    exps = [e for e in EXPERIMENTS if e in data]
    n_exp = len(exps)
    n_models = len(MODELS)

    x = np.arange(n_models)
    width = 0.8 / n_exp

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, exp in enumerate(exps):
        vals = [data[exp].get(m, 0) for m in MODELS]
        offset = (i - n_exp / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=DISPLAY_NAMES[exp],
                      color=COLORS[exp], edgecolor="white", linewidth=0.5)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(LOG_DIR, filename)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    summaries = load_summaries()

    # 1. Accuracy plot
    acc_data = {}
    for exp, s in summaries.items():
        acc_data[exp] = s["final_accuracy"]

    plot_grouped_bars(
        acc_data,
        "Final Accuracy Across 8 Experiments (seed=123)",
        "Accuracy",
        "final_accuracy.png",
    )

    # 2-4. Sycophancy plots
    for metric in SYCO_METRICS:
        syco_data = {}
        for exp, s in summaries.items():
            pds = s.get("post_discussion_sycophancy", {})
            if metric in pds:
                syco_data[exp] = pds[metric]

        plot_grouped_bars(
            syco_data,
            SYCO_TITLES[metric] + " (seed=123)",
            "Sycophancy Rate",
            f"sycophancy_{metric}.png",
            highlight_lower=True,
        )


if __name__ == "__main__":
    main()
