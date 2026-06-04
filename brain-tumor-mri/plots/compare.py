# ============================================================
# plots/compare.py  —  BrainMRI FL Benchmark (repeat-aware)
# Reads:  results/rep{N}_{run}.json
# Writes: plots/learning_curves.png
#         plots/final_metrics.png
#         plots/confusion_matrices.png
#         plots/maskp_tradeoff.png
#         plots/per_client_auc.png
#         plots/per_client_final.png
#         plots/per_class_accuracy.png
# Usage:
#   python plots/compare.py
#   python plots/compare.py --show
# ============================================================

import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RUNS, RESULTS_DIR, PLOTS_DIR, N_ROUNDS, N_REPEATS, CLASSES

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "baseline": "#5a5a5a",
    "gs_0"    : "#2196F3",
    "gs_20"   : "#4CAF50",
    "gs_50"   : "#F44336",
}
LABELS = {
    "baseline": "Baseline (no GS)",
    "gs_0"    : "GS  maskP=0.0",
    "gs_20"   : "GS  maskP=0.2",
    "gs_50"   : "GS  maskP=0.5",
}
SITE_COLORS = ["#E91E63", "#9C27B0", "#FF9800"]

plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})


# ── Load ──────────────────────────────────────────────────────────────────────

def load_all_results() -> dict:
    all_results = {}
    for run_name, _, _ in RUNS:
        repeats = []
        for rep in range(1, N_REPEATS + 1):
            path = RESULTS_DIR / f"rep{rep}_{run_name}.json"
            if path.exists():
                with open(path) as f:
                    repeats.append(json.load(f))
        if repeats:
            all_results[run_name] = repeats
            print(f"  loaded {run_name}: {len(repeats)} repeat(s), "
                  f"{len(repeats[0].get('rounds',[]))} rounds each")
        else:
            print(f"  WARNING: no results for {run_name} -- skipping")
    return all_results


def extract_global(all_results, metric):
    out = {}
    for run_name, repeats in all_results.items():
        curves = []
        for rep in repeats:
            curves.append([r.get(metric, 0.0)
                           for r in rep.get("rounds", [])])
        min_len = min(len(c) for c in curves)
        curves  = [c[:min_len] for c in curves]
        arr     = np.array(curves)
        out[run_name] = (arr.mean(0), arr.std(0), curves)
    return out


def extract_client_metric(all_results, metric):
    out = {}
    for run_name, repeats in all_results.items():
        n_sites     = len(repeats[0]["rounds"][0].get("clients", []))
        site_curves = {i: [] for i in range(n_sites)}
        for rep in repeats:
            per_site = {i: [] for i in range(n_sites)}
            for rnd in rep.get("rounds", []):
                for idx, c in enumerate(rnd.get("clients", [])):
                    per_site[idx].append(
                        c.get("local_metrics", {}).get(metric, 0.0)
                    )
            for i in range(n_sites):
                site_curves[i].append(per_site[i])
        result = {}
        for i in range(n_sites):
            min_len = min(len(c) for c in site_curves[i])
            arr     = np.array([c[:min_len] for c in site_curves[i]])
            result[i] = (arr.mean(0), arr.std(0))
        out[run_name] = result
    return out


def extract_client_final(all_results, metric):
    out = {}
    for run_name, repeats in all_results.items():
        n_sites     = len(repeats[0]["rounds"][-1].get("clients", []))
        site_finals = {i: [] for i in range(n_sites)}
        for rep in repeats:
            for idx, c in enumerate(rep["rounds"][-1].get("clients", [])):
                site_finals[idx].append(
                    c.get("local_metrics", {}).get(metric, 0.0)
                )
        out[run_name] = {
            i: (np.mean(site_finals[i]), np.std(site_finals[i]))
            for i in range(n_sites)
        }
    return out


# ── Plot 1: Global learning curves ───────────────────────────────────────────

def plot_learning_curves(all_results, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        "Brain MRI 4-Class FL — Global Test Metrics\n"
        "(mean ± std across repeats)",
        fontsize=14, fontweight="bold", y=1.02
    )
    for metric, ax, ylabel in [
        ("auc_roc", axes[0], "AUC-ROC (macro OvR)"),
        ("f1",      axes[1], "F1 (macro)"),
    ]:
        data = extract_global(all_results, metric)
        for run_name in [r[0] for r in RUNS if r[0] in data]:
            mean, std, curves = data[run_name]
            rounds = list(range(1, len(mean) + 1))
            color  = COLORS[run_name]
            for c in curves:
                ax.plot(rounds, c, color=color, alpha=0.15, linewidth=1)
            ax.plot(rounds, mean, color=color, linewidth=2.5,
                    marker="o", markersize=4, label=LABELS[run_name])
            ax.fill_between(rounds, mean-std, mean+std,
                            color=color, alpha=0.12)
        ax.set_xlabel("FL Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Round")
        ax.set_ylim(0.6, 1.0)
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    out = PLOTS_DIR / "learning_curves.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  saved: {out}")
    if show: plt.show()
    plt.close()


# ── Plot 2: Final metrics bar chart ───────────────────────────────────────────

def plot_final_metrics(all_results, show=False):
    run_names = [r[0] for r in RUNS if r[0] in all_results]
    metrics   = ["accuracy", "auc_roc", "f1"]
    m_labels  = ["Accuracy", "AUC-ROC", "F1 (macro)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Final Round Global Metrics — Mean ± Std across Repeats",
        fontsize=14, fontweight="bold"
    )
    x     = np.arange(len(run_names))
    width = 0.6

    for ax, metric, m_label in zip(axes, metrics, m_labels):
        means, stds = [], []
        for rn in run_names:
            vals = [rep["rounds"][-1].get(metric, 0.0)
                    for rep in all_results[rn] if rep.get("rounds")]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        bars = ax.bar(x, means, width,
                      color=[COLORS[r] for r in run_names],
                      alpha=0.85, edgecolor="white",
                      yerr=stds, capsize=5,
                      error_kw={"elinewidth": 1.5, "ecolor": "black"})
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + s + 0.008,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(m_label)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[r] for r in run_names],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0.6, 1.05)
        ax.set_ylabel(m_label)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    out = PLOTS_DIR / "final_metrics.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  saved: {out}")
    if show: plt.show()
    plt.close()


# ── Plot 3: Confusion matrices ────────────────────────────────────────────────

def plot_confusion_matrices(all_results, show=False):
    run_names = [r[0] for r in RUNS if r[0] in all_results]
    n_runs    = len(run_names)
    ncols     = min(n_runs, 2)
    nrows     = (n_runs + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7*ncols, 6*nrows))
    if n_runs == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Confusion Matrices — Final Round (summed across repeats)",
        fontsize=14, fontweight="bold"
    )
    for idx, run_name in enumerate(run_names):
        row, col = divmod(idx, ncols)
        ax       = axes[row][col]
        cm_sum   = None
        auc_vals = []
        for rep in all_results[run_name]:
            last = rep["rounds"][-1]
            cm   = np.array(last.get("conf_matrix", [[0]*4]*4))
            cm_sum = cm if cm_sum is None else cm_sum + cm
            auc_vals.append(last.get("auc_roc", 0.0))
        sns.heatmap(cm_sum, annot=True, fmt="d",
                    cmap="Blues", ax=ax,
                    xticklabels=CLASSES,
                    yticklabels=CLASSES,
                    linewidths=0.5, linecolor="white",
                    cbar=False)
        ax.set_title(
            f"{LABELS[run_name]}\n"
            f"AUC={np.mean(auc_vals):.3f}±{np.std(auc_vals):.3f}",
            fontsize=11
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for idx in range(n_runs, nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "confusion_matrices.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  saved: {out}")
    if show: plt.show()
    plt.close()


# ── Plot 4: maskP tradeoff ────────────────────────────────────────────────────

def plot_maskp_tradeoff(all_results, show=False):
    gs_runs = [(r[0], r[2]) for r in RUNS
               if r[1] and r[0] in all_results]
    if not gs_runs:
        print("  no GS runs -- skipping maskP tradeoff")
        return

    maskp_vals = [mp for _, mp in gs_runs]
    run_names  = [rn for rn, _ in gs_runs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "maskP vs Utility — Mean ± Std across Repeats",
        fontsize=14, fontweight="bold"
    )
    for ax, metric, m_label in zip(
        axes, ["auc_roc", "f1"], ["AUC-ROC", "F1 (macro)"]
    ):
        if "baseline" in all_results:
            b_vals = [rep["rounds"][-1].get(metric, 0.0)
                      for rep in all_results["baseline"]
                      if rep.get("rounds")]
            bm, bs = np.mean(b_vals), np.std(b_vals)
            ax.axhline(bm, color=COLORS["baseline"],
                       linestyle="--", linewidth=2,
                       label=f"Baseline {bm:.3f}±{bs:.3f}", alpha=0.8)
            ax.axhspan(bm-bs, bm+bs,
                       color=COLORS["baseline"], alpha=0.07)
        means, stds = [], []
        for rn in run_names:
            vals = [rep["rounds"][-1].get(metric, 0.0)
                    for rep in all_results[rn] if rep.get("rounds")]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(maskp_vals, means, yerr=stds,
                    color="steelblue", linewidth=2.5,
                    marker="o", markersize=8,
                    capsize=5, capthick=1.5,
                    label="GS runs", zorder=3)
        for mp, m, s, rn in zip(maskp_vals, means, stds, run_names):
            ax.annotate(f"{m:.3f}±{s:.3f}", (mp, m),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=8, color=COLORS[rn])
        ax.set_xlabel("maskP")
        ax.set_ylabel(m_label)
        ax.set_title(f"maskP vs {m_label}")
        ax.set_xticks(maskp_vals)
        ax.set_ylim(0.6, 1.0)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    out = PLOTS_DIR / "maskp_tradeoff.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  saved: {out}")
    if show: plt.show()
    plt.close()


# ── Plot 5: Per-client local AUC curves ───────────────────────────────────────

def plot_per_client_auc(all_results, show=False):
    run_names = [r[0] for r in RUNS if r[0] in all_results]
    n_runs    = len(run_names)
    ncols     = min(n_runs, 2)
    nrows     = (n_runs + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(8*ncols, 5*nrows))
    if n_runs == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Per-Client Local AUC-ROC vs Round\n(mean ± std across repeats)",
        fontsize=14, fontweight="bold", y=1.02
    )
    client_data = extract_client_metric(all_results, "auc_roc")

    for idx, run_name in enumerate(run_names):
        row, col  = divmod(idx, ncols)
        ax        = axes[row][col]
        site_data = client_data[run_name]
        for site_idx, (mean, std) in site_data.items():
            rounds = list(range(1, len(mean)+1))
            color  = SITE_COLORS[site_idx % len(SITE_COLORS)]
            ax.plot(rounds, mean, color=color, linewidth=2,
                    marker="o", markersize=3,
                    label=f"site-{site_idx+1}")
            ax.fill_between(rounds, mean-std, mean+std,
                            color=color, alpha=0.12)
        ax.set_title(LABELS[run_name])
        ax.set_xlabel("FL Round")
        ax.set_ylabel("Local AUC-ROC")
        ax.set_ylim(0.6, 1.0)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

    for idx in range(n_runs, nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "per_client_auc.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  saved: {out}")
    if show: plt.show()
    plt.close()


# ── Plot 6: Per-client final metrics ─────────────────────────────────────────

def plot_per_client_final(all_results, show=False):
    run_names = [r[0] for r in RUNS if r[0] in all_results]
    metrics   = ["auc_roc", "f1", "accuracy"]
    m_labels  = ["AUC-ROC", "F1 (macro)", "Accuracy"]
    n_sites   = len(all_results[run_names[0]][0]["rounds"][-1]
                    .get("clients", []))

    fig, axes = plt.subplots(
        len(metrics), len(run_names),
        figsize=(5*len(run_names), 4*len(metrics))
    )
    if len(run_names) == 1:
        axes = axes.reshape(-1, 1)
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Per-Client Final Metrics — Mean ± Std across Repeats",
        fontsize=14, fontweight="bold", y=1.01
    )
    for m_idx, (metric, m_label) in enumerate(zip(metrics, m_labels)):
        client_data = extract_client_final(all_results, metric)
        for r_idx, run_name in enumerate(run_names):
            ax     = axes[m_idx][r_idx]
            site_d = client_data[run_name]
            x      = np.arange(n_sites)
            means  = [site_d[i][0] for i in range(n_sites)]
            stds   = [site_d[i][1] for i in range(n_sites)]
            bars   = ax.bar(x, means, 0.6,
                            color=SITE_COLORS[:n_sites],
                            alpha=0.85, edgecolor="white",
                            yerr=stds, capsize=4,
                            error_kw={"elinewidth": 1.5,
                                      "ecolor": "black"})
            for bar, m, s in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + s + 0.008,
                        f"{m:.3f}", ha="center", va="bottom",
                        fontsize=8)
            ax.set_title(f"{LABELS[run_name]}\n{m_label}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([f"site-{i+1}" for i in range(n_sites)],
                               fontsize=9)
            ax.set_ylim(0.5, 1.1)
            ax.set_ylabel(m_label if r_idx == 0 else "")
            ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    out = PLOTS_DIR / "per_client_final.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  saved: {out}")
    if show: plt.show()
    plt.close()


# ── Plot 7: Per-class accuracy ────────────────────────────────────────────────

def plot_per_class_accuracy(all_results, show=False):
    run_names = [r[0] for r in RUNS if r[0] in all_results]

    fig, axes = plt.subplots(1, len(run_names),
                             figsize=(5*len(run_names), 4))
    if len(run_names) == 1:
        axes = [axes]

    fig.suptitle(
        "Per-Class Accuracy — Final Round (mean across repeats)",
        fontsize=14, fontweight="bold"
    )
    for ax, run_name in zip(axes, run_names):
        per_class = []
        for rep in all_results[run_name]:
            cm  = np.array(rep["rounds"][-1].get("conf_matrix", [[0]*4]*4))
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            per_class.append(np.diag(cm) / row_sums.flatten())
        per_class = np.array(per_class).mean(axis=0)

        bars = ax.bar(np.arange(len(CLASSES)), per_class, 0.6,
                      color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, per_class):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(LABELS[run_name], fontsize=11)
        ax.set_xticks(np.arange(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy" if ax == axes[0] else "")
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    plt.tight_layout()
    out = PLOTS_DIR / "per_class_accuracy.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  saved: {out}")
    if show: plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print()
    print("  BrainMRI Benchmark — Generating Plots")
    print("  " + "─" * 50)
    print()

    print("  Loading results...")
    all_results = load_all_results()

    if not all_results:
        print("  No results found.")
        sys.exit(1)

    print()
    plot_learning_curves(all_results,    show=args.show)
    plot_final_metrics(all_results,      show=args.show)
    plot_confusion_matrices(all_results, show=args.show)
    plot_maskp_tradeoff(all_results,     show=args.show)
    plot_per_client_auc(all_results,     show=args.show)
    plot_per_client_final(all_results,   show=args.show)
    plot_per_class_accuracy(all_results, show=args.show)

    print()
    print(f"  All plots saved to {PLOTS_DIR}")
    print()