#!/usr/bin/env python3
"""
render_results.py — Standalone result renderer for CompulsionBench.

Reads raw CSV tables from a completed pipeline bundle and produces
publication-quality figures and formatted markdown tables WITHOUT
re-running any experiments or touching the simulator.

Usage:
    python render_results.py --bundle_dir outputs/paper_param_v2
    python render_results.py --bundle_dir outputs/paper_param_v2 --style paper --format pdf --dpi 300
    python render_results.py --bundle_dir outputs/paper_param_v2 --style presentation --format png --dpi 150
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

STYLE_PRESETS = {
    "paper": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    },
    "presentation": {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 150,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    },
    "default": {},
}

POLICY_COLORS = {
    "PPO": "#e74c3c",
    "Lagrangian PPO": "#3498db",
    "PPO + SessionCap(120)": "#2ecc71",
    "PPO+AutoplayOff": "#9b59b6",
    "Myopic": "#f39c12",
    "Random": "#95a5a6",
    "RoundRobinPolicy": "#1abc9c",
    "LeastRecentPolicy": "#e67e22",
    "NoveltyGreedyPolicy": "#34495e",
}

ABLATION_COLORS = {
    "baseline": "#3498db",
    "NoHabit": "#e74c3c",
    "NoPers": "#f39c12",
    "NoVar": "#2ecc71",
    "HomogeneousUsers": "#9b59b6",
}


def apply_style(style_name: str) -> None:
    """Apply a style preset to matplotlib."""
    preset = STYLE_PRESETS.get(style_name, {})
    for key, value in preset.items():
        try:
            matplotlib.rcParams[key] = value
        except (KeyError, ValueError):
            pass


def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Load a CSV file, returning None if it doesn't exist."""
    if path.exists():
        return pd.read_csv(path)
    return None


def load_json_safe(path: Path) -> Optional[Dict]:
    """Load a JSON file, returning None if it doesn't exist."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Figure renderers
# ---------------------------------------------------------------------------


def render_scorecard_heatmap(
    df: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Render a heatmap of scorecard metrics across policies."""
    df = df.copy()
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        print(f"  [SKIP] No metrics found for heatmap: {title}")
        return

    if "policy" in df.columns:
        df = df.set_index("policy")

    data = df[available_metrics].astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(available_metrics) * 1.2), max(3, len(data) * 0.6)))
    normed = (data - data.min()) / (data.max() - data.min() + 1e-12)
    im = ax.imshow(normed.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(available_metrics)))
    ax.set_xticklabels(available_metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)

    for i in range(len(data)):
        for j in range(len(available_metrics)):
            val = data.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if normed.iloc[i, j] > 0.6 else "black")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized (0=best, 1=worst)")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print(f"  [OK] {out_path.with_suffix(f'.{fmt}').name}")


def render_frontier_plot(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    title: str,
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Render a CumWatch vs Risk frontier scatter plot."""
    if df is None or x_metric not in df.columns or y_metric not in df.columns:
        print(f"  [SKIP] Missing columns for frontier: {x_metric}, {y_metric}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    method_col = "method" if "method" in df.columns else ("policy" if "policy" in df.columns else None)
    methods = df[method_col].unique() if method_col else df.index
    for method in methods:
        subset = df[df[method_col] == method] if method_col else df.loc[[method]]
        color = POLICY_COLORS.get(str(method), "#666666")
        ax.scatter(subset[x_metric], subset[y_metric], label=method,
                   color=color, s=80, edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print(f"  [OK] {out_path.with_suffix(f'.{fmt}').name}")


def render_mechanism_ablation_bars(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Render grouped bar chart for mechanism ablations."""
    if df is None:
        print("  [SKIP] No ablation data")
        return

    metrics = ["CumWatch", "CVaR_0.95(L)", "OverCapMinutes"]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        print("  [SKIP] No ablation metrics found")
        return

    policies = df["policy"].unique() if "policy" in df.columns else ["PPO"]
    ablations = df["ablation"].unique() if "ablation" in df.columns else []

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        x = np.arange(len(policies))
        width = 0.15
        for i, abl in enumerate(ablations):
            subset = df[df["ablation"] == abl] if "ablation" in df.columns else df
            vals = []
            for pol in policies:
                row = subset[subset["policy"] == pol] if "policy" in subset.columns else subset
                vals.append(float(row[metric].iloc[0]) if len(row) > 0 else 0.0)
            color = ABLATION_COLORS.get(abl, f"C{i}")
            ax.bar(x + i * width, vals, width, label=abl, color=color, edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Policy")
        ax.set_ylabel(metric)
        ax.set_xticks(x + width * (len(ablations) - 1) / 2)
        ax.set_xticklabels(policies, rotation=20, ha="right")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print(f"  [OK] {out_path.with_suffix(f'.{fmt}').name}")


def render_cap_sensitivity_plot(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Render cap sensitivity sweep as a multi-metric line plot."""
    if df is None or "T_cap" not in df.columns:
        print("  [SKIP] No cap sensitivity data")
        return

    metrics = ["CumWatch", "CVaR_0.95(L)", "OverCapMinutes", "SessionCapTriggerRate"]
    available = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        ax.plot(df["T_cap"], df[metric], marker="o", linewidth=2, color="#3498db")
        ax.set_xlabel("T_cap (minutes)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print(f"  [OK] {out_path.with_suffix(f'.{fmt}').name}")


def render_training_convergence(
    bundle_dir: Path,
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Render training curves from training_history CSVs."""
    history_files = sorted(bundle_dir.glob("training_history_ppo_seed*.csv"))
    if not history_files:
        print("  [SKIP] No PPO training history files found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics_to_plot = [
        ("reward_per_rollout", "Reward per Rollout"),
        ("policy_entropy", "Policy Entropy"),
        ("NightMinutes_per_rollout", "NightMinutes per Rollout"),
    ]

    for hist_file in history_files:
        df = pd.read_csv(hist_file)
        seed_label = hist_file.stem.split("seed")[-1]
        for ax, (col, ylabel) in zip(axes, metrics_to_plot):
            if col in df.columns:
                ax.plot(df["global_step"], df[col], alpha=0.7, label=f"seed {seed_label}")
                ax.set_xlabel("Global Step")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

    # Also plot LagPPO if available
    lag_files = sorted(bundle_dir.glob("training_history_lagppo_seed*.csv"))
    if lag_files:
        for lag_file in lag_files:
            df = pd.read_csv(lag_file)
            seed_label = lag_file.stem.split("seed")[-1].split("_")[0]
            axes[0].plot(df["global_step"], df["reward_per_rollout"],
                        alpha=0.5, linestyle="--", label=f"LagPPO s{seed_label}")
            if "lambda1" in df.columns:
                # Add lambda plot overlay
                pass

    fig.suptitle("Training Convergence")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print(f"  [OK] {out_path.with_suffix(f'.{fmt}').name}")


def render_backend_sensitivity(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Render backend sensitivity comparison (Param vs LLM)."""
    if df is None or "backend" not in df.columns:
        print("  [SKIP] No backend sensitivity data")
        return

    metrics = ["CumWatch", "CVaR_0.95(L)", "NightFraction", "OverCapMinutes"]
    available = [m for m in metrics if m in df.columns]

    param_df = df[df["backend"] == "Param"]
    llm_df = df[df["backend"] == "LLM"]

    if len(param_df) == 0 or len(llm_df) == 0:
        print("  [SKIP] Need both Param and LLM rows for backend sensitivity")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    policies = param_df["policy"].unique()
    x = np.arange(len(policies))
    width = 0.35

    for ax, metric in zip(axes, available):
        param_vals = [float(param_df[param_df["policy"] == p][metric].iloc[0]) for p in policies]
        llm_vals = [float(llm_df[llm_df["policy"] == p][metric].iloc[0]) for p in policies]

        ax.bar(x - width/2, param_vals, width, label="Param", color="#3498db", edgecolor="black", linewidth=0.5)
        ax.bar(x + width/2, llm_vals, width, label="LLM", color="#e74c3c", edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Policy")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=7)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print(f"  [OK] {out_path.with_suffix(f'.{fmt}').name}")


def render_policy_radar(
    df: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Render a radar/spider chart comparing policies across normalized metrics."""
    if df is None:
        print("  [SKIP] No data for radar chart")
        return

    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        print("  [SKIP] Need ≥ 3 metrics for radar chart")
        return

    if "policy" in df.columns:
        df = df.set_index("policy")

    data = df[available].astype(float)
    # Normalize each metric to [0, 1]
    normed = (data - data.min()) / (data.max() - data.min() + 1e-12)

    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for policy in normed.index:
        values = normed.loc[policy].tolist()
        values += values[:1]
        color = POLICY_COLORS.get(str(policy), None)
        ax.plot(angles, values, "o-", linewidth=2, label=policy, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=8)
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print(f"  [OK] {out_path.with_suffix(f'.{fmt}').name}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render all figures and tables from a CompulsionBench result bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bundle_dir", required=True, type=Path,
                        help="Path to the result bundle directory (e.g. outputs/paper_param_v2)")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory for rendered figures. Defaults to bundle_dir/rendered/")
    parser.add_argument("--style", choices=["paper", "presentation", "default"], default="paper",
                        help="Visual style preset")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png",
                        help="Output figure format")
    parser.add_argument("--dpi", type=int, default=None,
                        help="Override DPI (default: style-dependent)")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    table_dir = bundle_dir / "tables"
    fig_dir = args.out_dir or (bundle_dir / "rendered")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    if not bundle_dir.exists():
        print(f"ERROR: Bundle directory does not exist: {bundle_dir}")
        sys.exit(1)

    apply_style(args.style)
    if args.dpi:
        matplotlib.rcParams["figure.dpi"] = args.dpi
        matplotlib.rcParams["savefig.dpi"] = args.dpi

    print(f"\n{'='*60}")
    print(f"CompulsionBench Result Renderer")
    print(f"  Bundle: {bundle_dir}")
    print(f"  Output: {fig_dir}")
    print(f"  Style:  {args.style} | Format: {fmt}")
    print(f"{'='*60}\n")

    # --- Load all available tables ---
    scorecard_det = load_csv_safe(table_dir / "table_scorecard_deterministic.csv")
    scorecard_sto = load_csv_safe(table_dir / "table_scorecard_stochastic.csv")
    frontier = load_csv_safe(table_dir / "table_frontier.csv")
    frontier_rebuilt = load_csv_safe(table_dir / "table_frontier_rebuilt.csv")
    cap_sensitivity = load_csv_safe(table_dir / "table_cap_sensitivity.csv")
    cap_sensitivity_rebuilt = load_csv_safe(table_dir / "table_cap_sensitivity_rebuilt.csv")
    mechanism_ablations = load_csv_safe(table_dir / "table_mechanism_ablations.csv")
    backend_sensitivity = load_csv_safe(table_dir / "table_backend_sensitivity.csv")
    manifest = load_json_safe(bundle_dir / "manifest.json")

    print("--- Scorecard Figures ---")
    main_metrics = ["CumWatch", "CVaR_0.95(L)", "ReturnRate60", "NightFraction", "OverCapMinutes"]

    if scorecard_det is not None:
        render_scorecard_heatmap(scorecard_det, main_metrics,
                                "Deterministic Scorecard", fig_dir / "scorecard_deterministic_heatmap", fmt)
        render_policy_radar(scorecard_det, main_metrics,
                           "Policy Comparison (Deterministic)", fig_dir / "policy_radar_deterministic", fmt)

    if scorecard_sto is not None:
        render_scorecard_heatmap(scorecard_sto, main_metrics,
                                "Stochastic Scorecard", fig_dir / "scorecard_stochastic_heatmap", fmt)

    print("\n--- Training Convergence ---")
    render_training_convergence(bundle_dir, fig_dir / "training_convergence", fmt)

    print("\n--- Frontier Figures ---")
    frontier_df = frontier_rebuilt if frontier_rebuilt is not None else frontier
    if frontier_df is None and scorecard_det is not None:
        frontier_df = scorecard_det
        
    if frontier_df is not None:
        render_frontier_plot(frontier_df, "CumWatch", "OverCapMinutes",
                            "Engagement–Risk Frontier (OverCap)", fig_dir / "frontier_overcap", fmt)
        if "NightFraction" in frontier_df.columns:
            render_frontier_plot(frontier_df, "CumWatch", "NightFraction",
                                "Engagement–Risk Frontier (NightFraction)", fig_dir / "frontier_nightfraction", fmt)

    print("\n--- Cap Sensitivity ---")
    cap_df = cap_sensitivity_rebuilt if cap_sensitivity_rebuilt is not None else cap_sensitivity
    render_cap_sensitivity_plot(cap_df, "Session Cap Sensitivity Sweep", fig_dir / "cap_sensitivity", fmt)

    print("\n--- Mechanism Ablations ---")
    render_mechanism_ablation_bars(mechanism_ablations, "Mechanism Ablation Comparison",
                                  fig_dir / "mechanism_ablations", fmt)

    print("\n--- Backend Sensitivity ---")
    render_backend_sensitivity(backend_sensitivity, "Backend Sensitivity (AGParam vs AGLLM)",
                               fig_dir / "backend_sensitivity", fmt)

    print(f"\n{'='*60}")
    print(f"Rendering complete. All figures written to: {fig_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
