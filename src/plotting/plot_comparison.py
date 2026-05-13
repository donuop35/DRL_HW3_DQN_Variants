"""
src/plotting/plot_comparison.py
================================
多實驗比較圖（HW3-2 三方比較、HW3-3 E1-E4 消融實驗）。

所有比較圖從多個 CSV 載入真實數據後生成，
不得從假資料產生。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    load_experiment_log,
    add_moving_averages,
    compute_metrics,
)
from src.plotting.plot_curves import FIGURE_FORMAT, FIGURE_DPI, FIGURE_SIZE, _apply_style, _save_figure

# ──────────────────────────────────────────────
# 比較圖調色盤（固定顏色對應固定演算法）
# ──────────────────────────────────────────────

ALGORITHM_COLORS: Dict[str, str] = {
    # HW3-2
    "NaiveDQN":   "#4488ff",   # 藍色
    "DoubleDQN":  "#ff6644",   # 橙色
    "DuelingDQN": "#44cc88",   # 綠色
    # HW3-3
    "E1_Baseline":    "#aaaaff",  # 淡紫
    "E2_Stabilized":  "#ffaa44",  # 暖橙
    "E3_PER":         "#44ffcc",  # 青色
    "E4_Rainbow":     "#ff44aa",  # 粉紅（Bonus）
}

DEFAULT_COLORS = ["#4488ff", "#ff6644", "#44cc88", "#ffaa44", "#aa44ff", "#ff44aa"]


def _get_color(algorithm: str, idx: int) -> str:
    return ALGORITHM_COLORS.get(algorithm, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


# ──────────────────────────────────────────────
# 多曲線比較（同圖）
# ──────────────────────────────────────────────

def plot_reward_comparison(
    csv_paths: List[str | Path],
    output_path: str | Path,
    window: int = 100,
    title: str = "Algorithm Comparison — Episode Reward",
    labels: Optional[List[str]] = None,
    smoke_test: bool = False,
) -> Path:
    """
    多實驗 Reward 曲線同圖比較（主要用於 HW3-2 三方比較）。

    Args:
        csv_paths:   各實驗 CSV 路徑列表（順序對應 labels）
        output_path: 輸出路徑
        window:      移動平均視窗
        title:       圖表標題
        labels:      各實驗自訂標籤（None 時從 algorithm 欄位讀取）
        smoke_test:  是否標記為 smoke test

    Returns:
        儲存的圖表路徑
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, csv_path in enumerate(csv_paths):
        df = load_experiment_log(csv_path)
        df = add_moving_averages(df, window=window)
        alg = labels[i] if labels else df["algorithm"].iloc[0] if "algorithm" in df.columns else f"Exp{i}"
        color = _get_color(alg, i)

        ax.plot(df["episode"], df["episode_reward"],
                color=color, alpha=0.12, linewidth=0.7)
        ax.plot(df["episode"], df["reward_ma"],
                color=color, linewidth=2.2, label=f"{alg} (MA{window})")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


def plot_win_rate_comparison(
    csv_paths: List[str | Path],
    output_path: str | Path,
    window: int = 100,
    title: str = "Algorithm Comparison — Win Rate",
    labels: Optional[List[str]] = None,
    smoke_test: bool = False,
) -> Path:
    """
    多實驗 Win Rate 移動平均同圖比較。
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, csv_path in enumerate(csv_paths):
        df = load_experiment_log(csv_path)
        df = add_moving_averages(df, window=window)
        alg = labels[i] if labels else df["algorithm"].iloc[0] if "algorithm" in df.columns else f"Exp{i}"
        color = _get_color(alg, i)

        ax.plot(df["episode"], df["win_ma"],
                color=color, linewidth=2.2, label=f"{alg} (MA{window})")

    ax.axhline(y=0.5, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6, label="50% threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Win Rate", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


def plot_loss_comparison(
    csv_paths: List[str | Path],
    output_path: str | Path,
    window: int = 100,
    title: str = "Algorithm Comparison — Loss",
    labels: Optional[List[str]] = None,
    smoke_test: bool = False,
) -> Path:
    """
    多實驗 Loss 移動平均同圖比較。
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, csv_path in enumerate(csv_paths):
        df = load_experiment_log(csv_path)
        df = add_moving_averages(df, window=window)
        df = df[df["loss_mean"] > 0]
        alg = labels[i] if labels else df["algorithm"].iloc[0] if "algorithm" in df.columns else f"Exp{i}"
        color = _get_color(alg, i)

        ax.plot(df["episode"], df["loss_ma"],
                color=color, linewidth=2.0, label=f"{alg} (MA{window})")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Loss (MSE)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


# ──────────────────────────────────────────────
# 最終性能 Bar Chart
# ──────────────────────────────────────────────

def plot_final_performance_bar(
    csv_paths: List[str | Path],
    output_path: str | Path,
    metric: str = "final_win_rate",
    window: int = 100,
    title: str = "Final Performance Comparison",
    labels: Optional[List[str]] = None,
    smoke_test: bool = False,
) -> Path:
    """
    最終性能 Bar Chart（用最後 window 個 episodes 計算）。

    Args:
        csv_paths:   各實驗 CSV 路徑
        output_path: 輸出路徑
        metric:      "final_win_rate" | "final_average_reward"
        window:      計算最終性能的 episodes 數
        title:       圖表標題
        labels:      各實驗標籤
        smoke_test:  是否標記為 smoke test

    Returns:
        儲存的圖表路徑
    """
    _apply_style()

    alg_names = []
    values = []
    colors = []

    for i, csv_path in enumerate(csv_paths):
        df = load_experiment_log(csv_path)
        m = compute_metrics(df, window=window)
        alg = labels[i] if labels else m.get("algorithm", f"Exp{i}")
        val = m.get(metric, 0.0)
        alg_names.append(alg)
        values.append(val)
        colors.append(_get_color(alg, i))

    fig, ax = plt.subplots(figsize=(max(8, len(alg_names) * 2), 6))
    bars = ax.bar(alg_names, values, color=colors, width=0.5, alpha=0.85, edgecolor="#ffffff22")

    # 在 bar 上方標示數值
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, color="#e0e0f0",
        )

    ylabel = "Win Rate" if "win" in metric else "Average Reward"
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, min(1.15, max(values) * 1.25) if values else 1.0)
    ax.grid(True, axis="y")
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


# ──────────────────────────────────────────────
# HW3-2 專用：三方比較全套圖
# ──────────────────────────────────────────────

def plot_hw3_2_comparison(
    naive_csv: str | Path,
    double_csv: str | Path,
    dueling_csv: str | Path,
    output_dir: str | Path,
    window: int = 100,
    smoke_test: bool = False,
) -> dict:
    """
    HW3-2 三方比較完整圖表套件。

    Args:
        naive_csv, double_csv, dueling_csv: 各演算法 CSV 路徑
        output_dir:  輸出目錄
        window:      移動平均視窗
        smoke_test:  是否標記為 smoke test

    Returns:
        dict 包含各圖表路徑
    """
    csv_paths = [naive_csv, double_csv, dueling_csv]
    labels = ["NaiveDQN", "DoubleDQN", "DuelingDQN"]
    output_dir = Path(output_dir)
    paths = {}

    paths["reward_comparison"] = plot_reward_comparison(
        csv_paths, output_dir / "hw3_2_reward_comparison.png",
        window=window, title="HW3-2 Player Mode — Reward Comparison",
        labels=labels, smoke_test=smoke_test)

    paths["win_rate_comparison"] = plot_win_rate_comparison(
        csv_paths, output_dir / "hw3_2_win_rate_comparison.png",
        window=window, title="HW3-2 Player Mode — Win Rate Comparison",
        labels=labels, smoke_test=smoke_test)

    paths["loss_comparison"] = plot_loss_comparison(
        csv_paths, output_dir / "hw3_2_loss_comparison.png",
        window=window, title="HW3-2 Player Mode — Loss Comparison",
        labels=labels, smoke_test=smoke_test)

    paths["final_bar"] = plot_final_performance_bar(
        csv_paths, output_dir / "hw3_2_final_performance.png",
        metric="final_win_rate", window=window,
        title="HW3-2 Final Win Rate (last 100 episodes)",
        labels=labels, smoke_test=smoke_test)

    return paths


# ──────────────────────────────────────────────
# HW3-3 專用：E1/E2/E3(/E4) 消融比較圖
# ──────────────────────────────────────────────

def plot_hw3_3_ablation(
    e1_csv: str | Path,
    e2_csv: str | Path,
    e3_csv: str | Path,
    output_dir: str | Path,
    e4_csv: Optional[str | Path] = None,
    window: int = 100,
    smoke_test: bool = False,
) -> dict:
    """
    HW3-3 消融實驗比較圖（E1 vs E2 vs E3，選填 E4 Bonus）。

    Args:
        e1_csv, e2_csv, e3_csv: E1/E2/E3 CSV 路徑
        output_dir:  輸出目錄
        e4_csv:      E4 Rainbow Bonus CSV（可選）
        window:      移動平均視窗
        smoke_test:  是否標記為 smoke test

    Returns:
        dict 包含各圖表路徑
    """
    csv_paths = [e1_csv, e2_csv, e3_csv]
    labels = ["E1_Baseline", "E2_Stabilized", "E3_PER"]

    if e4_csv is not None:
        csv_paths.append(e4_csv)
        labels.append("E4_Rainbow")

    output_dir = Path(output_dir)
    paths = {}

    paths["reward_ablation"] = plot_reward_comparison(
        csv_paths, output_dir / "hw3_3_reward_ablation.png",
        window=window, title="HW3-3 Random Mode — Reward Ablation (E1→E3)",
        labels=labels, smoke_test=smoke_test)

    paths["win_rate_ablation"] = plot_win_rate_comparison(
        csv_paths, output_dir / "hw3_3_win_rate_ablation.png",
        window=window, title="HW3-3 Random Mode — Win Rate Ablation (E1→E3)",
        labels=labels, smoke_test=smoke_test)

    paths["final_bar"] = plot_final_performance_bar(
        csv_paths, output_dir / "hw3_3_final_performance.png",
        metric="final_win_rate", window=window,
        title="HW3-3 Final Win Rate (last 100 episodes)",
        labels=labels, smoke_test=smoke_test)

    return paths
