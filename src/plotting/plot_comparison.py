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

def plot_steps_comparison(
    csv_paths: list,
    output_path,
    window: int = 100,
    title: str = "Algorithm Comparison — Steps per Episode",
    labels=None,
    smoke_test: bool = False,
):
    """多實驗 Steps per Episode 移動平均比較圖。"""
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, csv_path in enumerate(csv_paths):
        df = load_experiment_log(csv_path)
        df = add_moving_averages(df, window=window)
        alg = labels[i] if labels else df["algorithm"].iloc[0] if "algorithm" in df.columns else f"Exp{i}"
        color = _get_color(alg, i)
        if "steps_ma" not in df.columns:
            df["steps_ma"] = df["episode_steps"].rolling(window, min_periods=1).mean()
        ax.plot(df["episode"], df["steps_ma"],
                color=color, linewidth=2.0, label=f"{alg} (MA{window})")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Steps", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()
    return _save_figure(fig, output_path, smoke_test=smoke_test)


# ── HW3-2 專用全套比較圖（dict 介面，供 run_hw3_2_player.py 呼叫）──

def plot_hw3_2_comparison(
    csv_map: Dict[str, str],
    output_dir,
    window: int = 100,
    title_prefix: str = "HW3-2 Player Mode",
    smoke_test: bool = False,
) -> dict:
    """
    HW3-2 三方比較全套圖（dict 介面）。

    Args:
        csv_map:     {label: csv_path} 字典，key 作為圖例標籤
        output_dir:  輸出目錄
        window:      移動平均視窗
        title_prefix: 圖表標題前綴
        smoke_test:  是否標記為 smoke test

    Returns:
        dict 包含各圖表路徑
    """
    output_dir = Path(output_dir)
    labels     = list(csv_map.keys())
    csv_paths  = [csv_map[k] for k in labels]
    paths      = {}

    paths["reward"] = plot_reward_comparison(
        csv_paths, output_dir / "hw3_2_player_reward_comparison.png",
        window=window, title=f"{title_prefix} — Reward",
        labels=labels, smoke_test=smoke_test)

    paths["win_rate"] = plot_win_rate_comparison(
        csv_paths, output_dir / "hw3_2_player_win_rate_comparison.png",
        window=window, title=f"{title_prefix} — Win Rate",
        labels=labels, smoke_test=smoke_test)

    paths["loss"] = plot_loss_comparison(
        csv_paths, output_dir / "hw3_2_player_loss_comparison.png",
        window=window, title=f"{title_prefix} — Loss",
        labels=labels, smoke_test=smoke_test)

    paths["steps"] = plot_steps_comparison(
        csv_paths, output_dir / "hw3_2_player_steps_comparison.png",
        window=window, title=f"{title_prefix} — Steps per Episode",
        labels=labels, smoke_test=smoke_test)

    paths["final_bar"] = plot_final_performance_bar(
        csv_paths, output_dir / "hw3_2_player_final_metrics_bar.png",
        metric="final_win_rate", window=window,
        title=f"{title_prefix} — Final Win Rate (last {window} ep)",
        labels=labels, smoke_test=smoke_test)

    print(f"[Comparison] Generated {len(paths)} HW3-2 figures → {output_dir}/hw3_2_player_*.png")
    return paths


# ──────────────────────────────────────────────
# HW3-3 專用：E1/E2/E3(/E4) 消融比較圖
# ──────────────────────────────────────────────

def plot_hw3_3_ablation(
    e1_csv,
    e2_csv,
    e3_csv,
    output_dir,
    e4_csv=None,
    window: int = 100,
    smoke_test: bool = False,
) -> dict:
    """
    HW3-3 消融實驗比較圖（E1 vs E2 vs E3，選填 E4 Bonus）。
    """
    csv_paths = [e1_csv, e2_csv, e3_csv]
    labels    = ["E1_Baseline", "E2_Stabilized", "E3_PER"]
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



# ── HW3-3 專用全套比較圖（dict 介面）────────────────────────

def plot_hw3_3_comparison(
    csv_map: Dict[str, str],
    output_dir,
    window: int = 100,
    title_prefix: str = "HW3-3 Random Mode",
    smoke_test: bool = False,
) -> dict:
    """
    HW3-3 E1/E2/E3 比較全套圖（dict 介面）。

    Args:
        csv_map:     {label: csv_path} 字典
        output_dir:  輸出目錄
        window:      移動平均視窗
        title_prefix: 圖表標題前綴

    Returns:
        dict 包含各圖表路徑
    """
    output_dir = Path(output_dir)
    labels     = list(csv_map.keys())
    csv_paths  = [csv_map[k] for k in labels]
    paths      = {}

    paths["reward"] = plot_reward_comparison(
        csv_paths, output_dir / "hw3_3_random_reward_comparison_e1_e2_e3.png",
        window=window, title=f"{title_prefix} — Reward (E1/E2/E3)",
        labels=labels, smoke_test=smoke_test)

    paths["win_rate"] = plot_win_rate_comparison(
        csv_paths, output_dir / "hw3_3_random_win_rate_comparison_e1_e2_e3.png",
        window=window, title=f"{title_prefix} — Win Rate (E1/E2/E3)",
        labels=labels, smoke_test=smoke_test)

    paths["loss"] = plot_loss_comparison(
        csv_paths, output_dir / "hw3_3_random_loss_comparison_e1_e2_e3.png",
        window=window, title=f"{title_prefix} — Loss (E1/E2/E3)",
        labels=labels, smoke_test=smoke_test)

    paths["steps"] = plot_steps_comparison(
        csv_paths, output_dir / "hw3_3_random_steps_comparison_e1_e2_e3.png",
        window=window, title=f"{title_prefix} — Steps (E1/E2/E3)",
        labels=labels, smoke_test=smoke_test)

    paths["final_bar"] = plot_final_performance_bar(
        csv_paths, output_dir / "hw3_3_random_final_metrics_e1_e2_e3.png",
        metric="final_win_rate", window=window,
        title=f"{title_prefix} — Final Win Rate (last {window} ep)",
        labels=labels, smoke_test=smoke_test)

    # 額外：epsilon decay 比較（從 CSV epsilon 欄）
    try:
        _apply_style()
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for i, (label, csv_path) in enumerate(csv_map.items()):
            df = load_experiment_log(csv_path)
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            ax.plot(df["episode"], df["epsilon"], color=color, linewidth=1.8, label=label)
        ax.set_title(f"{title_prefix} — Epsilon Decay Comparison", fontsize=14, pad=12)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Epsilon", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True)
        fig.tight_layout()
        paths["epsilon"] = _save_figure(fig, output_dir / "hw3_3_random_epsilon_decay_comparison.png", smoke_test=smoke_test)
    except Exception as e:
        print(f"  [WARN] Epsilon comparison skipped: {e}")

    # 額外：learning rate 曲線（從 CSV learning_rate 欄）
    try:
        _apply_style()
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for i, (label, csv_path) in enumerate(csv_map.items()):
            df = load_experiment_log(csv_path)
            if "learning_rate" in df.columns and df["learning_rate"].max() > 0:
                color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                ax.plot(df["episode"], df["learning_rate"], color=color, linewidth=1.8, label=label)
        ax.set_title(f"{title_prefix} — Learning Rate Curve", fontsize=14, pad=12)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Learning Rate", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True)
        fig.tight_layout()
        paths["lr"] = _save_figure(fig, output_dir / "hw3_3_random_learning_rate_curve.png", smoke_test=smoke_test)
    except Exception as e:
        print(f"  [WARN] LR curve skipped: {e}")

    print(f"[Comparison] Generated {len(paths)} HW3-3 figures → {output_dir}/hw3_3_random_*.png")
    return paths
