"""
src/plotting/plot_curves.py
============================
單一實驗的標準訓練曲線繪圖。

所有圖表：
- 從 results/csv/ 的真實 CSV 載入
- 輸出至 results/figures/
- 不得從假資料產生正式圖表

SPEC-06 規格：
- FIGURE_FORMAT = 'png'
- FIGURE_DPI = 150
- FIGURE_SIZE = (10, 7)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics import load_experiment_log, add_moving_averages

# ──────────────────────────────────────────────
# 全域圖表風格設定
# ──────────────────────────────────────────────

FIGURE_FORMAT = "png"
FIGURE_DPI = 150
FIGURE_SIZE = (10, 7)

STYLE = {
    "axes.facecolor": "#1a1a2e",
    "figure.facecolor": "#0f0f1a",
    "axes.edgecolor": "#444466",
    "axes.labelcolor": "#e0e0f0",
    "xtick.color": "#b0b0cc",
    "ytick.color": "#b0b0cc",
    "text.color": "#e0e0f0",
    "grid.color": "#333355",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "lines.linewidth": 1.5,
    "font.family": "DejaVu Sans",
}

COLOR_RAW = "#4488ff"
COLOR_MA = "#ff6644"
COLOR_WIN = "#44cc88"


def _apply_style():
    plt.rcParams.update(STYLE)


def _save_figure(fig: plt.Figure, output_path: str | Path, smoke_test: bool = False) -> Path:
    """儲存圖表並在 smoke test 時加上浮水印。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if smoke_test:
        fig.text(
            0.5, 0.5, "SMOKE TEST — NOT OFFICIAL RESULT",
            ha="center", va="center", fontsize=20, color="red",
            alpha=0.3, rotation=30, transform=fig.transFigure,
        )
    fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {output_path}")
    return output_path


# ──────────────────────────────────────────────
# 個別曲線圖
# ──────────────────────────────────────────────

def plot_reward_curve(
    df: pd.DataFrame,
    output_path: str | Path,
    window: int = 100,
    title: Optional[str] = None,
    smoke_test: bool = False,
) -> Path:
    """
    繪製 Episode Reward 曲線（raw + moving average）。

    Args:
        df:          episode log DataFrame（含 episode_reward 欄位）
        output_path: 輸出圖表路徑
        window:      移動平均視窗
        title:       圖表標題（None 時自動從 df 推導）
        smoke_test:  是否標記為 smoke test

    Returns:
        儲存的圖表路徑
    """
    _apply_style()
    df = add_moving_averages(df, window=window)

    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else "unknown"
    alg = df["algorithm"].iloc[0] if "algorithm" in df.columns else "unknown"
    t = title or f"{exp_id} — Episode Reward"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(df["episode"], df["episode_reward"],
            color=COLOR_RAW, alpha=0.25, linewidth=0.8, label="Raw Reward")
    ax.plot(df["episode"], df["reward_ma"],
            color=COLOR_MA, linewidth=2.0, label=f"MA({window})")
    ax.set_title(t, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


def plot_loss_curve(
    df: pd.DataFrame,
    output_path: str | Path,
    window: int = 100,
    title: Optional[str] = None,
    smoke_test: bool = False,
) -> Path:
    """
    繪製 Loss 曲線。
    """
    _apply_style()
    df = add_moving_averages(df, window=window)
    df_loss = df[df["loss_mean"] > 0]  # 過濾無訓練的 episode

    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else "unknown"
    t = title or f"{exp_id} — Training Loss"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(df_loss["episode"], df_loss["loss_mean"],
            color=COLOR_RAW, alpha=0.2, linewidth=0.8, label="Raw Loss")
    ax.plot(df_loss["episode"], df_loss["loss_ma"],
            color=COLOR_MA, linewidth=2.0, label=f"MA({window})")
    ax.set_title(t, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Loss (MSE)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


def plot_win_rate_curve(
    df: pd.DataFrame,
    output_path: str | Path,
    window: int = 100,
    title: Optional[str] = None,
    smoke_test: bool = False,
) -> Path:
    """
    繪製 Win Rate 移動平均曲線。
    """
    _apply_style()
    df = add_moving_averages(df, window=window)

    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else "unknown"
    t = title or f"{exp_id} — Win Rate (MA {window})"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(df["episode"], df["win_ma"],
            color=COLOR_WIN, linewidth=2.0, label=f"Win Rate MA({window})")
    ax.axhline(y=0.5, color="#ff8800", linestyle="--", linewidth=1.0, alpha=0.7, label="50% threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(t, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Win Rate", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


def plot_steps_curve(
    df: pd.DataFrame,
    output_path: str | Path,
    window: int = 100,
    title: Optional[str] = None,
    smoke_test: bool = False,
) -> Path:
    """
    繪製每 episode 步數曲線。
    """
    _apply_style()
    df = add_moving_averages(df, window=window)

    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else "unknown"
    t = title or f"{exp_id} — Steps per Episode"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(df["episode"], df["episode_steps"],
            color=COLOR_RAW, alpha=0.25, linewidth=0.8, label="Raw Steps")
    ax.plot(df["episode"], df["steps_ma"],
            color=COLOR_MA, linewidth=2.0, label=f"MA({window})")
    ax.set_title(t, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Steps", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


def plot_epsilon_curve(
    df: pd.DataFrame,
    output_path: str | Path,
    title: Optional[str] = None,
    smoke_test: bool = False,
) -> Path:
    """
    繪製 Epsilon 衰減曲線。
    """
    _apply_style()

    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else "unknown"
    t = title or f"{exp_id} — Epsilon Decay"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(df["episode"], df["epsilon"],
            color="#aa66ff", linewidth=2.0, label="Epsilon")
    ax.set_title(t, fontsize=14, pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Epsilon (ε)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(fig, output_path, smoke_test=smoke_test)


def plot_all_curves(
    csv_path: str | Path,
    output_dir: str | Path,
    window: int = 100,
    smoke_test: bool = False,
) -> dict:
    """
    從單一 CSV 產生所有標準曲線圖。

    Args:
        csv_path:   CSV 路徑
        output_dir: 圖表輸出目錄
        window:     移動平均視窗
        smoke_test: 是否標記為 smoke test

    Returns:
        dict 包含各圖表的輸出路徑
    """
    df = load_experiment_log(csv_path)
    output_dir = Path(output_dir)
    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else Path(csv_path).stem

    paths = {}
    paths["reward"] = plot_reward_curve(
        df, output_dir / f"{exp_id}_reward.png", window=window, smoke_test=smoke_test)
    paths["loss"] = plot_loss_curve(
        df, output_dir / f"{exp_id}_loss.png", window=window, smoke_test=smoke_test)
    paths["win_rate"] = plot_win_rate_curve(
        df, output_dir / f"{exp_id}_win_rate.png", window=window, smoke_test=smoke_test)
    paths["steps"] = plot_steps_curve(
        df, output_dir / f"{exp_id}_steps.png", window=window, smoke_test=smoke_test)
    paths["epsilon"] = plot_epsilon_curve(
        df, output_dir / f"{exp_id}_epsilon.png", smoke_test=smoke_test)
    return paths
