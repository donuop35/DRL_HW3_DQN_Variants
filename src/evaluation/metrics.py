"""
src/evaluation/metrics.py
=========================
統一 Evaluation Metrics 計算。

從 ExperimentLogger 輸出的 CSV 載入後，
用此模組計算標準評估指標。

使用範例：
    from src.evaluation.metrics import load_experiment_log, compute_metrics

    df = load_experiment_log("results/csv/hw3_1_static_naive_dqn_log.csv")
    metrics = compute_metrics(df, window=100)
    print(metrics)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 資料載入
# ──────────────────────────────────────────────

def load_experiment_log(csv_path: str | Path) -> pd.DataFrame:
    """
    載入 ExperimentLogger 輸出的 CSV。

    Args:
        csv_path: CSV 檔案路徑

    Returns:
        pandas DataFrame，欄位對應 EPISODE_LOG_FIELDS
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Log not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 型別修正
    if "win" in df.columns:
        df["win"] = df["win"].astype(int)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    return df


def load_multiple_logs(csv_paths: List[str | Path]) -> pd.DataFrame:
    """
    載入多個 experiment logs，合併為單一 DataFrame（用於比較圖）。

    Args:
        csv_paths: CSV 路徑列表

    Returns:
        合併的 DataFrame，以 experiment_id 區分
    """
    dfs = []
    for p in csv_paths:
        df = load_experiment_log(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ──────────────────────────────────────────────
# 指標計算
# ──────────────────────────────────────────────

def moving_average(series: pd.Series, window: int = 100) -> pd.Series:
    """
    計算移動平均。

    Args:
        series: 時間序列
        window: 視窗大小（預設 100 episodes）

    Returns:
        移動平均 Series
    """
    return series.rolling(window=window, min_periods=1).mean()


def compute_metrics(df: pd.DataFrame, window: int = 100) -> Dict[str, float]:
    """
    從 episode log DataFrame 計算標準評估指標。

    Args:
        df:     ExperimentLogger 輸出的 DataFrame
        window: 移動平均視窗大小

    Returns:
        dict 包含以下指標：
        - average_reward:          全體 episode 平均 reward
        - final_average_reward:    最後 window 個 episodes 的平均 reward
        - win_rate:                全體勝率
        - final_win_rate:          最後 window 個 episodes 的勝率
        - average_steps:           全體平均步數
        - final_average_steps:     最後 window 個 episodes 的平均步數
        - average_loss:            全體平均 loss
        - total_episodes:          總 episode 數
        - convergence_episode:     首次 win_rate（window 移動平均）超過 0.5 的 episode
    """
    if df.empty:
        return {}

    total = len(df)
    final_slice = df.tail(window)

    # 基本指標
    avg_reward = float(df["episode_reward"].mean())
    final_avg_reward = float(final_slice["episode_reward"].mean())
    win_rate = float(df["win"].mean())
    final_win_rate = float(final_slice["win"].mean())
    avg_steps = float(df["episode_steps"].mean())
    final_avg_steps = float(final_slice["episode_steps"].mean())
    avg_loss = float(df["loss_mean"].replace(0, np.nan).dropna().mean()) if "loss_mean" in df else 0.0

    # 收斂 episode（移動平均 win rate > 0.5 的首次出現）
    ma_win = moving_average(df["win"].astype(float), window=window)
    convergence_episodes = df.index[ma_win >= 0.5].tolist()
    convergence_ep = int(convergence_episodes[0]) if convergence_episodes else -1

    return {
        "experiment_id": df["experiment_id"].iloc[0] if "experiment_id" in df else "unknown",
        "algorithm": df["algorithm"].iloc[0] if "algorithm" in df else "unknown",
        "mode": df["mode"].iloc[0] if "mode" in df else "unknown",
        "total_episodes": total,
        "average_reward": round(avg_reward, 4),
        "final_average_reward": round(final_avg_reward, 4),
        "win_rate": round(win_rate, 4),
        "final_win_rate": round(final_win_rate, 4),
        "average_steps": round(avg_steps, 2),
        "final_average_steps": round(final_avg_steps, 2),
        "average_loss": round(avg_loss, 6),
        "convergence_episode": convergence_ep,
    }


def compute_all_metrics(csv_dir: str | Path, window: int = 100) -> pd.DataFrame:
    """
    批次計算 csv_dir 下所有 *_log.csv 的指標，匯整為一個 DataFrame。

    Args:
        csv_dir: results/csv/ 目錄路徑
        window:  移動平均視窗

    Returns:
        每行一個實驗的指標 DataFrame
    """
    csv_dir = Path(csv_dir)
    rows = []
    for csv_file in sorted(csv_dir.glob("*_log.csv")):
        try:
            df = load_experiment_log(csv_file)
            m = compute_metrics(df, window=window)
            rows.append(m)
        except Exception as e:
            print(f"[Metrics] Warning: skipped {csv_file.name} — {e}")
    return pd.DataFrame(rows)


def add_moving_averages(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    在 DataFrame 中新增移動平均欄位，供 plotting 使用。

    Args:
        df:     episode log DataFrame
        window: 移動平均視窗

    Returns:
        新增以下欄位的 DataFrame：
        - reward_ma:  episode_reward 移動平均
        - win_ma:     win 移動平均
        - loss_ma:    loss_mean 移動平均
        - steps_ma:   episode_steps 移動平均
    """
    df = df.copy()
    df["reward_ma"] = moving_average(df["episode_reward"], window)
    df["win_ma"] = moving_average(df["win"].astype(float), window)
    df["loss_ma"] = moving_average(df["loss_mean"], window)
    df["steps_ma"] = moving_average(df["episode_steps"].astype(float), window)
    return df
