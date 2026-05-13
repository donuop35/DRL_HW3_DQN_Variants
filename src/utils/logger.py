"""
src/utils/logger.py
===================
統一 Experiment Logger。

每個 episode 必須呼叫 logger.log_episode(...)，
確保所有實驗輸出相同的 CSV schema（對應 SPEC-05）。

CSV Schema（每列一個 episode）：
    experiment_id, hw_part, mode, algorithm, seed,
    episode, episode_reward, episode_steps, loss_mean,
    epsilon, win, terminal_state, learning_rate,
    buffer_size, timestamp

使用範例：
    from src.utils.logger import ExperimentLogger
    from src.utils.config import load_config

    cfg = load_config("configs/hw3_1_static/default.yaml")
    logger = ExperimentLogger(cfg)

    for ep in range(cfg.training.episodes):
        # ... 訓練一個 episode ...
        logger.log_episode(
            episode=ep,
            episode_reward=total_reward,
            episode_steps=steps,
            loss_mean=mean_loss,
            epsilon=epsilon,
            win=reached_goal,
            terminal_state="goal",
            learning_rate=current_lr,
            buffer_size=len(replay),
        )

    logger.close()
    print(f"Log saved to: {logger.csv_path}")
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.config import ExperimentConfig, config_to_dict


# ──────────────────────────────────────────────
# 統一 CSV 欄位定義（SPEC-05）
# ──────────────────────────────────────────────

EPISODE_LOG_FIELDS = [
    "experiment_id",
    "hw_part",
    "mode",
    "algorithm",
    "seed",
    "episode",
    "episode_reward",
    "episode_steps",
    "loss_mean",
    "epsilon",
    "win",
    "terminal_state",    # "goal" | "pit" | "timeout" | "unknown"
    "learning_rate",
    "buffer_size",
    "timestamp",
]


class ExperimentLogger:
    """
    實驗 Logger。

    每次 __init__ 時建立（或追加）CSV 檔案。
    log_episode() 寫入一列 episode 紀錄。
    close() 確保檔案正確關閉。
    """

    def __init__(self, cfg: ExperimentConfig, append: bool = False) -> None:
        """
        Args:
            cfg:    ExperimentConfig 物件
            append: 是否以追加模式開啟（用於 resume 訓練）
        """
        self.cfg = cfg
        self.experiment_id = cfg.experiment_id
        self.hw_part = cfg.hw_part
        self.mode = cfg.mode
        self.algorithm = cfg.algorithm
        self.seed = cfg.seed

        # 確保目錄存在
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = log_dir / f"{cfg.experiment_id}_log.csv"

        # 儲存 config snapshot（YAML 格式，方便重現）
        config_dir = Path(cfg.checkpoint_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        self._save_config_snapshot(config_dir / "config_snapshot.csv")

        # 開啟 CSV
        mode_str = "a" if (append and self.csv_path.exists()) else "w"
        self._file = open(self.csv_path, mode_str, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_LOG_FIELDS)
        if mode_str == "w":
            self._writer.writeheader()

        self._episode_count = 0
        print(f"[Logger] Writing to: {self.csv_path}")

    def log_episode(
        self,
        episode: int,
        episode_reward: float,
        episode_steps: int,
        loss_mean: float,
        epsilon: float,
        win: bool,
        terminal_state: str = "unknown",
        learning_rate: float = 0.0,
        buffer_size: int = 0,
    ) -> None:
        """
        記錄一個 episode 的結果。

        Args:
            episode:        episode 編號（0-indexed）
            episode_reward: 本 episode 累積 reward
            episode_steps:  本 episode 使用步數
            loss_mean:      本 episode 平均 loss（若無訓練則為 0.0）
            epsilon:        當前 epsilon 值
            win:            是否到達 Goal（True/False）
            terminal_state: 終止原因（"goal" / "pit" / "timeout"）
            learning_rate:  當前學習率
            buffer_size:    當前 replay buffer 大小
        """
        row = {
            "experiment_id": self.experiment_id,
            "hw_part": self.hw_part,
            "mode": self.mode,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "episode": episode,
            "episode_reward": round(float(episode_reward), 4),
            "episode_steps": int(episode_steps),
            "loss_mean": round(float(loss_mean), 6) if loss_mean is not None else 0.0,
            "epsilon": round(float(epsilon), 6),
            "win": int(win),               # 0 / 1（方便 pandas 計算）
            "terminal_state": terminal_state,
            "learning_rate": round(float(learning_rate), 8),
            "buffer_size": int(buffer_size),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._writer.writerow(row)
        self._episode_count += 1

        # 每 100 episode flush 一次，避免資料遺失
        if self._episode_count % 100 == 0:
            self._file.flush()

    def close(self) -> None:
        """關閉 CSV 檔案。訓練結束後必須呼叫。"""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
        print(f"[Logger] Closed. Total episodes logged: {self._episode_count}")

    def _save_config_snapshot(self, path: Path) -> None:
        """將 config 儲存為 CSV 格式的 snapshot（方便追蹤）。"""
        snapshot = config_to_dict(self.cfg)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(snapshot.keys()))
            writer.writeheader()
            writer.writerow(snapshot)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
