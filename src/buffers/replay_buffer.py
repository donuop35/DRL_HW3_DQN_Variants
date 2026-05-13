"""
src/buffers/replay_buffer.py
============================
Experience Replay Buffer（經驗回放緩衝區）。

對應教授 starter code 程式 3.5：
    replay = deque(maxlen=mem_size)

Transition 格式（SPEC-05）：
    (state, action, reward, next_state, done)

S1 機制說明：
    - 打破時間相關性（temporal correlation）
    - 使訓練樣本更接近 i.i.d.
    - 提升樣本效率（每個 transition 可被多次採樣）
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────
# Transition 型別定義
# ──────────────────────────────────────────────

class Transition(NamedTuple):
    """
    一筆 Experience Replay 的資料單位。

    Fields:
        state:      當前狀態，shape (state_dim,)
        action:     執行的動作 index
        reward:     即時獎勵
        next_state: 下一個狀態，shape (state_dim,)
        done:       是否為終止狀態（bool）
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# ──────────────────────────────────────────────
# Replay Buffer 主體
# ──────────────────────────────────────────────

class ReplayBuffer:
    """
    基礎 Experience Replay Buffer。

    使用 deque（雙端佇列）存儲 transitions，
    當 buffer 滿時自動移除最舊的 transition。

    使用範例：
        buffer = ReplayBuffer(capacity=1000)
        buffer.push(state, action, reward, next_state, done)

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            # 使用 batch 訓練
    """

    def __init__(self, capacity: int = 1000) -> None:
        """
        Args:
            capacity: Buffer 最大容量（對應 starter code mem_size=1000）
        """
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        將一筆 transition 加入 buffer。

        Args:
            state:      當前狀態（numpy array）
            action:     動作 index
            reward:     即時獎勵
            next_state: 下一狀態（numpy array）
            done:       遊戲是否結束
        """
        self._buffer.append(Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        隨機採樣一個 mini-batch。

        Args:
            batch_size: 採樣數量（對應 starter code batch_size=200）

        Returns:
            (states, actions, rewards, next_states, dones) — 全部為 Tensor
        """
        batch = random.sample(self._buffer, batch_size)

        states      = torch.FloatTensor(np.stack([t.state      for t in batch]))
        actions     = torch.LongTensor( np.array([t.action     for t in batch]))
        rewards     = torch.FloatTensor(np.array([t.reward     for t in batch]))
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch]))
        dones       = torch.FloatTensor(np.array([t.done       for t in batch], dtype=np.float32))

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """回傳 buffer 是否有足夠樣本可以採樣。"""
        return len(self._buffer) >= batch_size
