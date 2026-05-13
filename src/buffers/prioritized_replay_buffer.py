"""
src/buffers/prioritized_replay_buffer.py
=========================================
Prioritized Experience Replay Buffer（PER）

實作 Schaul et al. 2016 的 PER，使用 SumTree 進行 O(log n) 優先抽樣。

參數：
    capacity     : buffer 最大容量
    alpha        : 優先度指數（0=uniform, 1=full priority）
    beta_start   : IS weight 起始值
    beta_end     : IS weight 結束值（訓練尾期 annealing 到 1）
    per_epsilon  : 防止 priority=0 的小常數
"""

from __future__ import annotations

import random
from collections import namedtuple
from typing import Tuple, List

import numpy as np
import torch

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class SumTree:
    """
    SumTree：用於 O(log n) priority sampling。

    - 葉節點（leaf）儲存各 transition 的 priority
    - 父節點儲存子節點 priority 之和
    - 採樣時：從 [0, total] 均勻採樣，沿樹找到對應葉節點
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: List = [None] * capacity
        self.write = 0          # 下一個寫入位置（circular）
        self.n_entries = 0      # 實際儲存數

    @property
    def total(self) -> float:
        return self.tree[0]

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def update(self, idx: int, priority: float) -> None:
        """更新指定位置的 priority。"""
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority: float, data) -> None:
        """加入新資料。"""
        leaf_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def get(self, s: float) -> Tuple[int, float, object]:
        """取得 s 對應的（tree_idx, priority, data）。"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx % self.capacity]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer。

    使用範例::

        per = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)
        per.push(state, action, reward, next_state, done)
        batch = per.sample(batch_size=200, beta=0.4)
        # 訓練後更新 priorities
        per.update_priorities(batch["indices"], td_errors)
    """

    def __init__(
        self,
        capacity: int = 1000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        per_epsilon: float = 1e-5,
    ) -> None:
        self.tree         = SumTree(capacity)
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.per_epsilon  = per_epsilon
        self._max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """加入 transition，新 transition 使用最大 priority（確保至少被採樣一次）。"""
        t = Transition(state, action, reward, next_state, done)
        self.tree.add(self._max_priority ** self.alpha, t)

    def sample(self, batch_size: int, beta: float) -> dict:
        """
        按 priority 比例採樣 batch。

        Returns dict with keys:
            states, actions, rewards, next_states, dones  → Tensors
            weights   → IS importance sampling weights Tensor
            indices   → SumTree leaf indices（用於更新 priority）
        """
        segment = self.tree.total / batch_size
        indices, priorities, transitions = [], [], []

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s  = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # fallback: 用最大 priority 填補
                idx, priority, data = self.tree.get(self.tree.total * random.random())
            indices.append(idx)
            priorities.append(priority)
            transitions.append(data)

        # 計算 Importance Sampling weights
        N = self.tree.n_entries
        probs = np.array(priorities, dtype=np.float64) / self.tree.total
        probs = np.clip(probs, 1e-10, None)
        weights = (N * probs) ** (-beta)
        weights /= weights.max()  # normalize

        states      = torch.FloatTensor(np.array([t.state      for t in transitions]))
        actions     = torch.LongTensor( np.array([t.action     for t in transitions]))
        rewards     = torch.FloatTensor(np.array([t.reward     for t in transitions]))
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions]))
        dones       = torch.FloatTensor(np.array([float(t.done) for t in transitions]))
        w_tensor    = torch.FloatTensor(weights).unsqueeze(1)

        return {
            "states":      states,
            "actions":     actions,
            "rewards":     rewards,
            "next_states": next_states,
            "dones":       dones,
            "weights":     w_tensor,
            "indices":     indices,
        }

    def update_priorities(self, indices: list, td_errors: np.ndarray) -> None:
        """用 TD error 更新各 transition 的 priority。"""
        for idx, td_err in zip(indices, td_errors):
            p = (abs(td_err) + self.per_epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, p)
            self.tree.update(idx, p)

    def beta_by_step(self, current_step: int, total_steps: int) -> float:
        """線性 anneal beta：beta_start → beta_end。"""
        fraction = min(1.0, current_step / max(1, total_steps))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self.tree.n_entries
