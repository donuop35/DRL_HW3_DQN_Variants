"""
src/buffers/nstep_per_buffer.py
================================
N-step + PER 組合 Buffer（Rainbow 所需）。

N-step Return：
    不直接儲存 (s, a, r, s', done)，
    而是收集連續 n 步並計算折扣回報：
        R_n = r_1 + γ*r_2 + γ²*r_3 + ... + γ^(n-1)*r_n

    TD Target 變為：
        y = R_n + γ^n * max Q(s_{t+n}, ·)

    好處：減少 Bootstrap 的 bias（更多的實際 reward，更少的 Q 估計）。

整合 PER：
    N-step transition 存入 SumTree，以 TD error 為 priority。
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Tuple, List, Optional

import numpy as np
import torch

from src.buffers.prioritized_replay_buffer import PrioritizedReplayBuffer

NStepTransition = namedtuple(
    "NStepTransition", ["state", "action", "reward", "next_state", "done"]
)


class NStepPERBuffer:
    """
    N-step Prioritized Experience Replay Buffer。

    使用方式::

        buf = NStepPERBuffer(capacity=1000, n_step=3, gamma=0.9, alpha=0.6)
        buf.push(state, action, reward, next_state, done)
        batch = buf.sample(batch_size=32, beta=0.4)
        buf.update_priorities(batch["indices"], td_errors)
    """

    def __init__(
        self,
        capacity: int    = 1000,
        n_step: int      = 3,
        gamma: float     = 0.9,
        alpha: float     = 0.6,
        beta_start: float = 0.4,
        beta_end: float   = 1.0,
        per_epsilon: float = 1e-5,
    ) -> None:
        self.capacity    = capacity
        self.n_step      = n_step
        self.gamma       = gamma
        self.per         = PrioritizedReplayBuffer(
            capacity    = capacity,
            alpha       = alpha,
            beta_start  = beta_start,
            beta_end    = beta_end,
            per_epsilon = per_epsilon,
        )
        # N-step 暫存隊列
        self._nstep_buf: deque = deque(maxlen=n_step)

    def _compute_nstep(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """從 n-step buffer 計算折扣累積 reward。"""
        R       = 0.0
        done    = False
        for i, (s, a, r, ns, d) in enumerate(self._nstep_buf):
            R += (self.gamma ** i) * r
            if d:
                done     = True
                ns_final = ns
                break
            ns_final = ns

        # 回傳：最初的 s, a；n-step return R；最終 ns；是否終止
        s_init, a_init = self._nstep_buf[0][0], self._nstep_buf[0][1]
        return s_init, a_init, R, ns_final, done

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """累積 n 步後存入 PER buffer。"""
        self._nstep_buf.append((state, action, reward, next_state, done))
        if len(self._nstep_buf) < self.n_step:
            return   # 還不夠 n 步

        s, a, R, ns, d = self._compute_nstep()
        self.per.push(s, a, R, ns, d)

    def flush(self) -> None:
        """episode 結束時強制 flush 剩餘的 n-step buffer。"""
        while len(self._nstep_buf) > 0:
            s, a, R, ns, d = self._compute_nstep()
            self.per.push(s, a, R, ns, d)
            self._nstep_buf.popleft()

    def sample(self, batch_size: int, beta: float) -> dict:
        """代理至 PER.sample()。"""
        return self.per.sample(batch_size, beta)

    def update_priorities(self, indices: list, td_errors: np.ndarray) -> None:
        """代理至 PER.update_priorities()。"""
        self.per.update_priorities(indices, td_errors)

    def beta_by_step(self, current_step: int, total_steps: int) -> float:
        """代理至 PER.beta_by_step()。"""
        return self.per.beta_by_step(current_step, total_steps)

    def __len__(self) -> int:
        return len(self.per)
