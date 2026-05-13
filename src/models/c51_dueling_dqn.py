"""
src/models/c51_dueling_dqn.py
==============================
C51 + Dueling 網路架構（Rainbow 的核心）。

C51（Categorical DQN / Distributional DQN）：
    不直接輸出 Q 值，而是輸出 Q 值的「機率分佈」。
    每個動作對應一個有 N_atoms 個類別的分佈，
    支撐點（support）為 [v_min, v_max] 均勻分割。

    輸出形狀：(batch_size, n_actions, n_atoms)

Dueling 結構：
    共享特徵層 → Value stream（V(s)）+ Advantage stream（A(s,a)）
    合併：Q_dist(s,a) = V_dist(s) + A_dist(s,a) - mean(A_dist)

NoisyNet：
    Value / Advantage stream 的線性層替換為 NoisyLinear，
    提供隱式的隨機探索（取代 ε-greedy）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.noisy_layers import NoisyLinear


class C51DuelingNetwork(nn.Module):
    """
    C51 + Dueling + NoisyNet 網路（Rainbow 核心架構）。

    Args:
        input_dim:  輸入狀態維度（本作業：64）
        hidden_1:   共享特徵層隱藏維度（預設 128）
        hidden_2:   Value / Advantage stream 隱藏維度（預設 64）
        n_actions:  動作數（本作業：4）
        n_atoms:    C51 原子數（預設 51）
        v_min:      Q 值分佈下界（預設 -10）
        v_max:      Q 值分佈上界（預設 10）
        noisy:      是否使用 NoisyLinear（預設 True）
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_1: int  = 128,
        hidden_2: int  = 64,
        n_actions: int = 4,
        n_atoms: int   = 51,
        v_min: float   = -10.0,
        v_max: float   = 10.0,
        noisy: bool    = True,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms
        self.v_min     = v_min
        self.v_max     = v_max
        self.noisy     = noisy

        # 支撐點（support），形狀 (n_atoms,)
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, n_atoms),
        )

        # ── 共享特徵層（標準 Linear，穩定）────────
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
        )

        # ── Value stream ──────────────────────────
        Linear = NoisyLinear if noisy else nn.Linear
        self.value_hidden = nn.Sequential(
            Linear(hidden_1, hidden_2),
            nn.ReLU(),
        )
        self.value_out = Linear(hidden_2, n_atoms)   # V(s) 的分佈：(n_atoms,)

        # ── Advantage stream ──────────────────────
        self.adv_hidden = nn.Sequential(
            Linear(hidden_1, hidden_2),
            nn.ReLU(),
        )
        self.adv_out = Linear(hidden_2, n_actions * n_atoms)  # A(s,a) 分佈

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass。

        Returns:
            Q-value distribution（log softmax 前的 logit）
            shape: (batch_size, n_actions, n_atoms)
        """
        batch = x.shape[0]
        feat  = self.feature(x)

        v = self.value_out(self.value_hidden(feat))          # (B, n_atoms)
        a = self.adv_out(self.adv_hidden(feat))              # (B, n_actions * n_atoms)

        v = v.view(batch, 1, self.n_atoms)
        a = a.view(batch, self.n_actions, self.n_atoms)

        # Dueling aggregation（在 atom 維度）
        q_dist_logits = v + a - a.mean(dim=1, keepdim=True)  # (B, n_actions, n_atoms)
        return q_dist_logits

    def get_q_dist(self, x: torch.Tensor) -> torch.Tensor:
        """回傳 softmax 後的 Q 分佈。shape: (B, n_actions, n_atoms)"""
        return F.softmax(self.forward(x), dim=-1)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """從分佈計算期望 Q 值。shape: (B, n_actions)"""
        dist = self.get_q_dist(x)                             # (B, n_actions, n_atoms)
        # E[Q] = Σ_z p(z) * z
        return (dist * self.support.view(1, 1, -1)).sum(dim=-1)  # (B, n_actions)

    def reset_noise(self) -> None:
        """重置 NoisyLinear 的 noise（每次 forward 前呼叫）。"""
        if self.noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
