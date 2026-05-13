"""
src/models/dqn.py
=================
Q-Network 模型定義。

對應教授 starter code 程式 3.2 / 3.7：
    L1 = 64, L2 = 150, L3 = 100, L4 = 4
    model = nn.Sequential(Linear→ReLU→Linear→ReLU→Linear)

包含：
    - QNetwork:    基礎 MLP Q-Network
    - DuelingNet:  Dueling Architecture（HW3-2 用）
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 基礎 Q-Network（HW3-1 / HW3-2 NaiveDQN / DoubleDQN）
# ──────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    基礎 MLP Q-Network。

    對應 starter code 架構：
        64 → ReLU → 150 → ReLU → 100 → 4

    Args:
        input_dim:   輸入維度（預設 64，對應 4x4x4 GridWorld flatten）
        hidden_1:    第一隱藏層寬度（預設 150）
        hidden_2:    第二隱藏層寬度（預設 100）
        output_dim:  輸出維度（預設 4，對應 4 個動作）
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_1: int = 150,
        hidden_2: int = 100,
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., input_dim)

        Returns:
            Q values: shape (..., output_dim)
        """
        return self.net(x)


def build_target_network(online_net: QNetwork) -> QNetwork:
    """
    建立 Target Network（S2：Target Stabilization）。

    對應 starter code 程式 3.7：
        model2 = copy.deepcopy(model)
        model2.load_state_dict(model.state_dict())

    Args:
        online_net: Online Q-Network

    Returns:
        target_net: 完整複製的 Target Network（參數凍結）
    """
    target_net = copy.deepcopy(online_net)
    target_net.load_state_dict(online_net.state_dict())
    # Target Network 不需要梯度
    for param in target_net.parameters():
        param.requires_grad = False
    return target_net


def sync_target_network(online_net: QNetwork, target_net: QNetwork) -> None:
    """
    將 Online Network 的參數同步到 Target Network。

    對應 starter code：
        model2.load_state_dict(model.state_dict())

    Args:
        online_net: 來源網路（Online）
        target_net: 目標網路（Target，in-place 更新）
    """
    target_net.load_state_dict(online_net.state_dict())


# ──────────────────────────────────────────────
# Dueling Network（HW3-2 用，此處預先定義）
# ──────────────────────────────────────────────

class DuelingNet(nn.Module):
    """
    Dueling Network Architecture（S4：Value–Advantage Decomposition）。

    Q(s,a) = V(s) + [A(s,a) - mean_a'(A(s,a'))]

    Args:
        input_dim:  輸入維度
        hidden_dim: 共享層寬度
        n_actions:  動作數
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 150,
        n_actions: int = 4,
    ) -> None:
        super().__init__()
        # 共享特徵層
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream：估計 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        # Advantage stream：估計 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        V = self.value_stream(feat)            # (batch, 1)
        A = self.advantage_stream(feat)        # (batch, n_actions)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
