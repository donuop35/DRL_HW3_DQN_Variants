"""
src/models/noisy_layers.py
===========================
NoisyNet 探索層（Fortunato et al., 2017）。

Factorised Noisy Linear 層：將固定隨機性替換為可學習的噪聲參數，
讓網路自行決定「需要多少探索」，取代 epsilon-greedy 策略。

數學形式：
    y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)

Factorised 版本：用 p 個獨立噪聲生成 p×q 的噪聲矩陣，
減少隨機數需求從 O(pq) 降為 O(p+q)。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Factorised Noisy Linear 層（Fortunato et al., 2017）。

    Args:
        in_features:  輸入維度
        out_features: 輸出維度
        sigma_init:   噪聲標準差初始值（default 0.5）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        # 可學習參數：均值 μ
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))

        # 可學習參數：噪聲強度 σ
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # 非參數噪聲（在 forward 時採樣）
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """初始化 μ（均勻分佈）和 σ（常數）。"""
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Factorised noise：f(x) = sgn(x) * sqrt(|x|)。"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        """重新採樣 factorised noise（每個 forward pass 前呼叫）。"""
        eps_i = self._scale_noise(self.in_features)
        eps_j = self._scale_noise(self.out_features)
        # Outer product：ε^w_ij = f(ε_i) * f(ε_j)
        self.weight_epsilon.copy_(eps_j.outer(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass。
        在 training 時使用 μ + σ⊙ε；
        在 evaluation 時僅使用 μ（noise=0）。
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)
