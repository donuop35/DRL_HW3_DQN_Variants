"""
src/agents/dqn_agent.py
========================
DQN Agent — 整合 QNetwork + ReplayBuffer + epsilon-greedy 策略。

支援：
    - Naive DQN（無 Target Network）
    - Basic DQN（S1 Experience Replay + S2 Target Network）
    - Double DQN（S3，HW3-2 用）
    - Dueling DQN（S4，HW3-2 用）

config flags 控制行為（對應 PROJECT_SPEC.md SPEC-03）：
    use_target_network: bool
    use_double_dqn:     bool
    use_dueling_dqn:    bool
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.buffers.replay_buffer import ReplayBuffer
from src.models.dqn import QNetwork, DuelingNet, build_target_network, sync_target_network
from src.utils.config import ExperimentConfig


class DQNAgent:
    """
    DQN Agent。

    根據 ExperimentConfig 的 algorithm_flags 自動切換：
    - use_dueling_dqn  → 使用 DuelingNet
    - use_target_network → 建立 target network
    - use_double_dqn   → Double DQN TD target 計算

    使用範例：
        cfg = load_config("configs/hw3_1_static/default.yaml")
        agent = DQNAgent(cfg)
        state = env.reset()
        action = agent.select_action(state)
        agent.push(state, action, reward, next_state, done)
        loss = agent.update()
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net_cfg = cfg.network
        alg_cfg = cfg.algorithm_flags

        # ── 選擇網路架構 ──────────────────────────────
        if alg_cfg.use_dueling_dqn:
            self.online_net = DuelingNet(
                input_dim=net_cfg.input_dim,
                hidden_dim=net_cfg.hidden_1,
                n_actions=net_cfg.output_dim,
            ).to(self.device)
        else:
            self.online_net = QNetwork(
                input_dim=net_cfg.input_dim,
                hidden_1=net_cfg.hidden_1,
                hidden_2=net_cfg.hidden_2,
                output_dim=net_cfg.output_dim,
            ).to(self.device)

        # ── Target Network（S2）──────────────────────
        self.use_target = alg_cfg.use_target_network
        self.target_net: Optional[QNetwork] = None
        if self.use_target:
            self.target_net = build_target_network(self.online_net)
            self.target_net.to(self.device)

        # ── Double DQN（S3）──────────────────────────
        self.use_double = alg_cfg.use_double_dqn

        # ── Replay Buffer（S1）──────────────────────
        self.replay = ReplayBuffer(capacity=cfg.training.replay_capacity)
        self.batch_size = cfg.training.batch_size

        # ── Optimizer & Loss ─────────────────────────
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=cfg.training.learning_rate,
        )
        self.loss_fn = nn.MSELoss()

        # ── Gradient Clipping ─────────────────────────
        self.use_grad_clip = cfg.training.use_gradient_clipping
        self.max_grad_norm = cfg.training.max_grad_norm

        # ── Hyperparameters ──────────────────────────
        self.gamma = cfg.training.gamma
        self.sync_freq = cfg.training.target_update_frequency

        # ── Epsilon ──────────────────────────────────
        self.epsilon = cfg.epsilon.epsilon_start
        self.epsilon_start = cfg.epsilon.epsilon_start
        self.epsilon_end = cfg.epsilon.epsilon_end
        self.epsilon_decay_type = cfg.epsilon.epsilon_decay_type
        self.epsilon_decay_steps = cfg.epsilon.epsilon_decay_steps
        self._total_steps = 0

        # ── LR Scheduler ─────────────────────────────
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if cfg.training.use_lr_scheduler:
            if cfg.training.lr_scheduler_type == "StepLR":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=cfg.training.lr_scheduler_step_size,
                    gamma=cfg.training.lr_scheduler_gamma,
                )
            elif cfg.training.lr_scheduler_type == "CosineAnnealingLR":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cfg.training.episodes,
                )

        print(f"[DQNAgent] device={self.device} | net={type(self.online_net).__name__} | "
              f"target={self.use_target} | double={self.use_double} | "
              f"replay={self.replay.capacity} | batch={self.batch_size}")

    # ──────────────────────────────────────────────
    # 動作選擇（ε-greedy）
    # ──────────────────────────────────────────────

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        ε-greedy 動作選擇。

        Args:
            state:     shape (state_dim,) numpy array
            eval_mode: 若 True，直接 greedy（不探索）

        Returns:
            action index (int)
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.cfg.network.output_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.online_net(state_t)
            return int(q_vals.argmax(dim=1).item())

    # ──────────────────────────────────────────────
    # Buffer 操作
    # ──────────────────────────────────────────────

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """將 transition 推入 replay buffer。"""
        self.replay.push(state, action, reward, next_state, done)
        self._total_steps += 1

    # ──────────────────────────────────────────────
    # 訓練更新
    # ──────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        從 replay buffer 採樣並更新 Q-Network。

        Returns:
            loss (float)，若 buffer 不足則回傳 None
        """
        if not self.replay.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # ── 計算 TD Target ────────────────────────────
        with torch.no_grad():
            if self.use_double and self.target_net is not None:
                # Double DQN：online 選動作，target 評估 Q 值
                next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            elif self.target_net is not None:
                # Standard DQN with Target Network
                next_q = self.target_net(next_states).max(dim=1)[0]
            else:
                # Naive DQN（無 Target Network）
                next_q = self.online_net(next_states).max(dim=1)[0]

            # TD Target = r + γ * max Q(s', a') * (1 - done)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # ── 計算當前 Q 值 ─────────────────────────────
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── 計算 Loss 並更新 ──────────────────────────
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # ── Target Network 同步（S2）─────────────────
        if self.use_target and self.target_net is not None:
            if self._total_steps % self.sync_freq == 0:
                sync_target_network(self.online_net, self.target_net)

        return loss.item()

    # ──────────────────────────────────────────────
    # Epsilon 衰減
    # ──────────────────────────────────────────────

    def decay_epsilon(self, episode: int) -> None:
        """
        依照設定更新 epsilon。

        Linear：epsilon -= 1/episodes（對應 starter code）
        Exponential：epsilon = end + (start - end) * exp(-episode / decay_steps)
        """
        ep_cfg = self.cfg.epsilon
        if ep_cfg.epsilon_decay_type == "linear":
            if self.epsilon > ep_cfg.epsilon_end:
                self.epsilon -= (ep_cfg.epsilon_start - ep_cfg.epsilon_end) / ep_cfg.epsilon_decay_steps
                self.epsilon = max(self.epsilon, ep_cfg.epsilon_end)
        elif ep_cfg.epsilon_decay_type == "exponential":
            self.epsilon = ep_cfg.epsilon_end + (ep_cfg.epsilon_start - ep_cfg.epsilon_end) * \
                           math.exp(-episode / ep_cfg.epsilon_decay_steps)

    def step_lr_scheduler(self) -> None:
        """每個 episode 結束後呼叫（若啟用 LR Scheduler）。"""
        if self.scheduler is not None:
            self.scheduler.step()

    def get_lr(self) -> float:
        """取得當前學習率。"""
        return self.optimizer.param_groups[0]['lr']

    # ──────────────────────────────────────────────
    # 儲存 / 載入
    # ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        """儲存 online network 權重。"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'online_net': self.online_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self._total_steps,
        }, path)
        print(f"[DQNAgent] Saved to: {path}")

    def load(self, path: str) -> None:
        """載入 online network 權重。"""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        self._total_steps = ckpt.get('total_steps', 0)
        if self.target_net is not None:
            sync_target_network(self.online_net, self.target_net)
        print(f"[DQNAgent] Loaded from: {path}")
