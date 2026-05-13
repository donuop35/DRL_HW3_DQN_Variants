"""
src/training/lightning_rainbow_module.py
=========================================
Rainbow DQN PyTorch Lightning Module（E4 Bonus）。

整合六個 DQN 改進：
    1. Double DQN             — online 選 action，target 評估 Q
    2. Dueling Network        — V(s) + A(s,a) 分離（在 C51DuelingNetwork 內）
    3. Prioritized ER + N-step— NStepPERBuffer
    4. C51 Distributional DQN — 輸出 Q 分佈，最小化 KL divergence
    5. NoisyNet               — 取代 ε-greedy 探索

本 Module 僅用於 E4 Bonus，不影響 E1-E3。
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.c51_dueling_dqn import C51DuelingNetwork
from src.buffers.nstep_per_buffer import NStepPERBuffer
from src.envs.gridworld_env import GridworldEnv
from src.utils.logger import ExperimentLogger
from src.utils.seeding import set_global_seed


class LightningRainbowModule(pl.LightningModule):
    """
    Rainbow DQN Lightning Module（E4 Bonus）。

    Rainbow 組件清單：
        ✅ Double DQN（online select + target eval）
        ✅ Dueling Network（C51DuelingNetwork 內建）
        ✅ Prioritized ER（NStepPERBuffer）
        ✅ N-step Return（NStepPERBuffer）
        ✅ Distributional DQN / C51（categorical projection）
        ✅ NoisyNet（C51DuelingNetwork noisy=True）
        ℹ️ Epsilon：NoisyNet 接管探索，epsilon 設為 0（保留 greedy wrapper 相容性）
    """

    automatic_optimization = False

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg  = cfg
        flags     = cfg.algorithm_flags
        train     = cfg.training

        # ── C51 超參數 ────────────────────────────
        self.n_atoms   = getattr(flags, "c51_atoms", 51)
        self.v_min     = getattr(flags, "c51_v_min", -10.0)
        self.v_max     = getattr(flags, "c51_v_max", 10.0)
        self.n_actions = cfg.network.output_dim
        delta_z        = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.register_buffer(
            "support",
            torch.linspace(self.v_min, self.v_max, self.n_atoms),
        )
        self.delta_z = delta_z

        # ── 網路 ──────────────────────────────────
        self.online_net = C51DuelingNetwork(
            input_dim = cfg.network.input_dim,
            hidden_1  = cfg.network.hidden_1,
            hidden_2  = cfg.network.hidden_2,
            n_actions = self.n_actions,
            n_atoms   = self.n_atoms,
            v_min     = self.v_min,
            v_max     = self.v_max,
            noisy     = getattr(flags, "use_noisy_net", True),
        )
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        # ── N-step PER Buffer ─────────────────────
        n_step = getattr(flags, "n_step", 3)
        self.replay = NStepPERBuffer(
            capacity    = train.replay_capacity,
            n_step      = n_step,
            gamma       = train.gamma,
            alpha       = getattr(flags, "per_alpha", 0.6),
            beta_start  = getattr(flags, "per_beta_start", 0.4),
            beta_end    = getattr(flags, "per_beta_end", 1.0),
            per_epsilon = getattr(flags, "per_epsilon", 1e-5),
        )
        self.n_step = n_step
        self.gamma  = train.gamma

        # ── 超參數 ────────────────────────────────
        self.batch_size    = train.batch_size
        self.sync_freq     = train.target_update_frequency
        self.episodes      = train.episodes
        self.max_steps     = train.max_steps_per_episode
        self.lr            = train.learning_rate

        self.use_grad_clip = getattr(train, "use_gradient_clipping", True)
        self.max_grad_norm = getattr(train, "max_grad_norm", 10.0)
        self.use_scheduler = getattr(train, "use_lr_scheduler", False)

        # NoisyNet 取代 epsilon（epsilon 固定 0）
        self.epsilon = 0.0

        self.global_step_count = 0

        print(f"[RainbowDQN] C51(atoms={self.n_atoms}, v=[{self.v_min},{self.v_max}]) | "
              f"n_step={n_step} | PER | Dueling | NoisyNet | Double")

    # ─────────────────────────────────────────────
    # Lightning API
    # ─────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = Adam(self.online_net.parameters(), lr=self.lr, eps=1.5e-4)
        if self.use_scheduler:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.episodes * self.max_steps
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def _categorical_projection(
        self,
        rewards:     torch.Tensor,
        next_states: torch.Tensor,
        dones:       torch.Tensor,
    ) -> torch.Tensor:
        """
        C51 Distributional TD Target：Categorical Projection。

        Algorithm（Bellemare et al., 2017）：
        1. 用 target network 計算 next state 分佈
        2. 用 online network 選最好的動作（Double DQN）
        3. 計算 n-step projected support：z_j' = clip(r + γ^n * z_j)
        4. 線性內插回 [v_min, v_max] 的原始格點
        5. 回傳 projected 分佈（shape: batch_size x n_atoms）
        """
        batch_size = rewards.shape[0]
        gamma_n    = self.gamma ** self.n_step

        with torch.no_grad():
            # Double DQN：online net 選動作
            next_q_online  = self.online_net.get_q_values(next_states)
            best_actions   = next_q_online.argmax(dim=1)

            # target net 取得最佳動作的分佈
            next_dist_all  = self.target_net.get_q_dist(next_states)    # (B, n_actions, n_atoms)
            next_dist      = next_dist_all[range(batch_size), best_actions]  # (B, n_atoms)

            # 計算 projected support：z' = clip(r + γ^n * z_j, v_min, v_max)
            support = self.support.unsqueeze(0)                              # (1, n_atoms)
            tz      = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n * support
            tz      = tz.clamp(self.v_min, self.v_max)                      # (B, n_atoms)

            # 計算在 original support 上的索引
            b   = (tz - self.v_min) / self.delta_z                          # (B, n_atoms)
            l   = b.floor().long().clamp(0, self.n_atoms - 1)
            u   = b.ceil().long().clamp(0, self.n_atoms - 1)

            # 線性內插：分配 probability 到相鄰格點
            m = torch.zeros(batch_size, self.n_atoms, device=rewards.device)
            # 防止 l==u 時雙重計數
            eq_mask  = (l == u)
            neq_mask = ~eq_mask

            m.scatter_add_(1, l, next_dist * (u.float() - b) * neq_mask.float())
            m.scatter_add_(1, u, next_dist * (b - l.float()) * neq_mask.float())
            m.scatter_add_(1, l, next_dist * eq_mask.float())

        return m  # 目標分佈，shape (B, n_atoms)

    def training_step(self, batch, batch_idx):
        """C51 Distributional TD + Double + IS weights。"""
        opt = self.optimizers()

        sample = batch  # passed in as dict from _run_training_step
        states      = sample["states"]
        actions     = sample["actions"]
        rewards     = sample["rewards"]
        next_states = sample["next_states"]
        dones       = sample["dones"]
        weights     = sample["weights"]
        indices     = sample["indices"]

        # Reset NoisyNet noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # ── Online distribution for (s, a) ────────
        dist_logits = self.online_net(states)              # (B, n_actions, n_atoms)
        log_dist    = F.log_softmax(dist_logits, dim=-1)  # (B, n_actions, n_atoms)
        log_pa      = log_dist[range(states.shape[0]), actions]  # (B, n_atoms)

        # ── Target distribution projection ────────
        m = self._categorical_projection(rewards, next_states, dones)  # (B, n_atoms)

        # ── KL divergence loss（IS weighted）──────
        # loss = -Σ m_j * log p_j
        loss_per_sample = -(m * log_pa).sum(dim=-1)       # (B,)
        loss = (weights.squeeze(1) * loss_per_sample).mean()

        # ── Backward ─────────────────────────────
        opt.zero_grad()
        try:
            _t = super(pl.LightningModule, self).__getattribute__("_trainer")
            if _t is not None:
                self.manual_backward(loss)
            else:
                loss.backward()
        except (AttributeError, RuntimeError):
            loss.backward()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(), self.max_grad_norm
            )
        opt.step()

        if self.use_scheduler:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

        # ── PER priority update（TD error ≈ loss per sample）────
        td_errors = loss_per_sample.detach().cpu().numpy()
        self.replay.update_priorities(indices, td_errors)

        # ── Target sync ──────────────────────────
        self.global_step_count += 1
        if self.global_step_count % self.sync_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss

    def select_action(self, state: np.ndarray) -> int:
        """NoisyNet greedy action selection（NoisyNet 提供隱式探索）。"""
        self.online_net.reset_noise()
        with torch.no_grad():
            q = self.online_net.get_q_values(
                torch.FloatTensor(state).unsqueeze(0)
            )
        return int(q.argmax(dim=1).item())

    def run_training(self, verbose_every: int = 500) -> dict:
        """執行完整 Rainbow 訓練。"""
        opt_config = self.configure_optimizers()
        if isinstance(opt_config, dict):
            self._optimizer = opt_config["optimizer"]
            self._scheduler = opt_config["lr_scheduler"]
        else:
            self._optimizer = opt_config
            self._scheduler = None

        class _FakeOpt:
            def __init__(self, real): self._real = real
            def zero_grad(self): self._real.zero_grad()
            def step(self): self._real.step()

        self._manual_opt = _FakeOpt(self._optimizer)

        logger = ExperimentLogger(self.cfg)
        env    = GridworldEnv(mode=self.cfg.mode)

        import time
        t0 = time.time()

        for ep in range(1, self.episodes + 1):
            state = env.reset()
            self.online_net.train()
            ep_reward, ep_steps = 0.0, 0
            ep_losses = []
            won = False

            for _ in range(self.max_steps):
                action      = self.select_action(state)
                ns, r, done, info = env.step(action)
                won         = (info["terminal_state"] == "goal")
                self.replay.push(state, action, r, ns, done)
                state       = ns
                ep_reward  += r
                ep_steps   += 1

                if len(self.replay) >= self.batch_size:
                    beta  = self.replay.beta_by_step(
                        self.global_step_count,
                        self.episodes * self.max_steps,
                    )
                    batch = self.replay.sample(self.batch_size, beta)
                    loss  = self._run_training_step(batch)
                    ep_losses.append(loss)

                if done:
                    break

            # Flush n-step buffer at episode end
            self.replay.flush()

            cur_lr   = self._optimizer.param_groups[0]["lr"]
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

            logger.log_episode(
                episode       = ep,
                episode_reward= ep_reward,
                episode_steps = ep_steps,
                loss_mean     = avg_loss,
                epsilon       = 0.0,   # NoisyNet，無 epsilon
                win           = won,
                learning_rate = cur_lr,
                buffer_size   = len(self.replay),
            )

            if ep % verbose_every == 0:
                print(f"  Ep {ep:5d}/{self.episodes} | r={ep_reward:6.1f} | "
                      f"loss={avg_loss:.4f} | buf={len(self.replay)} | "
                      f"lr={cur_lr:.6f} | t={int(time.time()-t0)}s")

        logger.close()

        # Final evaluation（greedy，NoisyNet eval mode）
        print(f"\n[Rainbow] Final evaluation (200 games, eval mode) ...")
        self.online_net.eval()
        wins, total_r, total_st = 0, 0.0, 0
        for _ in range(200):
            s = env.reset()
            for _ in range(self.max_steps):
                a = self.select_action(s)
                s, r, d, info = env.step(a)
                total_r += r
                total_st += 1
                if d:
                    if info["terminal_state"] == "goal":
                        wins += 1
                    break
        win_rate   = wins / 200
        avg_reward = total_r / 200
        avg_steps  = total_st / 200
        print(f"  Final Win Rate:   {win_rate*100:.1f}%")
        print(f"  Final Avg Reward: {avg_reward:.2f}")
        print(f"  Final Avg Steps:  {avg_steps:.1f}")

        ckpt_dir = Path("results/checkpoints") / self.cfg.experiment_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), ckpt_dir / "final_model.pt")
        print(f"  [Rainbow] Saved to: {ckpt_dir}/final_model.pt")

        return {"win_rate": win_rate, "avg_reward": avg_reward, "avg_steps": avg_steps}

    def _run_training_step(self, batch_data) -> float:
        import types

        def fake_optimizers(s): return self._manual_opt
        def fake_lr_schedulers(s): return self._scheduler

        orig_opts   = self.optimizers
        orig_scheds = self.lr_schedulers
        self.optimizers    = types.MethodType(fake_optimizers, self)
        self.lr_schedulers = types.MethodType(fake_lr_schedulers, self)

        loss = self.training_step(batch_data, 0)

        self.optimizers    = orig_opts
        self.lr_schedulers = orig_scheds
        return float(loss.item())

    def register_buffer(self, name, tensor, persistent=True):
        """Override to handle case where tensor is None during __init__."""
        if tensor is None:
            return
        super().register_buffer(name, tensor, persistent=persistent)
