"""
src/training/train_dqn.py
==========================
統一訓練迴圈 — 接入 harness（config / logger / metrics / plotting）。

對應教授 starter code 程式 3.8 訓練架構：
    - 5000 episodes
    - Experience Replay (mem=1000, batch=200)
    - Target Network (sync_freq=500)
    - ε-greedy linear decay 1.0 → 0.1

所有結果透過 ExperimentLogger 寫入 CSV，
圖表由 plot_curves.plot_all_curves() 生成。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── 確保根目錄在 path 中 ──────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.gridworld_env import GridworldEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.config import ExperimentConfig, load_config
from src.utils.logger import ExperimentLogger
from src.utils.seeding import set_global_seed


def evaluate_agent(
    agent: DQNAgent,
    mode: str,
    n_games: int = 200,
    max_steps: int = 50,
) -> dict:
    """
    評估 agent 的勝率（greedy policy，不探索）。

    Args:
        agent:    已訓練的 DQN Agent
        mode:     環境模式
        n_games:  測試場數
        max_steps: 每場最大步數

    Returns:
        dict 含 win_rate, avg_reward, avg_steps
    """
    env = GridworldEnv(mode=mode, noise_scale=0.01)
    wins = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(n_games):
        state = env.reset()
        ep_reward = 0.0
        for t in range(max_steps):
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            if done:
                if info["terminal_state"] == "goal":
                    wins += 1
                break
            state = next_state
        total_reward += ep_reward

    return {
        "win_rate": wins / n_games,
        "avg_reward": total_reward / n_games,
        "avg_steps": total_steps / n_games,
    }


def train(
    cfg: ExperimentConfig,
    verbose_every: int = 200,
    eval_every: int = 500,
    save_checkpoint: bool = True,
) -> dict:
    """
    主訓練迴圈。

    Args:
        cfg:             ExperimentConfig 物件
        verbose_every:   每隔幾個 episode 印一次進度
        eval_every:      每隔幾個 episode 評估一次勝率
        save_checkpoint: 是否儲存最終 checkpoint

    Returns:
        final_metrics dict
    """
    # ── Step 1：Seed ──────────────────────────────
    set_global_seed(cfg.seed)

    # ── Step 2：建立環境 ──────────────────────────
    env = GridworldEnv(mode=cfg.mode, noise_scale=0.01)
    print(f"[Train] env mode={cfg.mode} | state_dim={env.state_dim} | n_actions={env.n_actions}")

    # ── Step 3：建立 Agent ────────────────────────
    agent = DQNAgent(cfg)

    # ── Step 4：Logger ────────────────────────────
    logger = ExperimentLogger(cfg)

    # ── Step 5：訓練迴圈 ──────────────────────────
    total_episodes = cfg.training.episodes
    max_steps = cfg.training.max_steps_per_episode
    start_time = time.time()

    print(f"[Train] Starting {total_episodes} episodes ...")

    for episode in range(total_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_losses = []
        ep_steps = 0
        terminal = "unknown"
        win = False

        for t in range(max_steps):
            # ε-greedy 選動作
            action = agent.select_action(state)

            # 執行動作
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            # 儲存 transition
            agent.push(state, action, reward, next_state, done)

            # 更新網路
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state

            if done:
                terminal = info["terminal_state"]
                win = (terminal == "goal")
                break

        # Epsilon decay
        agent.decay_epsilon(episode)
        agent.step_lr_scheduler()

        # 計算本 episode 平均 loss
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        # 記錄到 CSV
        logger.log_episode(
            episode=episode,
            episode_reward=ep_reward,
            episode_steps=ep_steps,
            loss_mean=mean_loss,
            epsilon=agent.epsilon,
            win=win,
            terminal_state=terminal,
            learning_rate=agent.get_lr(),
            buffer_size=len(agent.replay),
        )

        # 進度輸出
        if (episode + 1) % verbose_every == 0:
            elapsed = time.time() - start_time
            print(f"  Ep {episode+1:5d}/{total_episodes} | "
                  f"reward={ep_reward:6.1f} | loss={mean_loss:.4f} | "
                  f"ε={agent.epsilon:.3f} | buf={len(agent.replay)} | "
                  f"t={elapsed:.0f}s")

    logger.close()

    # ── Step 6：最終評估 ──────────────────────────
    print("\n[Train] Running final evaluation (200 games, greedy) ...")
    final_metrics = evaluate_agent(agent, mode=cfg.mode, n_games=200, max_steps=max_steps)
    print(f"  Final Win Rate:    {final_metrics['win_rate']*100:.1f}%")
    print(f"  Final Avg Reward:  {final_metrics['avg_reward']:.2f}")
    print(f"  Final Avg Steps:   {final_metrics['avg_steps']:.1f}")

    # ── Step 7：儲存 Checkpoint ───────────────────
    if save_checkpoint:
        ckpt_path = f"{cfg.checkpoint_dir}/final_model.pt"
        agent.save(ckpt_path)

    total_time = time.time() - start_time
    print(f"\n[Train] Done in {total_time:.1f}s")

    return final_metrics


def generate_figures(cfg: ExperimentConfig) -> dict:
    """
    從 CSV 生成所有訓練曲線圖。

    Returns:
        dict 含各圖表路徑
    """
    from src.plotting.plot_curves import plot_all_curves

    csv_path = f"{cfg.log_dir}/{cfg.experiment_id}_log.csv"
    paths = plot_all_curves(
        csv_path=csv_path,
        output_dir=cfg.figures_dir,
        window=100,
        smoke_test=False,  # ← 正式圖表，無浮水印
    )
    print(f"[Figures] Generated {len(paths)} figures in {cfg.figures_dir}/")
    return paths
