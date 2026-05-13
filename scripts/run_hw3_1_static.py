#!/usr/bin/env python3
"""
scripts/run_hw3_1_static.py
============================
HW3-1 執行腳本：Basic DQN on Static Mode GridWorld。

執行方式：
    python scripts/run_hw3_1_static.py
    python scripts/run_hw3_1_static.py --config configs/hw3_1_static/basic_dqn_static.yaml
    python scripts/run_hw3_1_static.py --config configs/hw3_1_static/basic_dqn_static.yaml --seed 42

輸出：
    results/csv/hw3_1_static_basic_dqn_log.csv
    results/figures/hw3_1_static_basic_dqn_*.png
    results/checkpoints/hw3_1_static_basic_dqn/final_model.pt
"""

import argparse
import sys
from pathlib import Path

# 確保 repo root 在 path 中
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.training.train_dqn import train, generate_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description="HW3-1: Basic DQN on Static Mode GridWorld"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hw3_1_static/basic_dqn_static.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed in config (optional)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of episodes (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  HW3-1: Basic DQN — Static Mode GridWorld")
    print(f"  Config: {args.config}")
    print("=" * 60)

    # 載入 Config
    cfg = load_config(args.config)

    # 覆寫 seed（若有指定）
    if args.seed is not None:
        cfg.seed = args.seed
        print(f"  Seed overridden to: {cfg.seed}")

    # 覆寫 episodes（若有指定）
    if args.episodes is not None:
        cfg.training.episodes = args.episodes
        print(f"  Episodes overridden to: {cfg.training.episodes}")

    print(f"\n  Experiment ID:  {cfg.experiment_id}")
    print(f"  Mode:           {cfg.mode}")
    print(f"  Algorithm:      {cfg.algorithm}")
    print(f"  Episodes:       {cfg.training.episodes}")
    print(f"  Batch Size:     {cfg.training.batch_size}")
    print(f"  Replay Buffer:  {cfg.training.replay_capacity}")
    print(f"  Gamma:          {cfg.training.gamma}")
    print(f"  LR:             {cfg.training.learning_rate}")
    print(f"  Target Network: {cfg.algorithm_flags.use_target_network}")
    print(f"  Sync Freq:      {cfg.training.target_update_frequency}")
    print(f"  Epsilon:        {cfg.epsilon.epsilon_start} → {cfg.epsilon.epsilon_end} ({cfg.epsilon.epsilon_decay_type})")
    print()

    # 執行訓練
    final_metrics = train(cfg, verbose_every=500, eval_every=500)

    # 生成圖表
    print("\n[Main] Generating training curve figures ...")
    figure_paths = generate_figures(cfg)

    print("\n" + "=" * 60)
    print("  HW3-1 Complete!")
    print(f"  CSV:       results/csv/{cfg.experiment_id}_log.csv")
    print(f"  Figures:   results/figures/{cfg.experiment_id}_*.png")
    print(f"  Checkpoint: {cfg.checkpoint_dir}/final_model.pt")
    print(f"\n  Final Evaluation Results:")
    print(f"    Win Rate:   {final_metrics['win_rate']*100:.1f}%")
    print(f"    Avg Reward: {final_metrics['avg_reward']:.2f}")
    print(f"    Avg Steps:  {final_metrics['avg_steps']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
