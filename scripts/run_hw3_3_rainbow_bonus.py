#!/usr/bin/env python3
"""
scripts/run_hw3_3_rainbow_bonus.py
====================================
HW3-3 E4 Rainbow DQN Bonus 執行腳本。

E4 是獨立 Bonus Experiment，完全不修改 E1-E3 結果。
E1-E3 若需重跑，請使用：python scripts/run_hw3_3_random.py

執行方式：
    python scripts/run_hw3_3_rainbow_bonus.py
    python scripts/run_hw3_3_rainbow_bonus.py --no-compare

輸出：
    results/csv/hw3_3_random_e4_rainbow_bonus_log.csv
    results/figures/hw3_3_random_e4_rainbow_*.png      (4 張個別圖)
    results/figures/hw3_3_random_*_comparison_e1_e2_e3_e4.png (5 張比較圖)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.training.lightning_rainbow_module import LightningRainbowModule
from src.training.train_dqn import generate_figures
from src.plotting.plot_comparison import plot_hw3_4_comparison


E4_CONFIG = "configs/hw3_3_random/e4_rainbow_dqn_bonus.yaml"

E1_CSV = "results/csv/hw3_3_random_e1_baseline_log.csv"
E2_CSV = "results/csv/hw3_3_random_e2_stabilized_log.csv"
E3_CSV = "results/csv/hw3_3_random_e3_per_stabilized_log.csv"


def run_e4() -> dict:
    cfg = load_config(E4_CONFIG)
    print(f"\n{'='*60}")
    print(f"  HW3-3 [E4] — Rainbow DQN Bonus")
    print(f"  Algorithm: {cfg.algorithm}")
    print(f"  Components: Double + Dueling + PER + N-step(3) + C51(51) + NoisyNet")
    print(f"  Note: E4 is BONUS only; E1-E3 remain unchanged.")
    print(f"{'='*60}\n")
    set_global_seed(cfg.seed)
    module  = LightningRainbowModule(cfg)
    metrics = module.run_training(verbose_every=500)
    # individual figures
    generate_figures(cfg)
    return metrics


def generate_comparison_figures() -> None:
    csv_map = {
        "E1 Baseline":      E1_CSV,
        "E2 Stabilized":    E2_CSV,
        "E3 PER+Stabilized":E3_CSV,
        "E4 Rainbow Bonus": f"results/csv/{load_config(E4_CONFIG).experiment_id}_log.csv",
    }
    # Only include available CSVs
    csv_map = {k: v for k, v in csv_map.items() if Path(v).exists()}
    if len(csv_map) < 2:
        print(f"[WARN] Not enough CSVs for comparison.")
        return
    plot_hw3_4_comparison(csv_map, "results/figures", window=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()

    metrics = run_e4()

    print(f"\n{'='*60}")
    print(f"  E4 Rainbow Bonus Final Result")
    print(f"  Final Win Rate:    {metrics['win_rate']*100:.1f}%")
    print(f"  Final Avg Reward:  {metrics['avg_reward']:.2f}")
    print(f"  Final Avg Steps:   {metrics['avg_steps']:.1f}")
    print(f"{'='*60}")

    if not args.no_compare:
        generate_comparison_figures()

    print("\n[E4 Rainbow] Done. E1-E3 are untouched.")
    print("  CSV:  results/csv/hw3_3_random_e4_rainbow_bonus_log.csv")
    print("  Figs: results/figures/hw3_3_random_e4_*.png")
    print("  Comparison figs: results/figures/hw3_3_random_*_e1_e2_e3_e4.png")


if __name__ == "__main__":
    main()
