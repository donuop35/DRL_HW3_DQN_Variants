#!/usr/bin/env python3
"""
scripts/run_hw3_2_player.py
============================
HW3-2 執行腳本：三組 Player Mode 比較實驗。

P1: Basic DQN   (baseline)
P2: Double DQN  (S3: overestimation reduction)
P3: Dueling DQN (S4: value-advantage decomposition)

執行方式：
    python scripts/run_hw3_2_player.py              # 跑全部三組
    python scripts/run_hw3_2_player.py --algo basic
    python scripts/run_hw3_2_player.py --algo double
    python scripts/run_hw3_2_player.py --algo dueling

輸出：
    results/csv/hw3_2_player_{basic,double,dueling}_dqn_log.csv
    results/figures/hw3_2_player_reward_comparison.png
    results/figures/hw3_2_player_win_rate_comparison.png
    results/figures/hw3_2_player_loss_comparison.png
    results/figures/hw3_2_player_steps_comparison.png
    results/figures/hw3_2_player_final_metrics_bar.png
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.training.train_dqn import train, generate_figures
from src.plotting.plot_comparison import plot_hw3_2_comparison

CONFIGS = {
    "basic":   "configs/hw3_2_player/basic_dqn_player.yaml",
    "double":  "configs/hw3_2_player/double_dqn_player.yaml",
    "dueling": "configs/hw3_2_player/dueling_dqn_player.yaml",
}

LABELS = {
    "basic":   "Basic DQN (Baseline)",
    "double":  "Double DQN",
    "dueling": "Dueling DQN",
}


def run_single(algo_key: str) -> dict:
    """執行單一演算法訓練，回傳 final_metrics。"""
    cfg_path = CONFIGS[algo_key]
    print(f"\n{'='*60}")
    print(f"  HW3-2 [{algo_key.upper()}] — Player Mode")
    print(f"  Config: {cfg_path}")
    print(f"{'='*60}")
    cfg = load_config(cfg_path)
    metrics = train(cfg, verbose_every=500)
    generate_figures(cfg)
    return metrics


def generate_comparison_figures(results: dict) -> None:
    """生成三組比較圖表。"""
    import os
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)

    # CSV 路徑對應
    csv_map = {
        label: f"results/csv/{cfg_id}_log.csv"
        for cfg_id, label in [
            ("hw3_2_player_basic_dqn",   "Basic DQN"),
            ("hw3_2_player_double_dqn",  "Double DQN"),
            ("hw3_2_player_dueling_dqn", "Dueling DQN"),
        ]
        if Path(f"results/csv/{cfg_id}_log.csv").exists()
    }

    if len(csv_map) < 2:
        print(f"  [WARN] 僅有 {len(csv_map)} 個 CSV，跳過比較圖生成")
        return

    print("\n[Comparison] Generating HW3-2 comparison figures ...")
    plot_hw3_2_comparison(
        csv_map=csv_map,
        output_dir=output_dir,
        window=100,
        title_prefix="HW3-2 Player Mode",
    )
    print(f"  [Comparison] Done → {output_dir}/hw3_2_player_*.png")


def print_summary(results: dict) -> None:
    """輸出三組實驗結果對比。"""
    print("\n" + "="*60)
    print("  HW3-2 Final Results Summary")
    print("="*60)
    print(f"  {'Algorithm':<20} {'Win Rate':>10} {'Avg Reward':>12} {'Avg Steps':>10}")
    print("  " + "-"*55)
    for algo, m in results.items():
        label = LABELS.get(algo, algo)
        print(f"  {label:<20} {m['win_rate']*100:>9.1f}% "
              f"{m['avg_reward']:>12.2f} {m['avg_steps']:>10.1f}")
    best = max(results.items(), key=lambda kv: kv[1]['win_rate'])
    print(f"\n  🏆 Best: {LABELS.get(best[0], best[0])} "
          f"(Win Rate {best[1]['win_rate']*100:.1f}%)")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description="HW3-2: DQN Variants on Player Mode")
    parser.add_argument("--algo", choices=["basic", "double", "dueling", "all"],
                        default="all", help="Which algorithm to run")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip comparison figure generation")
    return parser.parse_args()


def main():
    args = parse_args()

    algos = ["basic", "double", "dueling"] if args.algo == "all" else [args.algo]

    results = {}
    for algo in algos:
        results[algo] = run_single(algo)

    print_summary(results)

    if not args.no_compare and len(algos) > 1:
        generate_comparison_figures(results)

    print("\n[HW3-2] All done.")
    print("  CSVs:    results/csv/hw3_2_player_*_dqn_log.csv")
    print("  Figs:    results/figures/hw3_2_player_*.png")


if __name__ == "__main__":
    main()
