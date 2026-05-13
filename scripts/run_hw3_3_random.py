#!/usr/bin/env python3
"""
scripts/run_hw3_3_random.py
============================
HW3-3 執行腳本：三組 Random Mode 正式實驗。

E1: Random DQN Baseline     (對照組，無穩定技術)
E2: Stabilized DQN          (+ Gradient Clipping + LR Scheduling + Exp Epsilon)
E3: PER-DQN + Stabilization (+ PER，正式 HW3-3 主方法)

全部使用 PyTorch Lightning Module（LightningDQNModule）。

執行方式：
    python scripts/run_hw3_3_random.py           # 跑全部三組
    python scripts/run_hw3_3_random.py --exp e1
    python scripts/run_hw3_3_random.py --exp e2
    python scripts/run_hw3_3_random.py --exp e3

輸出：
    results/csv/hw3_3_random_e{1,2,3}_*_log.csv
    results/figures/hw3_3_random_*_comparison_e1_e2_e3.png (7 張)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.training.lightning_dqn_module import LightningDQNModule
from src.training.train_dqn import generate_figures
from src.plotting.plot_comparison import plot_hw3_3_comparison

CONFIGS = {
    "e1": "configs/hw3_3_random/e1_random_dqn_baseline.yaml",
    "e2": "configs/hw3_3_random/e2_stabilized_dqn.yaml",
    "e3": "configs/hw3_3_random/e3_per_dqn_stabilized.yaml",
}

LABELS = {
    "e1": "E1 Baseline",
    "e2": "E2 Stabilized",
    "e3": "E3 PER+Stabilized",
}


def run_single(exp_key: str) -> dict:
    """執行單一實驗，回傳 final_metrics。"""
    cfg_path = CONFIGS[exp_key]
    cfg = load_config(cfg_path)

    print(f"\n{'='*60}")
    print(f"  HW3-3 [{exp_key.upper()}] — Random Mode")
    print(f"  Algorithm: {cfg.algorithm}")
    print(f"  grad_clip={cfg.training.use_gradient_clipping} | "
          f"lr_sched={cfg.training.use_lr_scheduler} | "
          f"PER={cfg.algorithm_flags.use_per} | "
          f"eps_type={cfg.epsilon.epsilon_decay_type}")
    print(f"{'='*60}")

    set_global_seed(cfg.seed)
    module = LightningDQNModule(cfg)
    metrics = module.run_training(verbose_every=500)

    # 個別曲線圖
    generate_figures(cfg)

    return metrics


def generate_comparison_figures(completed_exps: list) -> None:
    """生成 E1/E2/E3 比較圖表。"""
    csv_map = {}
    for exp_key in completed_exps:
        cfg = load_config(CONFIGS[exp_key])
        csv_path = f"results/csv/{cfg.experiment_id}_log.csv"
        if Path(csv_path).exists():
            csv_map[LABELS[exp_key]] = csv_path

    if len(csv_map) < 2:
        print(f"  [WARN] 只有 {len(csv_map)} 個 CSV，跳過比較圖")
        return

    print(f"\n[Comparison] Generating HW3-3 comparison figures ...")
    plot_hw3_3_comparison(
        csv_map=csv_map,
        output_dir="results/figures",
        window=100,
        title_prefix="HW3-3 Random Mode",
    )
    print("  [Comparison] Done → results/figures/hw3_3_random_*.png")


def print_summary(results: dict) -> None:
    """輸出三組實驗結果對比。"""
    print("\n" + "=" * 65)
    print("  HW3-3 Final Results Summary (Random Mode)")
    print("=" * 65)
    print(f"  {'Experiment':<22} {'Win Rate':>10} {'Avg Reward':>12} {'Avg Steps':>10}")
    print("  " + "-" * 58)
    for exp, m in results.items():
        label = LABELS.get(exp, exp)
        print(f"  {label:<22} {m['win_rate']*100:>9.1f}% "
              f"{m['avg_reward']:>12.2f} {m['avg_steps']:>10.1f}")
    best = max(results.items(), key=lambda kv: kv[1]['win_rate'])
    print(f"\n  🏆 Best: {LABELS.get(best[0], best[0])} "
          f"(Win Rate {best[1]['win_rate']*100:.1f}%)")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="HW3-3: E1/E2/E3 on Random Mode")
    parser.add_argument("--exp", choices=["e1", "e2", "e3", "all"],
                        default="all")
    parser.add_argument("--no-compare", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    exps = ["e1", "e2", "e3"] if args.exp == "all" else [args.exp]

    results = {}
    for exp in exps:
        results[exp] = run_single(exp)

    print_summary(results)

    if not args.no_compare and len(exps) > 1:
        generate_comparison_figures(exps)

    print("\n[HW3-3] All done.")
    print("  CSVs:  results/csv/hw3_3_random_e*_log.csv")
    print("  Figs:  results/figures/hw3_3_random_*.png")


if __name__ == "__main__":
    main()
