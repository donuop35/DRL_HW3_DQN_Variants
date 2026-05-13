#!/usr/bin/env python3
"""
scripts/smoke_test.py
=====================
Harness Smoke Test — Phase 3 驗收腳本。

目的：驗證 config / logger / metrics / plotting pipeline 可正常運作。
⚠️  本腳本產生的圖表標記為 SMOKE TEST，不得用於正式報告。

執行方式：
    python scripts/smoke_test.py

驗收通過條件：
    1. Config 可從 YAML 載入
    2. Logger 可寫入 CSV
    3. Metrics 可從 CSV 計算
    4. Plot 可產出 placeholder 圖（帶 SMOKE TEST 浮水印）
    5. 所有測試 PASSED
"""

import sys
import os

# 確保 repo 根目錄在 Python path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import random
import math
import time
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────
# 測試計數
# ──────────────────────────────────────────────

PASSED = 0
FAILED = 0


def _ok(msg: str):
    global PASSED
    PASSED += 1
    print(f"  ✅ PASS: {msg}")


def _fail(msg: str, e: Exception):
    global FAILED
    FAILED += 1
    print(f"  ❌ FAIL: {msg}")
    print(f"         Error: {e}")


# ──────────────────────────────────────────────
# Test 1: Config 載入
# ──────────────────────────────────────────────

def test_config():
    print("\n[Test 1] Config System")
    try:
        from src.utils.config import load_config, config_to_dict
        cfg = load_config("configs/hw3_1_static/default.yaml")
        assert cfg.experiment_id == "hw3_1_static_naive_dqn"
        assert cfg.mode == "static"
        assert cfg.training.episodes == 5000
        assert cfg.training.gamma == 0.9
        assert cfg.epsilon.epsilon_decay_type == "linear"
        assert cfg.algorithm_flags.use_target_network == True
        _ok("load_config() 成功讀取 hw3_1_static/default.yaml")

        d = config_to_dict(cfg)
        assert "experiment_id" in d
        assert "use_double_dqn" in d
        _ok("config_to_dict() 序列化成功")

        # 測試所有 HW 的 config
        for yaml_path in [
            "configs/hw3_1_static/default.yaml",
            "configs/hw3_2_player/naive_dqn.yaml",
            "configs/hw3_2_player/double_dqn.yaml",
            "configs/hw3_2_player/dueling_dqn.yaml",
            "configs/hw3_3_random/e1_baseline.yaml",
            "configs/hw3_3_random/e2_stabilized.yaml",
            "configs/hw3_3_random/e3_per.yaml",
            "configs/hw3_3_random/e4_rainbow.yaml",
        ]:
            c = load_config(yaml_path)
            assert c.experiment_id, f"Missing experiment_id in {yaml_path}"
        _ok("所有 8 個 YAML config 載入成功")

    except Exception as e:
        _fail("Config system", e)


# ──────────────────────────────────────────────
# Test 2: Seeding
# ──────────────────────────────────────────────

def test_seeding():
    print("\n[Test 2] Seeding")
    try:
        # 只測試 Python random + numpy 的可重現性（不依賴 torch）
        random.seed(42)
        np.random.seed(42)
        r1 = random.random()
        n1 = float(np.random.rand())

        random.seed(42)
        np.random.seed(42)
        r2 = random.random()
        n2 = float(np.random.rand())

        assert r1 == r2, "Python random seeding 不可重現"
        assert n1 == n2, "NumPy seeding 不可重現"
        _ok("Python random + NumPy seeding 可重現")

        # 嘗試 torch（若未安裝則跳過）
        try:
            import torch
            from src.utils.seeding import set_global_seed
            set_global_seed(42)
            t1 = float(torch.rand(1))
            set_global_seed(42)
            t2 = float(torch.rand(1))
            assert t1 == t2
            _ok("PyTorch seeding 可重現")
        except ImportError:
            print("  ⚠️  SKIP: torch 未安裝，跳過 PyTorch seeding test（正式訓練前需安裝）")

    except Exception as e:
        _fail("Seeding", e)


# ──────────────────────────────────────────────
# Test 3: Logger 寫 CSV
# ──────────────────────────────────────────────

def test_logger():
    print("\n[Test 3] Logger")
    try:
        from src.utils.config import load_config, ExperimentConfig
        from src.utils.logger import ExperimentLogger

        cfg = load_config("configs/hw3_1_static/default.yaml")
        # 使用 smoke test experiment id，避免污染正式 CSV
        cfg.experiment_id = "SMOKE_TEST_DO_NOT_USE"
        cfg.log_dir = "results/csv"
        cfg.checkpoint_dir = "results/checkpoints/SMOKE_TEST_DO_NOT_USE"

        with ExperimentLogger(cfg) as logger:
            for ep in range(10):
                logger.log_episode(
                    episode=ep,
                    episode_reward=float(random.randint(-10, 10)),
                    episode_steps=random.randint(1, 50),
                    loss_mean=random.uniform(0.1, 2.0),
                    epsilon=max(0.1, 1.0 - ep * 0.05),
                    win=bool(random.random() > 0.7),
                    terminal_state="goal" if random.random() > 0.5 else "pit",
                    learning_rate=0.001,
                    buffer_size=ep * 10,
                )
        _ok("ExperimentLogger 寫入 10 個 episodes 成功")

        # 確認 CSV 存在
        csv_path = Path("results/csv/SMOKE_TEST_DO_NOT_USE_log.csv")
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        _ok(f"CSV 已建立: {csv_path}")

    except Exception as e:
        _fail("Logger", e)


# ──────────────────────────────────────────────
# Test 4: Metrics 計算
# ──────────────────────────────────────────────

def test_metrics():
    print("\n[Test 4] Evaluation Metrics")
    try:
        from src.evaluation.metrics import load_experiment_log, compute_metrics, add_moving_averages

        csv_path = "results/csv/SMOKE_TEST_DO_NOT_USE_log.csv"
        df = load_experiment_log(csv_path)
        assert len(df) == 10, f"Expected 10 rows, got {len(df)}"
        _ok("load_experiment_log() 成功")

        df_ma = add_moving_averages(df, window=5)
        assert "reward_ma" in df_ma.columns
        assert "win_ma" in df_ma.columns
        _ok("add_moving_averages() 成功")

        m = compute_metrics(df, window=5)
        assert "win_rate" in m
        assert "average_reward" in m
        assert "total_episodes" in m
        _ok(f"compute_metrics() 成功 — win_rate={m['win_rate']:.3f}, avg_reward={m['average_reward']:.3f}")

    except Exception as e:
        _fail("Metrics", e)


# ──────────────────────────────────────────────
# Test 5: Plotting 可產出圖表
# ──────────────────────────────────────────────

def test_plotting():
    print("\n[Test 5] Plotting Pipeline")
    try:
        from src.plotting.plot_curves import plot_all_curves
        from src.evaluation.metrics import load_experiment_log

        csv_path = "results/csv/SMOKE_TEST_DO_NOT_USE_log.csv"
        output_dir = "results/figures"

        paths = plot_all_curves(
            csv_path=csv_path,
            output_dir=output_dir,
            window=5,
            smoke_test=True,  # ⚠️ 標記為 smoke test
        )

        for name, path in paths.items():
            assert Path(path).exists(), f"Figure not found: {path}"
        _ok(f"plot_all_curves() 產出 {len(paths)} 張圖表（均標記為 SMOKE TEST）")

    except Exception as e:
        _fail("Plotting", e)


# ──────────────────────────────────────────────
# Test 6: Results 目錄結構
# ──────────────────────────────────────────────

def test_results_dirs():
    print("\n[Test 6] Results Directory Structure")
    try:
        required_dirs = [
            "results/csv",
            "results/figures",
            "results/checkpoints",
        ]
        for d in required_dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
            assert Path(d).is_dir()
        _ok("results/ 三個子目錄均存在")
    except Exception as e:
        _fail("Results dirs", e)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DRL HW3 — Experiment Harness Smoke Test")
    print("  ⚠️  所有產生的圖表為 SMOKE TEST，不得用於正式報告")
    print("=" * 60)

    os.chdir(ROOT)  # 確保相對路徑正確

    test_results_dirs()
    test_config()
    test_seeding()
    test_logger()
    test_metrics()
    test_plotting()

    print("\n" + "=" * 60)
    print(f"  結果：{PASSED} PASSED / {FAILED} FAILED")
    if FAILED == 0:
        print("  🎉 Smoke Test PASSED — Harness 可用於正式實驗！")
    else:
        print("  ⚠️  Smoke Test FAILED — 請修復以上錯誤再進行正式訓練。")
    print("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
