# 🧠 DRL HW3: DQN and its Variants

> **課程**：深度強化學習（Deep Reinforcement Learning）  
> **作業**：Homework 3 — DQN and its Variants  
> **作者**：Tony Lo（國立中興大學）  
> **GitHub**：https://github.com/donuop35/DRL_HW3_DQN_Variants  
> **最後更新**：2026-05-13（Phase 9 Final Submission）

---

## 📋 作業概述

本 Repo 為研究所深度強化學習課程 HW3 的完整作業繳交內容，涵蓋四個子任務：

| 子任務 | 名稱 | 模式 | 分數 |
|--------|------|------|------|
| HW3-1 | Naive / Basic DQN | Static | 30% |
| HW3-2 | Double DQN + Dueling DQN | Player | 40% |
| HW3-3 | Enhanced DQN + PyTorch Lightning + PER | Random | 30% |
| Bonus⚡ | Rainbow DQN（6 組件）| Random | 加分 |

所有實驗基於 **GridWorld 4×4 環境**（Lapan, 2020 / DRL-in-Action），使用統一 seed=42，5000 episodes。

---

## 🚀 快速開始

### 環境需求
- **Python**: 3.9+
- **OS**: macOS / Linux（Windows 未測試）

### 安裝依賴

```bash
git clone https://github.com/donuop35/DRL_HW3_DQN_Variants.git
cd DRL_HW3_DQN_Variants
pip install -r requirements.txt
```

### 一鍵執行所有實驗

```bash
# HW3-1（Static Mode Basic DQN）
python scripts/run_hw3_1_static.py

# HW3-2（Player Mode: Basic / Double / Dueling DQN）
python scripts/run_hw3_2_player.py

# HW3-3 E1/E2/E3（Random Mode，正式主線）
python scripts/run_hw3_3_random.py

# E4 Rainbow Bonus（獨立加分，不影響 E1-E3）
python scripts/run_hw3_3_rainbow_bonus.py
```

---

## 📁 專案結構

```
DRL_HW3_DQN_Variants/
├── README.md                        ← 本文件
├── REPRODUCIBILITY.md               ← 完整重現指南
├── FINAL_SUBMISSION_CHECKLIST.md    ← 繳交前驗收清單
├── FIGURE_MANIFEST.md               ← 圖表追蹤清單（53 張）
├── RESULT_INTERPRETATION_NOTES.md  ← 數據品管與解讀
├── REQUIREMENTS_TRACEABILITY_MATRIX.md ← 需求追蹤矩陣
├── EXPERIMENT_PROTOCOL.md           ← 實驗規則與數據誠信規定
├── ASSIGNMENT_REQUIREMENTS.md       ← 教授作業需求解析
├── EXPERIMENT_STORYLINE.md          ← 實驗故事線設計
│
├── requirements.txt                 ← Python 套件清單
│
├── configs/                         ← 所有超參數 YAML（統一管理）
│   ├── hw3_1_static/
│   │   └── basic_dqn_static.yaml
│   ├── hw3_2_player/
│   │   ├── basic_dqn_player.yaml
│   │   ├── double_dqn_player.yaml
│   │   └── dueling_dqn_player.yaml
│   └── hw3_3_random/
│       ├── e1_random_dqn_baseline.yaml
│       ├── e2_stabilized_dqn.yaml
│       ├── e3_per_dqn_stabilized.yaml
│       └── e4_rainbow_dqn_bonus.yaml  ← Bonus
│
├── scripts/                          ← 執行腳本
│   ├── run_hw3_1_static.py
│   ├── run_hw3_2_player.py
│   ├── run_hw3_3_random.py           ← E1/E2/E3 正式主線
│   ├── run_hw3_3_rainbow_bonus.py    ← E4 Bonus
│   └── smoke_test.py
│
├── src/                              ← 所有模組
│   ├── envs/          ← GridWorld 環境封裝
│   ├── models/        ← QNetwork, DuelingNet, C51DuelingNetwork, NoisyLinear
│   ├── buffers/       ← ReplayBuffer, PER (SumTree), NStepPERBuffer
│   ├── agents/        ← DQNAgent
│   ├── training/      ← train_dqn.py, LightningDQNModule, LightningRainbowModule
│   ├── evaluation/    ← metrics.py
│   ├── plotting/      ← plot_curves.py, plot_comparison.py
│   └── utils/         ← config.py, logger.py, seeding.py
│
├── report/
│   ├── understanding_report.md      ← ⭐ 教授要求的理解報告（1020 行）
│   └── HW3_DQN_Variants_研究型實驗報告.md ← ⭐ 主研究型實驗報告
│
├── results/
│   ├── csv/           ← 8 個實驗 CSV + 5 個 summary tables
│   ├── figures/       ← 53 張圖表（個別曲線 + 比較圖）
│   └── checkpoints/   ← 8 個訓練好的模型（.pt）
│
└── openspec/          ← OpenSpec SDD 管理（CHG-001～CHG-007）
```

---

## 📊 實驗結果摘要

| Label | 模式 | Algorithm | 全體 WR | Final Eval WR |
|-------|------|-----------|---------|---------------|
| S-static | static | Basic DQN | 75.5% | **100.0%** |
| P1 | player | Basic DQN | 86.1% | **100.0%** |
| P2 | player | Double DQN | 86.2% | **100.0%** |
| P3 | player | Dueling DQN | 86.2% | **100.0%** |
| E1 | random | E1 Baseline | 79.6% | 91.5% |
| E2 | random | E2 Stabilized | 82.3% | 88.5% |
| **E3** | **random** | **E3 PER+Stab（主方法）** | **85.2%** | **90.0%** |
| E4⚡ | random | Rainbow Bonus | 33.0% | 40.0% |

> 所有數值來自真實訓練（seed=42，5000 episodes），完整數據見 `results/csv/final_all_experiments_summary.csv`

---

## 🔬 執行詳細說明

### HW3-1 Static Mode

```bash
python scripts/run_hw3_1_static.py
# 輸出：results/csv/hw3_1_static_basic_dqn_log.csv
#       results/figures/hw3_1_static_basic_dqn_*.png（5 張）
#       results/checkpoints/hw3_1_static_basic_dqn/final_model.pt
```

**預期結果**：Final eval win rate ≈ 100%，訓練時間 ~230s

### HW3-2 Player Mode

```bash
python scripts/run_hw3_2_player.py
# 輸出：results/csv/hw3_2_player_{basic,double,dueling}_dqn_log.csv
#       results/figures/hw3_2_player_*_comparison.png（5 張比較圖）
```

**預期結果**：三方 final win rate 均 100%，Double DQN 後期最穩定

### HW3-3 Random Mode E1/E2/E3（正式主線）

```bash
python scripts/run_hw3_3_random.py           # 三組全跑（~15min）
python scripts/run_hw3_3_random.py --exp e1  # 僅 E1
python scripts/run_hw3_3_random.py --exp e3  # 僅 E3（正式主方法）
```

**預期結果**：E3 全體 win rate 最高（85.2%）；生成 7 張比較圖

### E4 Rainbow Bonus（獨立加分）

```bash
python scripts/run_hw3_3_rainbow_bonus.py
# 輸出：results/csv/hw3_3_random_e4_rainbow_bonus_log.csv
#       results/figures/hw3_3_random_e4_*.png（4 張個別 + 5 張 E1-E4 比較）
# 注意：E4 不影響 E1-E3 結果
```

**預期結果**：Final win rate ~40%（C51 在小環境 5000ep 收斂不足，屬正常現象）

### 重建所有圖表

```bash
python -c "
import sys; sys.path.insert(0,'.')
from src.utils.config import load_config
from src.training.train_dqn import generate_figures
cfg = load_config('configs/hw3_1_static/basic_dqn_static.yaml')
generate_figures(cfg)
"
```

---

## 📖 報告文件

| 文件 | 說明 |
|------|------|
| `report/understanding_report.md` | ⭐ **教授要求的理解報告**（1020 行，§1-32）|
| `report/HW3_DQN_Variants_研究型實驗報告.md` | ⭐ 主研究型實驗報告（含圖嵌入、數據分析）|
| `RESULT_INTERPRETATION_NOTES.md` | 數據品管記錄與結果解讀 |
| `FIGURE_MANIFEST.md` | 53 張圖表完整追蹤清單 |

---

## 🧪 重現性

完整重現指南見 `REPRODUCIBILITY.md`。

| 項目 | 設定 |
|------|------|
| Python | 3.9 |
| PyTorch | 2.0+ |
| PyTorch Lightning | 2.6.0 |
| Seed | 42（所有實驗） |
| 實驗 ID 鎖定 | ✅（見 `EXPERIMENT_PROTOCOL.md`）|

---

## 📚 參考文獻

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518.
2. van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
3. Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML*.
4. Schaul, T., et al. (2016). Prioritized Experience Replay. *ICLR*.
5. Bellemare, M. G., et al. (2017). A Distributional Perspective on Reinforcement Learning. *ICML*.
6. Fortunato, M., et al. (2017). Noisy Networks for Exploration. *ICLR*.
7. Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI*.
8. 教授提供的 Starter Code：`第3章程式_ALL_IN_ONE.ipynb`

---

*所有報告以**繁體中文**撰寫，英文專有名詞保留。*  
*實驗數據均來自真實訓練，不含捏造。*  
*GitHub Repo：https://github.com/donuop35/DRL_HW3_DQN_Variants*
