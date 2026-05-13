# 🧠 DRL HW3: DQN and its Variants

> **課程**：深度強化學習（Deep Reinforcement Learning）
> **作業**：Homework 3 — DQN and its Variants
> **截止日**：2026-05-13
> **作者**：Tony Lo（中興大學）
> **評分佔比**：10%

---

## 📋 作業概述

本 Repo 為研究所深度強化學習課程 HW3 的完整作業繳交內容，涵蓋以下四個子任務：

| 子任務 | 名稱 | 模式 | 分數 |
|--------|------|------|------|
| HW3-1 | Naive DQN (Static Mode) | Static | 30% |
| HW3-2 | Enhanced DQN Variants (Double DQN + Dueling DQN) | Player | 40% |
| HW3-3 | Enhanced DQN for Random Mode + Training Tips | Random | 30% |
| Bonus | Rainbow DQN Integration | — | Bonus |

所有實驗基於 **GridWorld 環境**，參考 [DRL in Action GitHub Repo](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)。

---

## 🗂️ 專案結構

```
DRL_DQN_Variants/
├── README.md                        ← 本文件
├── PROJECT_CHARTER.md               ← 作業目標與子任務定位
├── ASSIGNMENT_REQUIREMENTS.md       ← 評分標準原文整理
├── EXPERIMENT_STORYLINE.md          ← 實驗敘事規劃
├── REPRODUCIBILITY.md               ← 可重現性說明
├── requirements.txt                 ← Python 套件依賴
├── pyproject.toml                   ← 專案 metadata
├── environment.yml                  ← Conda 環境配置
├── openspec/                        ← OpenSpec SDD workflow 文件
│   ├── changes/                     ← Change Proposals (CHG-XXX)
│   ├── designs/                     ← Technical Design Documents
│   └── tasks/                       ← Task 追蹤
├── data/raw/                        ← 原始 starter code（保留不修改）
├── notebooks/starter/               ← 教授提供的 Starter Notebook
├── src/                             ← 核心程式碼
│   ├── envs/                        ← GridWorld 環境封裝
│   ├── buffers/                     ← Experience Replay Buffer
│   ├── models/                      ← Q-Network 模型架構
│   ├── agents/                      ← DQN / DoubleDQN / DuelingDQN / Rainbow
│   ├── training/                    ← 訓練迴圈與 Trainer
│   ├── evaluation/                  ← 評估與測試
│   ├── plotting/                    ← 視覺化工具
│   └── utils/                       ← 共用工具函式
├── configs/                         ← 各子任務實驗設定檔
│   ├── hw3_1_static/
│   ├── hw3_2_player/
│   └── hw3_3_random/
├── experiments/                     ← 實驗執行紀錄
├── results/                         ← 實驗結果
│   ├── csv/                         ← 訓練曲線數據
│   ├── figures/                     ← 圖表輸出
│   └── checkpoints/                 ← 模型權重
├── report/                          ← 研究型實驗報告
│   ├── understanding_report.md      ← HW3-1 理解報告
│   ├── HW3_DQN_Variants_研究型實驗報告.md
│   └── assets/                      ← 報告圖片資源
└── scripts/                         ← 執行腳本
    ├── run_hw3_1_static.py
    ├── run_hw3_2_player.py
    ├── run_hw3_3_random.py
    ├── run_all_experiments.py
    └── generate_report_assets.py
```

---

## 🚀 快速開始

### 環境安裝

```bash
# 建議使用 conda
conda env create -f environment.yml
conda activate drl-hw3

# 或使用 pip
pip install -r requirements.txt
```

### Step 0：Smoke Test（必須先通過）

```bash
# 驗證 harness 可用（config / logger / metrics / plotting）
python scripts/smoke_test.py

# 預期輸出：11 PASSED / 0 FAILED ✅
# ⚠️ Smoke test 產生的圖表標記為 SMOKE TEST，不得用於正式報告
```

### Step 1：執行 HW3-1（Static Mode）

```bash
# 執行 Naive DQN 訓練（5000 episodes）
python scripts/run_hw3_1_static.py \
  --config configs/hw3_1_static/default.yaml

# 結果輸出至：
# - results/csv/hw3_1_static_naive_dqn_log.csv
# - results/figures/hw3_1_static_naive_dqn_*.png
# - results/checkpoints/hw3_1_static_naive_dqn/
```

### Step 2：執行 HW3-2（Player Mode）

```bash
# 執行三個演算法比較
python scripts/run_hw3_2_player.py

# 個別執行：
python scripts/run_hw3_2_player.py --config configs/hw3_2_player/naive_dqn.yaml
python scripts/run_hw3_2_player.py --config configs/hw3_2_player/double_dqn.yaml
python scripts/run_hw3_2_player.py --config configs/hw3_2_player/dueling_dqn.yaml
```

### Step 3：執行 HW3-3（Random Mode，E1→E3）

```bash
# E1：Baseline（無 Training Tips）
python scripts/run_hw3_3_random.py --config configs/hw3_3_random/e1_baseline.yaml

# E2：Stabilized（Gradient Clipping + LR Scheduling）
python scripts/run_hw3_3_random.py --config configs/hw3_3_random/e2_stabilized.yaml

# E3：PER + Stabilization
python scripts/run_hw3_3_random.py --config configs/hw3_3_random/e3_per.yaml

# E4：Rainbow Bonus（不影響 E1-E3）
python scripts/run_hw3_3_random.py --config configs/hw3_3_random/e4_rainbow.yaml
```

### Step 4：生成報告圖表

```bash
# 從真實 CSV 生成所有正式圖表
python scripts/generate_report_assets.py

# 生成所有實驗
python scripts/run_all_experiments.py --seed 42
```

> ⚠️ **重要**：所有執行步驟請參考 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。  
> 所有圖表必須由真實 CSV 生成，不得手動修改數據。

> 📝 **注意**：此區塊將在正式訓練腳本完成後更新（Phase 4）。

---

## 📊 實驗環境規格

| 環境模式 | 說明 | Player 位置 | 其他物件位置 |
|----------|------|------------|------------|
| **Static** | 完全固定，所有物件位置固定 | 固定 (0,3) | Goal→(1,0), Pit→(0,1), Wall→(1,1) |
| **Player** | 只有 Player 隨機，其他固定 | 隨機 | 固定（同上） |
| **Random** | 所有物件（Player, Goal, Pit, Wall）完全隨機 | 隨機 | 隨機 |

---

## 📝 報告

- [`report/understanding_report.md`](report/understanding_report.md) — HW3-1 理解報告（繳交必要）
- [`report/HW3_DQN_Variants_研究型實驗報告.md`](report/HW3_DQN_Variants_研究型實驗報告.md) — 完整研究型實驗報告

---

## 🔬 OpenSpec Workflow

本專案採用 **OpenSpec Spec-Driven Development (SDD)** 方法論：

```
openspec/changes/CHG-001-repo-bootstrap.md      ← 本階段 change
openspec/designs/DES-001-project-architecture.md
openspec/tasks/TASK-001-phase1-bootstrap.md
```

---

## 📚 參考資料

- [DRL in Action Repo](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)
- [PyTorch 官方文件](https://pytorch.org/docs/)
- [PyTorch Lightning 官方文件](https://lightning.ai/docs/)
- [Rainbow DQN Paper](https://arxiv.org/abs/1710.02298)

---

*所有報告以**繁體中文**撰寫，英文專業名詞可保留。*
