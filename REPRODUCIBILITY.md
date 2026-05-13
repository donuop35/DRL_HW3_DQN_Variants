# REPRODUCIBILITY — DRL HW3: DQN and its Variants

> **文件版本**：v1.0  
> **建立日期**：2026-05-13  
> **作者**：Tony Lo

---

## 可重現性承諾

本專案嚴格遵循可重現性原則。**所有實驗結果必須來自真實執行，不得假造數據。**

---

## 1. 環境規格（Environment Specification）

### 作業系統
- macOS（Apple Silicon 或 Intel）

### Python 版本
- Python 3.10+（建議 3.10 或 3.11）

### 主要依賴套件（待實驗確認後更新版本號）

| 套件 | 版本 | 用途 |
|------|------|------|
| torch | ≥2.0.0 | 核心深度學習框架 |
| pytorch-lightning | ≥2.0.0 | HW3-3 訓練框架 |
| numpy | ≥1.24.0 | 數值計算 |
| matplotlib | ≥3.7.0 | 視覺化 |
| pandas | ≥2.0.0 | 數據分析 |
| gymnasium | ≥0.29.0 | 環境介面（可選） |
| pyyaml | ≥6.0 | 設定檔讀取 |
| tqdm | ≥4.65.0 | 進度顯示 |

---

## 2. 隨機種子管理（Random Seed Management）

所有實驗**必須**設定以下隨機種子：

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """固定所有隨機種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**預設種子**：`42`

---

## 3. 超參數記錄（Hyperparameter Logging）

每次實驗執行時，必須將完整超參數記錄至 `experiments/<hw>/<run_id>/config.yaml`：

```yaml
# 範例記錄格式
experiment:
  name: "hw3_1_naive_dqn"
  seed: 42
  timestamp: "2026-05-13T10:00:00"

environment:
  mode: "static"
  grid_size: [4, 4]

agent:
  type: "NaiveDQN"
  hidden_dim: 128
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995

training:
  num_episodes: 1000
  batch_size: 64
  memory_size: 10000
  target_update_freq: 10
```

---

## 4. 實驗追蹤（Experiment Tracking）

### 目錄結構

每次實驗執行產生一個 run 目錄：

```
experiments/hw3_1_static/
└── run_001/
    ├── config.yaml          ← 超參數設定
    ├── training_log.csv     ← 每 Episode 的 Reward, Loss, Epsilon
    ├── final_model.pt       ← 最終模型權重
    └── summary.md           ← 實驗摘要
```

### CSV 記錄格式

```csv
episode,reward,loss,epsilon,steps,timestamp
1,-1.0,0.523,1.000,12,2026-05-13T10:00:01
2,0.0,0.487,0.995,8,2026-05-13T10:00:02
...
```

---

## 5. 如何重現實驗（How to Reproduce）

### 完整重現步驟

```bash
# 1. 克隆 Repo
git clone https://github.com/donuop35/DRL_DQN_Variants.git
cd DRL_HW3_DQN_Variants

# 2. 建立環境
conda env create -f environment.yml
conda activate drl-hw3
# 或
pip install -r requirements.txt

# 3. 執行特定實驗
python scripts/run_hw3_1_static.py --seed 42 --config configs/hw3_1_static/default.yaml

# 4. 重現所有實驗
python scripts/run_all_experiments.py --seed 42

# 5. 生成報告圖表
python scripts/generate_report_assets.py
```

---

## 6. 不可重現風險管理（Non-reproducibility Risk Management）

| 風險來源 | 緩解措施 |
|---------|---------|
| GPU 非確定性計算 | `torch.backends.cudnn.deterministic = True` |
| 多執行緒非確定性 | 限制 DataLoader workers 為 0 |
| 環境初始化 | 固定 seed 於 env.reset() |
| Python/套件版本差異 | 記錄完整版本於 environment.yml |
| 作業系統差異 | Docker image（可選，未實作） |

---

## 7. 數據誠信聲明（Data Integrity Statement）

> 本作業所有實驗結果均來自真實執行。  
> 不得假造、篡改、或複製他人的實驗數據。  
> 所有圖表必須由 `scripts/generate_report_assets.py` 從真實 CSV 數據生成。

---

## 8. 版本歷史

| 日期 | 版本 | 說明 |
|------|------|------|
| 2026-05-13 | v1.0 | Phase 1 初版建立 |
