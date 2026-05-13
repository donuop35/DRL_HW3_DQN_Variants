# DES-001: Project Architecture Design

**Design ID**: DES-001  
**建立日期**: 2026-05-13  
**作者**: Tony Lo (via Antigravity)  
**對應 Change**: CHG-001  
**狀態**: Approved ✅

---

## 架構概述

本專案採用**模組化分層架構**，核心設計原則：

1. **關注點分離（Separation of Concerns）**：每個模組只負責一件事
2. **設定檔驅動（Config-driven）**：超參數集中管理，不寫死
3. **可擴展性（Extensibility）**：新增 DQN 變體不需修改現有程式碼
4. **實驗隔離（Experiment Isolation）**：每次執行有獨立的結果目錄

---

## 核心模組設計

### `src/envs/` — 環境封裝層

```
envs/
├── __init__.py
├── gridworld.py        # GridWorld 環境核心
└── wrappers.py         # 環境包裝器（觀察預處理等）
```

**責任**：提供統一的環境介面，封裝 Static / Player / Random 三種模式。

---

### `src/buffers/` — 記憶體管理層

```
buffers/
├── __init__.py
├── replay_buffer.py    # 基礎 Experience Replay Buffer
└── per_buffer.py       # Prioritized Experience Replay（Bonus 用）
```

**責任**：管理訓練樣本的存儲與採樣。

---

### `src/models/` — 神經網路架構層

```
models/
├── __init__.py
├── dqn_net.py          # 基礎 Q-Network
└── dueling_net.py      # Dueling Network 架構
```

**責任**：定義各種 Q-Network 的神經網路結構。

---

### `src/agents/` — 決策代理層

```
agents/
├── __init__.py
├── base_agent.py       # 抽象基類
├── naive_dqn.py        # Naive DQN（HW3-1）
├── double_dqn.py       # Double DQN（HW3-2）
├── dueling_dqn.py      # Dueling DQN（HW3-2）
└── rainbow_dqn.py      # Rainbow DQN（Bonus）
```

**責任**：實作各種 DQN 變體的決策邏輯（選動作、更新 Q 值）。

---

### `src/training/` — 訓練管理層

```
training/
├── __init__.py
├── trainer.py          # 通用訓練迴圈
└── lightning_trainer.py # PyTorch Lightning Trainer（HW3-3）
```

**責任**：管理訓練迴圈、記錄指標、儲存 checkpoint。

---

### `src/evaluation/` — 評估層

```
evaluation/
├── __init__.py
└── evaluator.py        # 模型評估工具
```

**責任**：評估訓練好的模型性能。

---

### `src/plotting/` — 視覺化層

```
plotting/
├── __init__.py
└── plot_utils.py       # 訓練曲線、比較圖生成
```

**責任**：生成報告所需的所有圖表。

---

### `src/utils/` — 通用工具層

```
utils/
├── __init__.py
├── seed.py             # 隨機種子管理
├── config.py           # 設定檔讀取
└── logger.py           # 日誌記錄
```

**責任**：提供跨模組使用的共用工具。

---

## 設定檔設計

### 設定檔格式（YAML）

```yaml
# configs/hw3_1_static/default.yaml
experiment:
  name: "naive_dqn_static"
  seed: 42

environment:
  mode: "static"
  
agent:
  type: "NaiveDQN"
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  hidden_dim: 128

training:
  num_episodes: 1000
  batch_size: 64
  memory_size: 10000
  target_update_freq: 10
```

---

## 資料流（Data Flow）

```
Config File → Agent.__init__()
                    ↓
           Environment.reset()
                    ↓
           Agent.select_action(state)
                    ↓
           Environment.step(action)
                    ↓
           Buffer.push(transition)
                    ↓
           Buffer.sample(batch)
                    ↓
           Agent.update(batch)
                    ↓
           Logger.record(metrics)
                    ↓
           results/csv/training_log.csv
```

---

## 擴展性設計

新增 DQN 變體只需：

1. 在 `src/models/` 新增網路架構（如需要）
2. 在 `src/agents/` 繼承 `BaseAgent`，覆寫 `update()` 方法
3. 在 `configs/` 新增對應設定檔
4. 在 `scripts/` 新增執行腳本（可選，也可用參數控制）
