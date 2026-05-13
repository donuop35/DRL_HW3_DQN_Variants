# ASSIGNMENT REQUIREMENTS — HW3: DQN and its Variants
# 完整逐項解析（Phase 2 正式版）

> **文件版本**：v2.0（Phase 2 正式版）
> **建立日期**：2026-05-13
> **作者**：Tony Lo（via Antigravity）
> **語言原則**：報告以繁體中文撰寫，英文專業名詞可保留
> **警告**：本文件是對教授需求的逐字解析，不得與實際作業描述衝突

---

## 一、作業基本資訊

| 項目 | 內容 |
|------|------|
| 課程 | 深度強化學習（Deep Reinforcement Learning） |
| 作業編號 | Homework 3 |
| 類型 | 個人作業 |
| 截止日 | 2026-05-13（繳交截止） |
| 成績佔比 | 10% |
| 評分方式 | 直接打分數（100 分制） |
| 允許遲交 | **否** |

---

## 二、Setup & Reference（必讀基礎）

### 教授指定參考來源

1. **DRL in Action GitHub Repo**（英文版）：
   `https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master`
2. **教授提供的 updated starter code**（必須作為 baseline，不得自行從頭重寫）：
   `第3章程式_ALL_IN_ONE.ipynb`

### Starter Code 解析（已逐字閱讀）

教授的 starter code 共 51 cells，包含：

| 程式編號 | 內容 | 對應功能 |
|---------|------|---------|
| 程式 3.1 | GridWorld 環境初始化 | `Gridworld(size=4, mode='static')` |
| 程式 3.2 | 基礎 Q-Network 架構 | MLP: 64→150→100→4 |
| 程式 3.3 | 基礎 DQN 訓練迴圈（static / random / player） | 無 Replay Buffer |
| 程式 3.4 | `test_model()` 評估函式 | 測試模型性能 |
| 程式 3.5 | **Experience Replay Buffer DQN** | `deque(maxlen=1000)`, batch=200 |
| 程式 3.6 | 測試 Experience Replay 模型 | 勝率計算 |
| 程式 3.7 | **Target Network 架構** | `copy.deepcopy(model)`, sync_freq=500 |
| 程式 3.8 | **Experience Replay + Target Network 完整 DQN** | 最完整 starter DQN |
| 程式 3.5 改良版 | **避免撞牆機制** | `validateMove()`, 撞牆 reward=-5 |

### Starter Code 超參數（教授預設值，必須記錄）

```python
L1 = 64      # 輸入層（4x4x4 grid flattened）
L2 = 150     # 第一隱藏層
L3 = 100     # 第二隱藏層
L4 = 4       # 輸出層（4個動作：上下左右）

learning_rate = 1e-3
optimizer = Adam
loss_fn = MSELoss
gamma = 0.9
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = linear（1/epochs）

# Experience Replay
mem_size = 1000
batch_size = 200
max_moves = 50

# Target Network
sync_freq = 500   # 每500步同步一次參數
```

### GridWorld 三種模式（完整定義）

| 模式名稱 | `mode=` | Player 位置 | Goal | Pit | Wall | 難度 |
|---------|---------|------------|------|-----|------|------|
| **static** | `'static'` | 固定 (0,3) | (1,0) | (0,1) | (1,1) | ★☆☆ |
| **player** | `'player'` | 隨機 | (1,0) | (0,1) | (1,1) | ★★☆ |
| **random** | `'random'` | 隨機 | 隨機 | 隨機 | 隨機 | ★★★ |

### GridWorld 獎勵函式

```python
Goal → +10
Pit  → -10
Wall → -1（步驟懲罰）或 -5（if 撞牆 reward shaping）
其他 → -1（每步懲罰，促使快速找到目標）
```

### 動作空間

```python
action_set = {
    0: 'u',  # 向上
    1: 'd',  # 向下
    2: 'l',  # 向左
    3: 'r'   # 向右
}
```

---

## 三、HW3-1：Naive DQN for Static Mode（30 分）

### 3.1 教授原文需求（逐字）

> - ✅ Run the provided code naive or Experience buffer reply
> - ✅ Chat with ChatGPT about the code to clarify your understanding
> - ✅ Submit a short understanding report

### 3.2 必要交付物（Deliverables）

| 交付物 | 必/選 | 說明 |
|-------|-------|------|
| 可執行的 DQN 程式碼（Static mode） | **必** | 基於 starter code 的 Naive DQN |
| Experience Replay Buffer 程式碼 | **必** | 對應程式 3.5 |
| **`understanding_report.md`** | **必（不得遺漏）** | 簡短但有深度的理解報告 |
| 訓練 Loss 曲線圖 | **強烈建議** | 隱性期待 |

### 3.3 隱性期待（Hidden Expectations）

1. **understanding_report.md 不是流水帳**：
   - 必須展示真實理解，而非複製貼上 ChatGPT 的回答
   - 需解釋：為什麼要用 Experience Replay？為什麼要用 Target Network？
   - 需包含：自己觀察到的訓練現象（loss 下降趨勢、收斂特性）

2. **Loss 曲線必須真實**：
   - 不能只說「模型學會了」，要有圖為證

3. **Static Mode 的意義**：
   - 教授用 Static 作為最簡單的驗證場景
   - 預期 DQN 在固定環境中應該能學會

### 3.4 必要程式檔案

| 檔案 | 說明 |
|------|------|
| `src/agents/naive_dqn.py` | Naive DQN Agent |
| `src/buffers/replay_buffer.py` | Experience Replay Buffer |
| `src/envs/gridworld.py` | GridWorld 封裝（Static mode） |
| `src/training/trainer.py` | 訓練迴圈 |
| `configs/hw3_1_static/default.yaml` | 超參數設定 |
| `scripts/run_hw3_1_static.py` | 執行腳本 |
| `report/understanding_report.md` | **理解報告（最高優先）** |

### 3.5 評分重點

1. 程式能執行（30 分中的基礎分）
2. understanding_report.md 的深度（決定是否有高分）

---

## 四、HW3-2：Enhanced DQN Variants for Player Mode（40 分）

### 4.1 教授原文需求（逐字）

> Implement and compare the following:
> - Double DQN
> - Dueling DQN
> - Focus on how they improve upon the basic DQN approach

### 4.2 必要交付物（Deliverables）

| 交付物 | 必/選 | 說明 |
|-------|-------|------|
| Double DQN 實作 | **必** | Player mode |
| Dueling DQN 實作 | **必** | Player mode |
| 與 Basic DQN 的比較圖 | **必** | 至少訓練曲線比較 |
| 改進原理說明 | **必** | 在報告中解釋 |

### 4.3 各演算法核心理解（必須正確實作）

#### Double DQN（解決 Overestimation Bias）

問題：標準 DQN 使用同一個網路選動作和評估 Q 值，導致 Q 值高估。

解法：
- **Online Network** 選動作（argmax）
- **Target Network** 評估 Q 值

```python
# Standard DQN Target
Y = r + gamma * max(Q_target(s'))

# Double DQN Target
a* = argmax(Q_online(s'))           # online network 選動作
Y = r + gamma * Q_target(s', a*)    # target network 評估
```

#### Dueling DQN（Value-Advantage Decomposition）

問題：標準 DQN 對「跟動作無關的狀態」學習效率低。

解法：分離網路為兩個分支：
- **Value Stream** V(s)：狀態的整體價值
- **Advantage Stream** A(s,a)：相對優勢

```python
Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
```

### 4.4 必要程式檔案

| 檔案 | 說明 |
|------|------|
| `src/agents/double_dqn.py` | Double DQN Agent |
| `src/agents/dueling_dqn.py` | Dueling DQN Agent |
| `src/models/dueling_net.py` | Dueling Network 架構 |
| `configs/hw3_2_player/double_dqn.yaml` | Double DQN 設定 |
| `configs/hw3_2_player/dueling_dqn.yaml` | Dueling DQN 設定 |
| `scripts/run_hw3_2_player.py` | 執行腳本 |

### 4.5 隱性期待（Hidden Expectations）

1. **比較圖必須包含 3 條曲線**：Naive DQN vs Double DQN vs Dueling DQN
2. **必須解釋「如何改善」**，不只是說「結果更好」
3. 實驗要在 **Player mode** 下執行（不是 static）

---

## 五、HW3-3：Enhanced DQN for Random Mode WITH Training Tips（30 分）

### 5.1 教授原文需求（逐字）

> - Convert the DQN model from PyTorch to either:
>   - Keras, or
>   - PyTorch Lightning
> - Bonus points for integrating training techniques to stabilize/improve learning
>   (e.g., gradient clipping, learning rate scheduling, etc.)

### 5.2 必要交付物（Deliverables）

| 交付物 | 必/選 | 說明 |
|-------|-------|------|
| PyTorch Lightning 實作 | **必** | 轉換 DQN 框架 |
| Random Mode 訓練 | **必** | 最難環境 |
| Gradient Clipping | **加分** | Training Tip |
| LR Scheduling | **加分** | Training Tip |
| Reward/Loss 視覺化 | **強烈建議** | |
| PER（Prioritized Experience Replay）| **加分** | 我們的策略 |
| Epsilon Decay Tuning | **加分** | 我們的策略 |

### 5.3 隱性期待（Hidden Expectations）

1. **框架轉換必須正確**：Lightning `LightningModule` 要正確覆寫 `training_step()`, `configure_optimizers()`
2. **Training Tips 要有消融實驗**：展示 with/without 的差異
3. **Random Mode 最難**，期待對訓練穩定性的深度分析

### 5.4 固定實驗設計（E1～E3 正式實驗）

| 實驗 ID | 名稱 | 設定 |
|---------|------|------|
| **E1** | Random DQN Baseline | 基礎 DQN，Random mode，無 Training Tips |
| **E2** | Stabilized DQN | + Gradient Clipping + LR Scheduling |
| **E3** | PER-DQN + Stabilization | + Prioritized Experience Replay + Epsilon Tuning |

> ⚠️ **防呆規則**：E1～E3 是正式 HW3-3 主線，不得被任何 Bonus 實驗取代或破壞。

### 5.5 必要程式檔案

| 檔案 | 說明 |
|------|------|
| `src/agents/lightning_dqn.py` | PyTorch Lightning DQN Module |
| `src/buffers/per_buffer.py` | Prioritized Experience Replay Buffer |
| `src/training/lightning_trainer.py` | Lightning Trainer 封裝 |
| `configs/hw3_3_random/e1_baseline.yaml` | E1 設定 |
| `configs/hw3_3_random/e2_stabilized.yaml` | E2 設定 |
| `configs/hw3_3_random/e3_per.yaml` | E3 設定 |
| `scripts/run_hw3_3_random.py` | 執行腳本 |

---

## 六、Rainbow DQN Bonus Pipeline（加分）

### 6.1 定位說明

> ⚠️ **Bonus 定位**：E4 是獨立加分實驗，**不得修改、取代、或破壞 E1～E3**

### 6.2 Rainbow 元素

| 元素 | 說明 | 來源 |
|------|------|------|
| Double DQN | 解決過估 | HW3-2 實作（可複用） |
| Dueling Network | 分離 V + A | HW3-2 實作（可複用） |
| PER | 優先採樣 | HW3-3 E3 實作（可複用） |
| Multi-step Learning | N-step Returns | 新增 |
| Noisy Networks | 參數化探索 | 可選 |
| Distributional RL | C51 / QR-DQN | 可選 |

### 6.3 固定實驗 ID

| 實驗 ID | 名稱 | 說明 |
|---------|------|------|
| **E4** | Advanced Rainbow DQN Bonus Pipeline | 整合 Double + Dueling + PER + Multi-step |

---

## 七、understanding_report.md 詳細規格

> ⚠️ **本節最高優先，不得遺漏**

### 位置

`report/understanding_report.md`

### 必要內容章節

1. **DQN 原理理解**（用自己的話解釋，非複製貼上）
   - 為什麼用 Neural Network 近似 Q 函數？
   - Experience Replay 的作用機制
   - Target Network 的穩定化原理

2. **GridWorld 環境分析**
   - 三種 mode 的差異
   - 獎勵設計的含義
   - 狀態空間表示（4x4x4 flattened to 64）

3. **Starter Code 逐段分析**
   - 程式 3.2：網路架構
   - 程式 3.5：Experience Replay 實作
   - 程式 3.8：Target Network 整合

4. **實驗結果**（真實跑出來後填入）
   - Loss 曲線圖（嵌入圖表，非只放在資料夾）
   - 訓練過程觀察
   - 收斂分析

5. **個人見解與討論**
   - 觀察到的問題
   - 對 DQN 侷限性的理解

### 語言要求

- **繁體中文**撰寫
- 英文專業名詞可保留（如 Experience Replay, Target Network, Q-value 等）
- 不可只是翻譯教科書，要有自己的觀察與分析

---

## 八、全局防呆規則

1. **不得假造實驗結果**：所有圖表必須從真實 CSV 數據生成
2. **報告必須嵌入圖表**：不可只說「見 results/figures/xxx.png」
3. **E1～E3 不得被 E4 破壞**：Bonus 是加分項，不是替代品
4. **所有實驗使用統一 logging schema**
5. **understanding_report.md 是正式交付物**，遺漏直接失分
6. **所有超參數必須記錄在 config yaml**，不寫死在程式碼

---

## 九、交付物完整清單

### 程式碼

- [ ] `src/envs/gridworld.py`
- [ ] `src/buffers/replay_buffer.py`
- [ ] `src/buffers/per_buffer.py`（HW3-3 / Bonus）
- [ ] `src/models/dqn_net.py`
- [ ] `src/models/dueling_net.py`（HW3-2）
- [ ] `src/agents/naive_dqn.py`（HW3-1）
- [ ] `src/agents/double_dqn.py`（HW3-2）
- [ ] `src/agents/dueling_dqn.py`（HW3-2）
- [ ] `src/agents/lightning_dqn.py`（HW3-3）
- [ ] `src/agents/rainbow_dqn.py`（Bonus）
- [ ] `src/training/trainer.py`
- [ ] `src/training/lightning_trainer.py`
- [ ] `src/plotting/plot_utils.py`
- [ ] `src/utils/seed.py`
- [ ] `src/utils/config.py`
- [ ] `src/utils/logger.py`

### 設定檔

- [ ] `configs/hw3_1_static/default.yaml`
- [ ] `configs/hw3_2_player/double_dqn.yaml`
- [ ] `configs/hw3_2_player/dueling_dqn.yaml`
- [ ] `configs/hw3_3_random/e1_baseline.yaml`
- [ ] `configs/hw3_3_random/e2_stabilized.yaml`
- [ ] `configs/hw3_3_random/e3_per.yaml`
- [ ] `configs/hw3_3_random/e4_rainbow.yaml`（Bonus）

### 報告

- [ ] `report/understanding_report.md`（**最高優先**）
- [ ] `report/HW3_DQN_Variants_研究型實驗報告.md`

### 結果

- [ ] `results/csv/`（各實驗訓練 log）
- [ ] `results/figures/`（各實驗圖表）
- [ ] `results/checkpoints/`（最終模型權重）

### GitHub

- [ ] GitHub Repo 公開可存取
- [ ] README.md 說明如何重現實驗
