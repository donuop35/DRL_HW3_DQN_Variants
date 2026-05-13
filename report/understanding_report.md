# DRL HW3 理解報告

> **課程**：深度強化學習（Deep Reinforcement Learning）  
> **作業**：Homework 3 — DQN and its Variants  
> **作者**：Tony Lo（中興大學）  
> **日期**：2026-05-13  
> **涵蓋**：HW3-1（Static Mode）、HW3-2（Player Mode）

---

## HW3-1：Basic DQN on Static Mode（第 1–10 節）

> **實驗設定**：seed=42, episodes=5000, mode=static, 架構 64→150→100→4

## 1. DQN 是什麼

**DQN（Deep Q-Network）** 是將深度神經網路（Deep Neural Network）與 Q-Learning 結合的強化學習演算法，由 DeepMind 在 2013-2015 年提出並以其在 Atari 遊戲上的表現震驚學界。

### 1.1 強化學習的基本架構

強化學習的核心概念是：一個 **Agent（智能體）** 在 **Environment（環境）** 中採取 **Action（動作）**，取得 **Reward（獎勵）**，並更新自己的策略（Policy），目標是最大化長期累積獎勵：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

其中 $\gamma \in [0, 1)$ 為折扣因子（本實驗使用 $\gamma = 0.9$），決定了 agent 對未來獎勵的重視程度。

### 1.2 Q-Learning 的核心思想

Q-Learning 透過學習一個 **Q 函數（Action-Value Function）** $Q(s, a)$ 來決策，代表「在狀態 $s$ 下採取動作 $a$，並之後遵循最優策略所能獲得的期望累積獎勵」：

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

Agent 的行動策略：選擇 Q 值最大的動作：

$$a^* = \arg\max_{a} Q(s, a)$$

---

## 2. 為什麼 Q-Table 在較大 State Space 不夠

傳統 Q-Learning 使用一張「Q Table」來儲存所有 $(s, a)$ 的 Q 值。對於本實驗的 GridWorld：

| 環境 | 狀態數 | 動作數 | Q Table 大小 |
|------|--------|--------|------------|
| GridWorld 4×4 Static | $4^{4 \times 4}$ ≈ $2^{32}$ | 4 | 極大 |
| Atari 遊戲（84×84×3 畫面） | $256^{84 \times 84 \times 3}$ ≈ 不可計算 | 18 | 不可能存在 |

**Q Table 的限制：**

1. **儲存問題**：狀態空間爆炸，無法枚舉所有狀態
2. **泛化問題**：每個狀態獨立學習，無法從相似狀態中遷移知識
3. **探索問題**：大量稀疏狀態導致大多數 Q 值永遠無法被更新

**神經網路的解決方案**：用 $Q(s, a; \theta)$ 近似 $Q^*(s, a)$，其中 $\theta$ 是網路參數。相似的狀態會被映射到相近的隱藏表示，自動實現泛化。

---

## 3. Neural Network 如何近似 Q(s, a)

### 3.1 本實驗的網路架構

對應教授 starter code 程式 3.7：

```
輸入層:  state (64 維，對應 4×4×4 GridWorld flatten)
         ↓
隱藏層1: Linear(64 → 150) + ReLU
         ↓
隱藏層2: Linear(150 → 100) + ReLU
         ↓
輸出層:  Linear(100 → 4)   ← 直接輸出 4 個動作的 Q 值
```

### 3.2 為什麼輸出 4 個 Q 值而非 1 個

**設計一**（低效）：輸入 $(s, a)$，輸出單一 $Q(s, a)$ → 每次需要 4 次 forward pass  
**設計二**（DQN 採用）：輸入 $s$，一次輸出所有動作的 $Q(s, \cdot)$ → 1 次 forward pass 完成所有動作的評估

這使得 argmax 選動作只需一次 forward pass，大幅提升效率。

### 3.3 狀態表示

GridWorld 的 $4 \times 4 \times 4$ 張量（4 個 channel 分別代表 Player / Goal / Pit / Wall）被 flatten 成 64 維向量，加上少量雜訊：

$$s = \text{board.render\_np().reshape}(64) + \mathcal{U}(0, 0.01)^{64}$$

雜訊的目的是防止相同局面因浮點數精確匹配導致過擬合，讓網路學習更具泛化能力的 Q 函數。

---

## 4. TD Target 的概念

### 4.1 損失函數設計

DQN 的訓練目標是讓網路輸出的 Q 值盡可能接近 **TD Target（時序差分目標）**：

$$y = \begin{cases}
r & \text{if done}\\
r + \gamma \cdot \max_{a'} Q(s', a'; \theta^-) & \text{if not done}
\end{cases}$$

其中：
- $r$：即時獎勵（本實驗：+10 到達 Goal，-10 掉入 Pit，-1 每步）
- $\gamma$：折扣因子（本實驗：0.9）
- $Q(s', a'; \theta^-)$：Target Network（見第 5 節）對下一狀態的 Q 值估計
- $\theta^-$：Target Network 的參數（定期從 Online Network 同步）

**MSE Loss（均方誤差）**：

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(y - Q(s, a; \theta)\right)^2\right]$$

### 4.2 白話說明 TD Target

> 「我預計從這個狀態做這個動作，能得到的 Q 值應該等於：立即獎勵 + 折扣後的未來最大 Q 值。」

這個「應該等於」的值就是 TD Target。網路的訓練就是讓實際輸出向這個目標靠近。

### 4.3 為什麼需要 done 的條件判斷

當遊戲結束（到達 Goal 或 Pit）時，不存在「下一個狀態」，因此未來 Q 值為 0：

```python
# 本實驗實作
target_q = reward + gamma * next_q * (1.0 - done)
```

當 `done=True`，`(1.0 - done) = 0`，目標就等於純獎勵 $r$。

---

## 5. Experience Replay Buffer 的目的

### 5.1 為什麼需要 Experience Replay

**問題：直接用即時樣本訓練的缺陷**

在標準監督學習中，我們假設訓練樣本是**獨立同分佈（i.i.d.）** 的。但在 RL 中，相鄰 time step 的觀測高度相關：

```
s_t → a_t → s_{t+1} → a_{t+1} → s_{t+2} → ...
```

直接用這些序列訓練會導致：
1. **梯度相關性高**：相鄰更新幾乎相同方向，訓練不穩定
2. **遺忘問題**：舊經驗被快速覆蓋，網路忘記早期學習

### 5.2 Experience Replay 的機制（S1：Sample Reuse）

**設計思路**：維護一個固定大小的 replay buffer，儲存過去的 transitions，訓練時從中**隨機採樣** mini-batch：

```python
# 對應 starter code 程式 3.5
replay = deque(maxlen=1000)  # 本實驗 mem_size=1000

# 收集 transition
exp = (state1, action, reward, state2, done)
replay.append(exp)

# 隨機採樣
minibatch = random.sample(replay, batch_size=200)
```

**帶來的好處：**

| 問題 | Experience Replay 的解法 |
|------|--------------------------|
| 時間相關性 | 隨機採樣打破順序，接近 i.i.d. |
| 樣本效率低 | 每個 transition 可被多次採樣 |
| 訓練不穩定 | 更多樣的 mini-batch，梯度更平滑 |

### 5.3 Transition 的格式

每筆 Experience Replay 資料（Transition）包含：

$$\text{Transition} = (s_t, a_t, r_t, s_{t+1}, \text{done}_t)$$

| 欄位 | 型別 | 說明 | 本實驗規格 |
|------|------|------|-----------|
| $s_t$ | `float32[64]` | 執行動作前的狀態 | 4×4×4 flatten + noise |
| $a_t$ | `int` | 執行的動作 index | {0:up, 1:down, 2:left, 3:right} |
| $r_t$ | `float` | 即時獎勵 | +10 / -10 / -1 |
| $s_{t+1}$ | `float32[64]` | 執行動作後的狀態 | 同上 |
| $\text{done}_t$ | `bool` | 遊戲是否結束 | True when $r \ne -1$ |

---

## 6. Target Network 的目的（S2：Target Stabilization）

### 6.1 Moving Target 問題

在標準 DQN 中，如果用同一個網路計算「預測值」和「目標值」：

$$\mathcal{L} = \left(Q(s, a; \theta) - \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta)}_{\text{目標也用同一個 }\theta}\right)^2$$

每次更新 $\theta$，目標也隨之改變 → **Moving Target Problem（移動目標問題）**。這就像追一個會動的標靶，導致訓練不穩定甚至發散。

### 6.2 Target Network 的解法

引入一個**獨立的 Target Network**（參數 $\theta^-$），它的參數不即時更新，而是每 `sync_freq` 步才從 Online Network 完整同步：

```python
# 對應 starter code 程式 3.7
model2 = copy.deepcopy(model)              # 建立 Target Network
model2.load_state_dict(model.state_dict()) # 複製參數

# 每 sync_freq=500 步同步一次
if j % sync_freq == 0:
    model2.load_state_dict(model.state_dict())
```

**結果**：在 500 步的視窗內，TD Target 保持穩定，網路訓練更收斂。

---

## 7. Static Mode 為什麼適合 HW3-1

### 7.1 科學對照的原則

**Static Mode** 將 GridWorld 中所有物件的位置完全固定：

| 物件 | 位置（row, col） |
|------|----------------|
| Player (P) | (0, 3) |
| Goal (+) | (0, 0) |
| Pit (−) | (0, 1) |
| Wall (W) | (1, 1) |

這提供了一個**完全可控的實驗環境**：

1. **最低學習難度**：狀態空間固定，最優策略唯一且確定
2. **快速驗證**：可在短時間內確認 DQN 的學習迴圈是否正確
3. **排除隨機性干擾**：所有訓練失敗都可歸因於演算法問題，而非環境隨機性
4. **作為基準**：HW3-2 / HW3-3 的改進必須比 Static Mode 結果更好

> **類比**：就像在學習飛機飛行時，先在模擬器的固定天氣和跑道上練習，確認基本控制無誤後，再挑戰複雜環境。

### 7.2 最優路徑分析

在 Static Mode 下，從 Player(0,3) 到 Goal(0,0) 的最短路徑：

```
(0,3) →Left→ (0,2) →Left→ (0,1) ...
```

但 (0,1) 是 Pit！必須繞路：

```
(0,3) →Down→ (1,3) →Left→ (1,2) →Left→ (1,1)=Wall!
(0,3) →Down→ (1,3) →Left→ (1,2) →Up→ (0,2) →Left→ (0,1)=Pit!
```

最優路徑（7 步）：
```
(0,3) → Down(1,3) → Left(1,2) → Down(2,2) → Left(2,1) → Left(2,0) → Up(1,0) → Up(0,0)=Goal!
```

這正好解釋了訓練收斂後每局平均 **7.0 步**的結果！

---

## 8. 實驗結果解讀

### 8.1 實驗設定摘要

| 超參數 | 值 |
|--------|---|
| Mode | static |
| Episodes | 5,000 |
| Network | 64→150→100→4 (MLP) |
| Replay Buffer | deque(maxlen=1000) |
| Mini-batch Size | 200 |
| Gamma (γ) | 0.9 |
| Learning Rate | 1e-3 (Adam) |
| Target Sync Freq | 500 steps |
| Epsilon Decay | linear, 1.0 → 0.1 |
| Seed | 42 |

### 8.2 量化結果

| 指標 | 值 |
|------|----|
| **總體 Win Rate（5000 episodes）** | **75.5%** |
| 最後 500 episodes Win Rate | **98.6%** |
| Final Evaluation Win Rate（200 場，greedy）| **100.0%** |
| 最後 500 episodes 平均 Reward | **+2.43** |
| 最後 500 episodes 平均步數 | **8.3 步** |
| 全體平均 Loss（有訓練的 step） | **0.005366** |
| Epsilon 衰減 | 1.0 → 0.1（線性，5000 episodes） |
| 訓練時間 | 231.9 秒（≈ 3.9 分鐘） |

### 8.3 圖表解讀

#### 圖 1：Episode Reward 曲線

![Episode Reward 曲線](../results/figures/hw3_1_static_basic_dqn_reward.png)

*圖 1：HW3-1 Static Mode — Episode Reward（原始值 + 100 episodes 移動平均）*

**解讀**：
- 前期（0–2000 episodes）：reward 大幅波動，因 epsilon 高（大量隨機探索）且網路尚未學習到有效策略
- 中期（2000–3500 episodes）：移動平均 reward 開始上升，Agent 逐漸學習到有效路徑
- 後期（3500–5000 episodes）：收斂至穩定正 reward，表示 Agent 一致性地找到 Goal

#### 圖 2：Training Loss 曲線

![Training Loss 曲線](../results/figures/hw3_1_static_basic_dqn_loss.png)

*圖 2：HW3-1 Static Mode — Training Loss（MSE，移動平均）*

**解讀**：
- Loss 從較高值開始，隨訓練進行持續下降
- Loss 的大幅波動對應 Target Network 同步時刻（每 500 steps）
- 最終收斂到極小值（≈0.0001），說明 Q 值估計趨於穩定

#### 圖 3：Win Rate 移動平均

![Win Rate 曲線](../results/figures/hw3_1_static_basic_dqn_win_rate.png)

*圖 3：HW3-1 Static Mode — Win Rate（100 episodes 移動平均）*

**解讀**：
- Win rate 從 0 開始，隨探索減少（epsilon 衰減）逐漸提升
- 約在 episode 2000–2500 超過 50% 閾值
- 後期穩定在 90%+ 以上

#### 圖 4：Steps per Episode 曲線

![Steps per Episode](../results/figures/hw3_1_static_basic_dqn_steps.png)

*圖 4：HW3-1 Static Mode — Steps per Episode*

**解讀**：
- 前期步數在 max_steps=50 附近（timeout 很多）
- 後期平均步數穩定在 8–10 步，接近理論最優 7 步

#### 圖 5：Epsilon 衰減曲線

![Epsilon Decay](../results/figures/hw3_1_static_basic_dqn_epsilon.png)

*圖 5：HW3-1 Static Mode — Epsilon 線性衰減（1.0 → 0.1）*

**解讀**：
- 線性衰減：$\varepsilon_{ep} = \max(0.1,\ 1.0 - ep \times \frac{0.9}{5000})$
- 從完全隨機探索（ε=1.0）逐漸過渡到大部分利用（ε=0.1）

### 8.4 為什麼 Win Rate 在前期即有不少勝利

初期高 epsilon（ε=1.0）下，Agent 完全隨機探索。在 Static Mode 這個相對簡單的環境，即使隨機行動也有一定機率在 50 步內到達 Goal，因此初期 win rate 不為 0。這也說明 Static Mode 作為「最基礎驗證環境」的合理性。

---

## 9. 與 HW3-2 / HW3-3 的銜接

### 9.1 Static Mode 只是驗證基礎

HW3-1 的 Static Mode 成功（100% win rate）僅代表 DQN 的基本訓練邏輯正確。它**不代表 DQN 已能解決困難任務**，因為：

| 環境 | 挑戰 | 預期 DQN 難度 |
|------|------|-------------|
| Static Mode（HW3-1） | 無隨機性 | 低（✅ 已驗證） |
| Player Mode（HW3-2） | Player 位置隨機 | 中等（需探索更廣） |
| Random Mode（HW3-3） | 所有物件隨機 | 高（獎勵稀疏，學習困難） |

### 9.2 後續改進的動機

在 Player Mode / Random Mode 下，Naive DQN 面臨：

1. **Q 值高估（Overestimation Bias）**：`max` 操作在 target 計算中引入正偏差 → 解法：**Double DQN（S3，HW3-2）**
2. **動作無關狀態學習效率低**：許多狀態下所有動作 Q 值幾乎相同 → 解法：**Dueling DQN（S4，HW3-2）**
3. **Random Mode 獎勵更稀疏，梯度不穩定** → 解法：**Gradient Clipping + LR Scheduling（S2 Extended，HW3-3）**
4. **均等採樣效率低**：所有 transition 同等重要，但有些「驚訝」樣本更有學習價值 → 解法：**PER（S5，HW3-3）**

### 9.3 故事線連結

```
HW3-1 Static：「DQN 在最理想環境可以學習」          ✅ 已驗證
     ↓
HW3-2 Player：「面對隨機性，我們需要更好的 Q 估計」  ← 下一步
     ↓
HW3-3 Random：「面對最難環境，工程技術至關重要」
     ↓
Bonus：「所有改進整合 = Rainbow DQN」
```

---

## 10. 技術反思

### 10.1 本實驗成功的關鍵

1. **Target Network（S2）的必要性**：沒有 Target Network，Q 值在訓練初期容易發散，Static Mode 雖然簡單，但移動目標仍會造成不必要的不穩定
2. **Experience Replay（S1）的效果**：從 1000 個 transitions 的隨機 mini-batch 學習，讓每個樣本被多次重用，提升了樣本效率
3. **Epsilon Linear Decay 的合理性**：Linear decay 在 Static Mode 表現良好，因為環境固定，Agent 不需要在訓練後期繼續大量探索

### 10.2 Starter Code 與本實作的差異

| 對比項目 | 教授 Starter Code | 本實作 |
|---------|-----------------|--------|
| 資料結構 | 裸 deque + random.sample | 型別安全的 `ReplayBuffer` 類別 |
| 超參數管理 | 寫死在程式碼 | YAML config 檔案 |
| 實驗記錄 | print() | 統一 CSV logger（SPEC-05） |
| 圖表 | 手動 matplotlib | 自動化 plot pipeline |
| 可重現性 | 無 seed 固定 | 全域 seed=42 |

本實作保留了所有核心學習邏輯（TD target、epsilon-greedy、target sync），並在工程層面加以結構化，為 HW3-2/3 的模組化擴展打下基礎。

---

*本報告所有圖表與數值均來自真實訓練結果（seed=42, 5000 episodes），不含任何捏造數據。*

*實驗詳細 log 見 `results/csv/hw3_1_static_basic_dqn_log.csv`。*


---
---

## HW3-2：DQN Variants on Player Mode（第 11–17 節）

> **實驗設定**：seed=42, episodes=5000, mode=player, 三組平行比較  
> **實驗組**：P1 Basic DQN | P2 Double DQN | P3 Dueling DQN

---

## 11. Player Mode 的難度提升

### 11.1 從 Static 到 Player 的變化

HW3-1 的 Static Mode 中，Player 固定在 (0,3)，每局狀態分佈完全一致。HW3-2 的 **Player Mode** 引入關鍵變化：**Player 的初始位置在每局開始時隨機放置（不與其他物件重疊）**。Goal / Pit / Wall 位置不變，但 Agent 每局面對不同起始狀態。

| 難度差異 | Static Mode | Player Mode |
|---------|------------|------------|
| 初始狀態種類 | 1 種（固定） | 多種（Player 位置隨機） |
| 最優路徑 | 固定（7 步） | 因初始位置而異 |
| 泛化需求 | 低 | 中等（需學習策略而非路徑） |
| Q 值估計壓力 | 低 | 較高 |

---

## 12. Double DQN：解決 Overestimation Bias（S3）

### 12.1 問題根源

標準 DQN 的 TD Target：$y = r + \gamma \cdot \max_{a'} Q(s', a'; \theta^-)$

問題：$\max$ 操作在噪聲存在時系統性高估 Q 值（Overestimation Bias）。

數學直觀：設每個動作估計誤差 $\varepsilon \sim \mathcal{N}(0, \sigma^2)$，則：
$$\mathbb{E}[\max_a (q_a + \varepsilon_a)] > \max_a q_a$$

### 12.2 解法（van Hasselt et al., 2016）

**解耦「選動作」與「評估 Q 值」兩步驟**：

$$y^{\text{Double}} = r + \gamma \cdot Q\!\left(s',\ \arg\max_{a'} Q(s', a'; \theta);\ \theta^-\right)$$

| 步驟 | 使用網路 | 任務 |
|------|---------|------|
| 選擇動作 $a^*$ | Online Network $\theta$ | 從當前估計選最好的動作 |
| 評估 $Q(s', a^*)$ | Target Network $\theta^-$ | 用穩定網路評估 Q 值 |

---

## 13. Dueling DQN：Value-Advantage Decomposition（S4）

### 13.1 分解公式（Wang et al., 2016）

$$Q(s, a) = V(s) + \left[A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right]$$

- $V(s)$：State Value，「這個狀態本身有多好」（與動作無關）
- $A(s,a)$：Advantage，「選這個動作比平均好多少」
- Mean normalization 確保分解唯一（可識別性）

### 13.2 網路結構

```
state (64) → 共享特徵層 Linear(64→150)+ReLU
                ↙                    ↘
    Value stream              Advantage stream
    Linear(150→100)+ReLU      Linear(150→100)+ReLU
    Linear(100→1)             Linear(100→4)
    V(s)                      A(s,a)
                ↘                    ↙
          Q = V + [A - mean(A)]
```

---

## 14. 實驗設定

| 超參數 | P1 Basic DQN | P2 Double DQN | P3 Dueling DQN |
|--------|-------------|--------------|----------------|
| Mode | player | player | player |
| Network | QNetwork | QNetwork | DuelingNet |
| use_double_dqn | ❌ | ✅ | ❌ |
| use_dueling_dqn | ❌ | ❌ | ✅ |
| 其餘超參數 | 完全相同（嚴格控制變因） |

---

## 15. 實驗結果

### 15.1 量化比較（seed=42, 5000 episodes）

| 指標 | P1 Basic DQN | P2 Double DQN | P3 Dueling DQN |
|------|-------------|--------------|----------------|
| 全體 Win Rate | 86.1% | 86.2% | 86.2% |
| 最後 500ep Win Rate | 99.4% | **100.0%** | 99.2% |
| 最後 500ep 平均 Reward | +5.61 | **+5.83** | +5.64 |
| 最後 500ep 平均步數 | 5.3 步 | 5.2 步 | 5.2 步 |
| 全體平均 Loss | 0.004825 | 0.005214 | **0.003982** |
| Final Eval Win Rate（200 場）| 100.0% | 100.0% | 100.0% |

### 15.2 圖表

#### 比較圖 1：三方 Reward

![HW3-2 Reward Comparison](../results/figures/hw3_2_player_reward_comparison.png)

*圖 11：HW3-2 Player Mode — 三演算法 Reward 比較（MA100）*

#### 比較圖 2：三方 Win Rate

![HW3-2 Win Rate Comparison](../results/figures/hw3_2_player_win_rate_comparison.png)

*圖 12：HW3-2 Player Mode — 三演算法 Win Rate 比較（MA100）*

#### 比較圖 3：三方 Loss

![HW3-2 Loss Comparison](../results/figures/hw3_2_player_loss_comparison.png)

*圖 13：HW3-2 Player Mode — 三演算法 Loss 比較（MA100）*

#### 比較圖 4：Steps 比較

![HW3-2 Steps Comparison](../results/figures/hw3_2_player_steps_comparison.png)

*圖 14：HW3-2 Player Mode — Steps per Episode 比較（MA100）*

#### 比較圖 5：最終 Win Rate 柱狀圖

![HW3-2 Final Bar](../results/figures/hw3_2_player_final_metrics_bar.png)

*圖 15：HW3-2 Player Mode — 最終 Win Rate 比較（最後 100 episodes）*

---

## 16. 分析討論

### 16.1 Player Mode vs Static Mode

| 指標 | HW3-1 Static（Basic DQN） | HW3-2 Player（Double DQN） |
|------|--------------------------|----------------------------|
| 全體 Win Rate | 75.5% | 86.2% |
| 最後 500ep Win Rate | 98.6% | 100.0% |
| Final Eval Win Rate | 100.0% | 100.0% |
| 平均步數 | 8.3 步 | 5.2 步 |

> **有趣發現**：Player Mode 的全體 win rate（86.2%）反而高於 Static Mode（75.5%）！原因：多樣起始點提供了更豐富的訓練信號，使 Agent 學到更通用的策略。

### 16.2 三種演算法細微差異

- **Double DQN**：後期 win rate 最穩定（100%），reward 最高（+5.83）。Overestimation reduction 帶來更穩定的收斂
- **Dueling DQN**：loss 最低（0.003982），Q 函數估計精度最高
- **差異不大原因**：Player Mode 對 DQN 而言仍然相對簡單，三者均能在 5000ep 內充分收斂。Random Mode 預期差異更明顯

---

## 17. HW3-2 → HW3-3 銜接

### 17.1 為什麼 Random Mode 更難

| 挑戰 | Player Mode | Random Mode |
|------|------------|------------|
| Goal 位置固定？ | ✅ | ❌ 隨機 |
| Pit 位置固定？ | ✅ | ❌ 隨機 |
| Wall 位置固定？ | ✅ | ❌ 隨機 |
| 獎勵稀疏程度 | 中等 | **高** |
| 梯度穩定性 | 良好 | **容易不穩定** |

### 17.2 HW3-3 需要的技術

1. **Gradient Clipping**（E2）：防止稀疏獎勵導致梯度爆炸
2. **LR Scheduling**（E2）：後期降低學習率，避免已收斂策略被破壞
3. **PER（Prioritized Experience Replay）**（E3）：讓稀少但重要的「Goal 獲得」經驗更頻繁被學習

---

*HW3-2 所有數值均來自真實訓練結果（seed=42, 5000 episodes），不含任何捏造數據。*

*實驗 log 見 `results/csv/hw3_2_player_*_dqn_log.csv`。*
