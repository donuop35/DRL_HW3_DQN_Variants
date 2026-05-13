# HW3-1 理解報告（Understanding Report）

> **課程**：深度強化學習  
> **作業**：HW3-1 — Naive DQN for Static Mode  
> **作者**：Tony Lo  
> **日期**：2026-05-13  
> **狀態**：🔄 待完成（實驗完成後填入）

---

## 1. DQN 基礎理解

### 1.1 為什麼需要 Deep Q-Network？

傳統 Q-Learning 使用查找表（Q-table）儲存每個狀態-動作對的 Q 值，但在狀態空間較大時會面臨：

- **維度災難**（Curse of Dimensionality）：狀態空間爆炸，無法儲存所有 Q 值
- **泛化能力不足**：相似的狀態無法共享學習結果

DQN 以**神經網路近似 Q 函數**，解決上述問題：

$$Q(s, a; \theta) \approx Q^*(s, a)$$

其中 $\theta$ 為神經網路參數。

---

### 1.2 DQN 的核心組件

#### Experience Replay Buffer（經驗回放緩衝區）

**作用**：打破時間相關性（temporal correlation），使訓練樣本更接近 i.i.d.

**機制**：
1. 將每步互動 $(s, a, r, s')$ 存入 Buffer
2. 訓練時從 Buffer 隨機抽取 mini-batch
3. 使用抽到的樣本更新網路參數

**為什麼重要**：
- 若直接使用連續的經驗訓練，相鄰樣本高度相關，易導致訓練不穩定
- Random sampling 使每個樣本被多次使用，提升資料效率

#### Target Network（目標網路）

**作用**：穩定訓練目標（Bellman Target），避免「追逐移動的目標」

**機制**：
- 維護兩個網路：Online Network（$\theta$）與 Target Network（$\theta^-$）
- Target Network 每隔固定步數從 Online Network 複製
- 計算 TD Target 時使用 Target Network：

$$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$$

---

### 1.3 DQN 更新公式

**損失函數（Loss）**：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( y_t - Q(s_t, a_t; \theta) \right)^2 \right]$$

其中 TD Target：$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$

---

## 2. GridWorld 環境分析

### 2.1 環境描述

GridWorld 是一個 4×4 的格子世界，包含：

| 物件 | 符號 | 說明 |
|------|------|------|
| Player | P | 受控制的代理人 |
| Goal | G | 目標位置，到達得正分 |
| Pit | X | 陷阱，跌入得負分 |
| Wall | W | 牆壁，碰到會彈回 |

### 2.2 Static Mode 設定

| 物件 | 固定位置 |
|------|---------|
| Player | (0, 3) |
| Goal | (1, 0) |
| Pit | (0, 1) |
| Wall | (1, 1) |

### 2.3 獎勵設計

```
到達 Goal：+10
跌入 Pit：-10
其他步驟：-1（鼓勵快速找到目標）
```

---

## 3. 實驗設定

> ⚠️ **本節待實驗執行後填入實際設定**

### 3.1 超參數

| 超參數 | 設定值 | 選擇理由 |
|--------|--------|---------|
| learning_rate | TBD | TBD |
| gamma | TBD | TBD |
| epsilon_start | TBD | TBD |
| epsilon_end | TBD | TBD |
| epsilon_decay | TBD | TBD |
| batch_size | TBD | TBD |
| memory_size | TBD | TBD |
| hidden_dim | TBD | TBD |
| target_update_freq | TBD | TBD |

---

## 4. 實驗結果

> ⚠️ **本節待實驗執行後填入**

### 4.1 訓練曲線

（圖片）

### 4.2 收斂分析

| 指標 | 數值 |
|------|------|
| 收斂所需 Episode 數 | TBD |
| 最終平均 Reward | TBD |
| 訓練時間 | TBD |

---

## 5. 討論與心得

> ⚠️ **本節待實驗完成後以繁體中文撰寫真實心得**

### 5.1 觀察到的現象

（待填入）

### 5.2 遭遇的困難與解決方式

（待填入）

### 5.3 對 DQN 的深入理解

（待填入）

---

## 6. 結論

（待填入）

---

*本報告以繁體中文撰寫，所有結果均來自真實實驗。*
