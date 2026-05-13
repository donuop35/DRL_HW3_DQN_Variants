# EXPERIMENT STORYLINE — DRL HW3: DQN and its Variants
# 連貫實驗敘事（Phase 2 正式版）

> **文件版本**：v2.0（Phase 2 正式版）
> **建立日期**：2026-05-13
> **作者**：Tony Lo（via Antigravity）
> **敘事原則**：三個 HW 是一條演進軌跡，不是三份斷裂作業

---

## 🎯 核心研究主線

> **「我們如何讓 DQN 在越來越複雜的環境中，越來越穩定地學習？」**

```
Basic DQN（無 Replay）
    ↓  [S1] 加入 Experience Replay → 打破時間相關性
Experience Replay DQN
    ↓  [S2] 加入 Target Network → 穩定學習目標
Stabilized DQN（HW3-1 基礎，Static Mode）
    ↓  [S3] 解決 Q 值過估 → Double DQN
    ↓  [S4] 改善狀態表示 → Dueling DQN
Enhanced DQN Variants（HW3-2，Player Mode）
    ↓  [S5] 框架化 + 訓練穩定化 + 優先採樣 + 探索調控
PER-Enhanced Stabilized DQN（HW3-3，Random Mode）
    ↓  [ALL] Rainbow 整合
Advanced Rainbow DQN Bonus（E4 Bonus Pipeline）
```

### 固定機制標識（不得更名）

| 標識 | 機制名稱 | 解決的問題 | 引入時機 |
|------|---------|-----------|---------|
| **S1** | Experience Replay / Sample Reuse | 時間相關性、樣本效率低 | HW3-1 |
| **S2** | Target Stabilization / Training Stabilization | 學習目標不穩定（moving target） | HW3-1 |
| **S3** | Overestimation Reduction（Double DQN） | Q 值高估偏差 | HW3-2 |
| **S4** | Value–Advantage Decomposition（Dueling DQN） | 動作無關狀態學習效率低 | HW3-2 |
| **S5** | Prioritized Sampling + Exploration Control（PER + Epsilon） | 樣本利用效率、探索-利用平衡 | HW3-3 |

---

## 第一章：從固定環境出發——驗證 DQN 的基礎能力（HW3-1）

### 1.1 研究背景

在開始挑戰複雜環境之前，我們首先需要回答一個基本問題：

> **「在最理想的條件下，DQN 是否真的能學會解決問題？」**

為此，我們選擇 **Static Mode** GridWorld——所有物件位置完全固定——作為我們的起點。這不是退縮，而是科學的基本姿態：先在可控環境中驗證方法的有效性，再進入複雜世界。

### 1.2 從 Naive DQN 到 Stabilized DQN

教授的 starter code 展示了三個演進階段（對應程式 3.3→3.5→3.8）：

**階段 1：Naive DQN（程式 3.3）**
- 沒有 Experience Replay
- 沒有 Target Network
- 訓練不穩定，容易發散

**階段 2：加入 S1（Experience Replay）（程式 3.5）**
- `deque(maxlen=1000)` 存儲歷史經驗
- 隨機抽 batch=200 訓練
- 打破相鄰樣本的時間相關性

**階段 3：加入 S2（Target Network）（程式 3.8）**
- `model2 = copy.deepcopy(model)` 建立目標網路
- 每 `sync_freq=500` 步同步一次
- 穩定 Bellman Target，避免追逐移動目標

### 1.3 HW3-1 實驗設計

| 指標 | 說明 |
|------|------|
| 環境 | GridWorld Static Mode（4×4） |
| 訓練 Episodes | 5000（對應程式 3.8） |
| 評估 | test_model()，勝率（1000 場） |
| 主要圖表 | Loss 曲線、Epsilon 衰減曲線、Reward 曲線 |

### 1.4 預期敘事結論

- Static Mode 下，加入 S1+S2 後，DQN 應能穩定收斂
- 這為我們進入 Player Mode 建立了信心基礎
- understanding_report.md 記錄這個學習過程的深層理解

---

## 第二章：環境開始變化——比較 DQN 變體如何應對（HW3-2）

### 2.1 研究背景

有了 Static Mode 的成功，我們進入 **Player Mode**：Player 的起始位置隨機，其他物件固定。

這引出了一個新問題：

> **「當環境引入不確定性，Naive DQN 的哪些缺陷會被放大？」**

答案是兩個核心問題：
1. **Q 值高估（Overestimation Bias）**：DQN 的 max 操作導致 Q 值持續高估
2. **狀態-動作表示效率**：許多狀態下，只有少數動作真正重要

### 2.2 S3：Double DQN 的解法

**問題根源**：在標準 DQN 中，同一個網路同時負責「選動作」和「評估 Q 值」，形成雙重偏差。

**解法邏輯**：
- **選動作**：用 Online Network 選出最佳動作 a*
- **評估 Q 值**：用 Target Network 評估 a* 的 Q 值

```
Standard DQN：Y = r + γ · max_a' Q_target(s', a')
Double DQN：  a* = argmax_a' Q_online(s', a')
             Y = r + γ · Q_target(s', a*)
```

### 2.3 S4：Dueling DQN 的解法

**問題根源**：在許多格子世界狀態中，不同動作的 Q 值差異很小（如遠離目標的空格），標準 DQN 難以有效學習。

**解法邏輯**：分離網路為兩個分支，分別學習「狀態值 V(s)」與「動作優勢 A(s,a)」：

```
Q(s,a) = V(s) + [A(s,a) - mean_a'(A(s,a'))]
```

好處：V(s) 對所有動作都更新（即使某動作未被選中），提升樣本效率。

### 2.4 HW3-2 實驗設計

三條實驗曲線必須同圖比較：

| 曲線 | 方法 | 顏色建議 |
|------|------|---------|
| Baseline | Naive DQN（S1+S2）| 藍色 |
| +S3 | Double DQN | 橙色 |
| +S4 | Dueling DQN | 綠色 |

### 2.5 預期敘事結論

- Player Mode 下，Naive DQN 應比 Static Mode 表現更不穩定
- Double DQN 應展示更低的 Q 值估計方差
- Dueling DQN 應在稀疏獎勵情境下展示更好的樣本效率

---

## 第三章：完全未知的世界——用工程力穩定學習（HW3-3）

### 3.1 研究背景

**Random Mode** 是最難的挑戰：Player、Goal、Pit、Wall 全部隨機放置。

問題不只是演算法層面的，還有工程層面的：

> **「在最難的環境中，我們需要哪些工程工具來讓訓練穩定可靠？」**

這正是 HW3-3 的核心：**框架化 + 訓練技術**。

### 3.2 為什麼轉換至 PyTorch Lightning？

PyTorch Lightning 解決的問題：
- **訓練迴圈模板化**：消除重複的 `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- **內建 logging**：自動記錄 training metrics
- **Trainer 抽象**：一行切換 CPU/GPU，自動處理 device
- **可組合性**：Callback 機制讓 Training Tips 易插拔

### 3.3 S5：訓練穩定化技術（Training Tips）

#### Gradient Clipping（梯度裁剪）

```python
# PyTorch Lightning 中一行搞定
trainer = Trainer(gradient_clip_val=1.0)
```

作用：防止梯度爆炸（在 Random Mode 獎勵稀疏時特別重要）

#### Learning Rate Scheduling（學習率排程）

```python
# StepLR：每 N epoch 乘以 gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

# CosineAnnealingLR：餘弦衰減
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
```

作用：訓練後期用較小學習率細調，避免震盪

#### Prioritized Experience Replay（PER）

標準 Replay Buffer 的問題：所有樣本均等採樣，但並非所有樣本同樣有教育意義。

PER 核心：根據 TD-Error 給每個樣本一個優先級 $p_i = |δ_i|^\alpha + ε$

```python
# 採樣概率
P(i) = p_i^α / Σ_k p_k^α

# 重要性採樣權重（修正偏差）
w_i = (N · P(i))^(-β) / max_j w_j
```

#### Epsilon Decay Tuning（探索策略調控）

```python
# Linear Decay（starter code 原版）
epsilon -= 1/epochs

# Exponential Decay（更常見）
epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-step / decay_rate)

# 策略選擇建議：Exponential 在前期快速探索，後期精細利用
```

### 3.4 HW3-3 固定實驗設計（不得更改）

#### E1：Random DQN Baseline
- 設定：基礎 DQN（S1+S2），Random Mode
- 目的：建立最難環境的 baseline
- 無任何 Training Tips

#### E2：Stabilized DQN（+S2 Extended）
- 設定：E1 + Gradient Clipping（`clip=1.0`）+ LR Scheduling（StepLR）
- PyTorch Lightning 框架
- 目的：展示 Training Tips 的穩定化效果

#### E3：PER-DQN + Stabilization（+S5）
- 設定：E2 + Prioritized Experience Replay + Exponential Epsilon Decay
- 目的：最優化的正式實驗配置

> ⚠️ **防呆鎖定**：E1→E2→E3 是 HW3-3 的正式主線，不得被 E4 取代。

### 3.5 預期敘事結論

- E1 應展示 Random Mode 的學習困難（高 loss 方差）
- E2 應展示 Training Tips 帶來的穩定性提升
- E3 應展示 PER 帶來的樣本效率提升

---

## 第四章：彙整所有改進——Rainbow DQN Bonus（E4）

### 4.1 定位

E4 是獨立 Bonus Pipeline，不影響 E1～E3。

> Rainbow 的意義不在於「最高分」，而在於展示：
> **「當我們把所有有效的改進組合在一起，它們能協同工作嗎？」**

### 4.2 Rainbow 組合策略

```
E4 = Double DQN (S3)
   + Dueling Network (S4)
   + PER (S5)
   + Multi-step Learning (N=3)
   [+ Noisy Networks（可選）]
```

### 4.3 消融分析（E4 內部）

Rainbow 的貢獻分析：

| 移除的元素 | 預期影響 |
|-----------|---------|
| 移除 Double DQN | Q 值高估重現 |
| 移除 Dueling | 稀疏獎勵狀態學習變慢 |
| 移除 PER | 樣本效率下降 |
| 移除 Multi-step | 獎勵傳播變慢 |

---

## 全局實驗敘事線圖（Summary Arc）

```
                    環境難度
                       ↑
                       │
    Random Mode  ──────┤──── E1（Baseline）
                       │       → E2（+Training Tips）
                       │           → E3（+PER+Epsilon）
                       │               → E4（Rainbow Bonus）
                       │
    Player Mode  ──────┤──── Naive DQN（from HW3-1）
                       │       → Double DQN（+S3）
                       │           → Dueling DQN（+S4）
                       │
    Static Mode  ──────┤──── Naive DQN（驗證）
                       │       → +Experience Replay（S1）
                       │           → +Target Network（S2）
                       │
                       └──────────────────────────────→ 模型複雜度
                           Simple          Complex
```

---

## 報告章節對應

| HW | 機制 | 報告章節 | 關鍵圖表 |
|----|------|---------|---------|
| HW3-1 | S1+S2 | understanding_report.md + 報告 Ch2 | Loss 曲線、勝率 |
| HW3-2 | +S3+S4 | 報告 Ch3 | 三方比較曲線 |
| HW3-3 E1 | Baseline | 報告 Ch4.1 | Random Mode Loss |
| HW3-3 E2 | +Training Tips | 報告 Ch4.2 | 穩定性對比 |
| HW3-3 E3 | +PER+Epsilon | 報告 Ch4.3 | 消融實驗 |
| Bonus E4 | Rainbow | 報告 Ch5 | Rainbow vs E3 |
