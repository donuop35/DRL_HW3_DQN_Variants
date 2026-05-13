# HW3 DQN Variants 研究型實驗報告

> **課程**：深度強化學習（Deep Reinforcement Learning）  
> **作業**：HW3 — DQN and its Variants  
> **作者**：Tony Lo（中興大學）  
> **日期**：2026-05-13  
> **狀態**：HW3-1 ✅、HW3-2 ✅、HW3-3 🔄（進行中）

---

## 摘要（Abstract）

本報告探討 DQN（Deep Q-Network）及其多種變體在 GridWorld 環境下的學習表現，以嚴謹的實驗設計呈現三個遞進難度：

1. **HW3-1（Static Mode）**：驗證 Basic DQN 的基礎學習能力，確認 Experience Replay + Target Network 的必要性。最終 win rate 100%。
2. **HW3-2（Player Mode）**：比較 Basic DQN / Double DQN / Dueling DQN 三種方法，探討在隨機起始位置下的泛化表現。三者均達到 100% final win rate，Double DQN 後期最穩定。
3. **HW3-3（Random Mode）**：（進行中）探討全隨機環境下的梯度穩定技巧與 PER。

最終目標：理解各改進方法的原理，並展示它們如何從 Naive DQN 逐步演進到 Rainbow DQN。

---

## 1. 實驗設置（Experimental Setup）

### 1.1 環境

GridWorld 4×4 格子世界，包含 Player、Goal、Pit、Wall 四種物件：

| 模式 | 難度 | 說明 | 本報告狀態 |
|------|------|------|-----------|
| Static | ★☆☆ | 所有物件固定 | ✅ HW3-1 完成 |
| Player | ★★☆ | Player 隨機、Goal/Pit/Wall 固定 | ✅ HW3-2 完成 |
| Random | ★★★ | 全部物件隨機 | 🔄 HW3-3 進行中 |

**獎勵設計**：Goal +10，Pit -10，每步 -1（最多 50 步）。

### 1.2 硬體環境

| 項目 | 規格 |
|------|------|
| CPU | Intel Core i7-9750H @ 2.60GHz |
| GPU | 無（CPU 訓練） |
| RAM | 16 GB |
| 作業系統 | macOS |

### 1.3 軟體版本

| 套件 | 版本 |
|------|------|
| Python | 3.9.6 |
| PyTorch | 2.2.2 |
| NumPy | 1.26.4 |
| Pandas | 最新穩定版 |

### 1.4 通用超參數（HW3-1、HW3-2 相同）

| 超參數 | 值 | 說明 |
|--------|---|------|
| Episodes | 5000 | 每組實驗總訓練局數 |
| Max Steps per Episode | 50 | 避免無限循環 |
| Gamma (γ) | 0.9 | 折扣因子 |
| Learning Rate | 1e-3 | Adam optimizer |
| Batch Size | 200 | Mini-batch 大小 |
| Replay Buffer | deque(1000) | 固定容量，FIFO |
| Target Sync Frequency | 500 steps | Target Network 更新頻率 |
| Epsilon | 1.0 → 0.1 | 線性衰減 |
| Seed | 42 | 全域可重現性固定 |

---

## 2. HW3-1：Naive DQN（Static Mode）

> **詳細理論說明見 `understanding_report.md` 第 1–10 節**

### 2.1 方法說明

Basic DQN 整合兩個核心技術：

1. **Experience Replay Buffer（S1）**：維護容量 1000 的 deque，訓練時隨機採樣 mini-batch（batch=200），打破時序相關性，實現接近 i.i.d. 的訓練分佈。

2. **Target Network（S2）**：獨立的固定參數網路，每 500 steps 同步一次，解決 Moving Target Problem，提升訓練穩定性。

**網路架構**（對應 starter code）：
```
輸入 state (64) → Linear(64→150)+ReLU → Linear(150→100)+ReLU → Linear(100→4)
                                                                   ↑
                                                           4 個動作的 Q 值
```

**TD Target**：$y = r + \gamma \cdot \max_{a'} Q(s', a'; \theta^-)$（done 時 $y = r$）

### 2.2 實驗結果

| 指標 | 值 |
|------|----|
| 全體 Win Rate（5000 episodes） | 75.5% |
| 最後 500ep Win Rate | 98.6% |
| **Final Evaluation Win Rate（200 場）** | **100.0%** |
| 最後 500ep 平均 Reward | +2.43 |
| 最後 500ep 平均步數 | 8.3 步（接近理論最優 7 步） |
| 全體平均 Loss | 0.005366 |
| 訓練時間 | 231.9 秒 |

![HW3-1 Reward 曲線](../results/figures/hw3_1_static_basic_dqn_reward.png)

*圖 1：HW3-1 Static Mode — Reward 學習曲線*

![HW3-1 Win Rate 曲線](../results/figures/hw3_1_static_basic_dqn_win_rate.png)

*圖 2：HW3-1 Static Mode — Win Rate 學習曲線*

### 2.3 分析

**成功原因**：
- Static Mode 的固定狀態空間讓 Basic DQN 能快速學會單一最優路徑
- Target Network 避免了訓練發散，Experience Replay 提供穩定的訓練批次
- 線性 Epsilon 衰減在固定環境下表現良好，不需要後期繼續探索

**侷限性**：
- 75.5% 的全體 win rate 說明前期探索效率不高（初期 epsilon 過高，浪費了很多隨機行動）
- Static Mode 只有一個固定起始點，無法測試策略的**泛化**能力

---

## 3. HW3-2：DQN Variants（Player Mode）

> **詳細理論說明見 `understanding_report.md` 第 11–17 節**

### 3.1 Double DQN（P2）

**核心改進**：解耦「選動作」與「評估 Q 值」兩步驟，消除 Overestimation Bias：

$$y^{\text{Double}} = r + \gamma \cdot Q\!\left(s',\ \arg\max_{a'} Q(s', a'; \theta);\  \theta^-\right)$$

- 選動作：使用 **Online Network**（$\theta$）
- 評估 Q 值：使用 **Target Network**（$\theta^-$）

**實現**（1 行程式碼差異）：

```python
# Basic DQN
next_q = target_net(next_states).max(dim=1)[0]

# Double DQN
next_actions = online_net(next_states).argmax(dim=1, keepdim=True)
next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
```

### 3.2 Dueling DQN（P3）

**核心改進**：明確分解 Q 函數為狀態值 V(s) 和動作優勢 A(s,a)：

$$Q(s, a) = V(s) + \left[A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right]$$

**網路架構**：共享特徵 → Value Stream（輸出 1 維）+ Advantage Stream（輸出 4 維）→ 組合

### 3.3 三種方法比較

| 方法 | 全體 Win Rate | 最後 500ep Win Rate | 最後 500ep Reward | Loss | 訓練時間 |
|------|-------------|-------------------|-----------------|------|---------|
| **P1 Basic DQN** | 86.1% | 99.4% | +5.61 | 0.004825 | 217.5s |
| **P2 Double DQN** | 86.2% | **100.0%** | **+5.83** | 0.005214 | 211.2s |
| **P3 Dueling DQN** | 86.2% | 99.2% | +5.64 | **0.003982** | 276.8s |
| Final Eval（200 場）| 100.0% | 100.0% | 100.0% | — | — |

**🏆 結論**：三者 Final Eval 均達 100%。細節上 **Double DQN 後期最穩定（100%），Dueling DQN 的 Q 值估計最精確（loss 最低）**。

![HW3-2 三方比較 Win Rate](../results/figures/hw3_2_player_win_rate_comparison.png)

*圖 3：HW3-2 Player Mode — 三演算法 Win Rate 比較（MA100）*

![HW3-2 三方比較 Reward](../results/figures/hw3_2_player_reward_comparison.png)

*圖 4：HW3-2 Player Mode — 三演算法 Reward 比較（MA100）*

![HW3-2 最終 Win Rate 柱狀圖](../results/figures/hw3_2_player_final_metrics_bar.png)

*圖 5：HW3-2 Player Mode — 最終 Win Rate 比較*

**關鍵觀察**：Player Mode 的全體 win rate（86%）**反而高於** Static Mode（75.5%）！多樣化的起始位置提供了更豐富的訓練信號，使 Agent 學到更通用的策略。

---

## 4. HW3-3：Enhanced DQN（Random Mode + Training Tips）

> **（進行中，待 Phase 6 完成後填入）**

---

## 5. Bonus：Rainbow DQN

> **（待完成後填入）**

---

## 6. 結論（Conclusion）

### 已完成結論（HW3-1、HW3-2）

1. **Basic DQN 在固定環境下有效**：HW3-1 Static Mode 達到 100% final win rate，驗證了 Experience Replay + Target Network 的基礎架構正確
2. **DQN Variants 在 Player Mode 均能收斂**：三種方法 final win rate 均達 100%，差異主要體現在收斂穩定性（Double DQN）和 Q 值精度（Dueling DQN）
3. **環境難度遞進驗證了演算法改進的動機**：Static → Player 的轉換顯示泛化能力的重要性，為 Random Mode 引入更多技術打下鋪墊

---

## 參考文獻（References）

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
2. van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI 2016*.
3. Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML 2016*.
4. Schaul, T., et al. (2016). Prioritized Experience Replay. *ICLR 2016*.
5. Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI 2018*.
6. 教授提供的 Starter Code：`第3章程式_ALL_IN_ONE.ipynb`
