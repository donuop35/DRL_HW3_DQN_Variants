# HW3 DQN Variants 研究型實驗報告

> **課程**：深度強化學習  
> **作業**：HW3 — DQN and its Variants  
> **作者**：Tony Lo  
> **日期**：2026-05-13  
> **狀態**：🔄 骨架初版（各節待實驗完成後填入）

---

## 摘要（Abstract）

本報告探討 DQN（Deep Q-Network）及其多種變體在 GridWorld 環境下的表現，從最基礎的 Naive DQN 出發，逐步引入 Double DQN、Dueling DQN，並在最難的 Random Mode 環境下整合多種訓練技巧（Training Tips）。最終目標是理解各種改進方法的原理與效果，並展示 Rainbow DQN 的綜合能力。

---

## 1. 實驗設置（Experimental Setup）

### 1.1 環境

GridWorld 4×4 格子環境，三種模式：

| 模式 | 難度 | 說明 |
|------|------|------|
| Static | ★☆☆ | 所有物件固定 |
| Player | ★★☆ | Player 隨機 |
| Random | ★★★ | 全部隨機 |

### 1.2 硬體環境

| 項目 | 規格 |
|------|------|
| CPU | TBD |
| GPU | TBD（若有） |
| RAM | TBD |
| 作業系統 | macOS |

### 1.3 軟體版本

| 套件 | 版本 |
|------|------|
| Python | TBD |
| PyTorch | TBD |
| PyTorch Lightning | TBD |

---

## 2. HW3-1：Naive DQN（Static Mode）

> 詳細內容見 `understanding_report.md`

### 2.1 方法說明

（待填入）

### 2.2 實驗結果

（圖表與數據待填入）

### 2.3 分析

（待填入）

---

## 3. HW3-2：Enhanced DQN Variants（Player Mode）

### 3.1 Double DQN

#### 3.1.1 原理說明

（待填入）

#### 3.1.2 實驗結果

（待填入）

### 3.2 Dueling DQN

#### 3.2.1 原理說明

（待填入）

#### 3.2.2 實驗結果

（待填入）

### 3.3 三種方法比較

| 方法 | 收斂速度 | 最終性能 | 訓練穩定性 |
|------|---------|---------|-----------|
| Naive DQN | TBD | TBD | TBD |
| Double DQN | TBD | TBD | TBD |
| Dueling DQN | TBD | TBD | TBD |

（比較圖待填入）

---

## 4. HW3-3：Enhanced DQN（Random Mode + Training Tips）

### 4.1 框架轉換：PyTorch Lightning

（待填入）

### 4.2 Training Tips

#### 4.2.1 Gradient Clipping

（待填入）

#### 4.2.2 Learning Rate Scheduling

（待填入）

#### 4.2.3 消融實驗（Ablation Study）

（表格與圖表待填入）

---

## 5. Bonus：Rainbow DQN

### 5.1 Rainbow 元素整合

（待填入）

### 5.2 實驗結果

（待填入）

---

## 6. 結論（Conclusion）

（待填入）

---

## 參考文獻（References）

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
2. van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI 2016*.
3. Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML 2016*.
4. Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI 2018*.
5. 教授提供的 Starter Code：`第3章程式_ALL_IN_ONE.ipynb`
