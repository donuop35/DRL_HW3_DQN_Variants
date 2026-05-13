# ASSIGNMENT REQUIREMENTS — HW3: DQN and its Variants

> **來源**：教授課程說明（2026-05-13 截止）  
> **整理者**：Tony Lo  
> **注意**：本文件為原始評分標準的整理，不得修改評分內容

---

## 基本資訊

| 項目 | 內容 |
|------|------|
| 類型 | 個人作業 |
| 開放繳交 | 2026-04-22 00:00 |
| **截止日期** | **2026-05-13（已逾期，務必立即完成）** |
| 成績佔比 | 10% |
| 評分方式 | 直接打分數 |
| 允許遲交 | 否 |

---

## 1. Setup & Reference

- 以 **DRL in Action（英文版）GitHub Repo** 作為基礎：  
  https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master
- 使用教授提供的 **updated starter code** 作為 baseline（`第3章程式_ALL_IN_ONE.ipynb`）

### 環境模式說明

| 模式名稱 | 說明 | Player 位置 | 其他物件（Goal, Pit, Wall）位置 | 適用情境 |
|---------|------|------------|-------------------------------|---------|
| **static** | 完全固定，所有物件位置固定 | 固定 (0,3) | Goal→(1,0), Pit→(0,1), Wall→(1,1) | 用於測試訓練基本能力或可能的過擬合問題 |
| **player** | 只有 Player 隨機，其餘固定 | 隨機位置 | 固定（同上） | 增加不確定性，測試泛化能力 |
| **random** | 所有物件（Player, Goal, Pit, Wall）完全隨機 | 隨機 | 隨機 | 用於訓練最強泛化能力 |

---

## 2. HW3-1：Naive DQN for Static Mode（30%）

### 要求

- ✅ 執行提供的 Naive DQN 程式碼，或 Experience Buffer 重放
- ✅ 與 ChatGPT 討論以澄清對程式碼的理解
- ✅ **繳交一份簡短的 Understanding Report**

### 包含內容

- **Basic DQN implementation** for an easy environment
- **Experience Replay Buffer**

### 評分重點

1. 程式碼能正確執行
2. Understanding Report 的品質（展示真實理解，不是複製貼上）

> ⚠️ **Understanding Report 是必繳項目，缺交直接失分**

---

## 3. HW3-2：Enhanced DQN Variants for Player Mode（40%）

### 要求

實作並比較以下方法：

- **Double DQN**
- **Dueling DQN**

### 核心問題

> 💡 這兩種方法如何改善基礎 DQN 的問題？

- **Double DQN**：解決 Q 值高估（overestimation bias）問題
- **Dueling DQN**：分離 State Value V(s) 與 Advantage A(s,a)，改善狀態表示

### 評分重點

1. Double DQN 實作正確性
2. Dueling DQN 架構設計
3. 與 Naive DQN 的比較分析（需有圖表）

---

## 4. HW3-3：Enhance DQN for Random Mode WITH Training Tips（30%）

### 要求

將 DQN 模型從 PyTorch 轉換至以下其中之一：

- **Keras**，或
- **PyTorch Lightning**

### 加分項目（Bonus）

整合以下 **training techniques** 以穩定/改善學習：

- **Gradient Clipping**（梯度裁剪）
- **Learning Rate Scheduling**（學習率排程）
- 其他穩定訓練的技術（Epsilon Decay 策略等）

### 評分重點

1. 框架轉換的正確性
2. Training Tips 整合完整度
3. 在 Random Mode 的學習曲線品質

---

## 5. Rainbow DQN（Bonus）

整合 Rainbow DQN 的核心元素：

- Double DQN ✓（已在 HW3-2 實作）
- Dueling Network ✓（已在 HW3-2 實作）
- Prioritized Experience Replay (PER)
- Multi-step Learning
- Noisy Networks（可選）
- Distributional RL：C51 / QR-DQN（可選）

---

## 6. 必繳交項目清單

| 項目 | 必/選 | 對應 HW |
|------|-------|---------|
| `understanding_report.md` | **必繳** | HW3-1 |
| Static Mode DQN 程式碼 | **必繳** | HW3-1 |
| Double DQN 實作 | **必繳** | HW3-2 |
| Dueling DQN 實作 | **必繳** | HW3-2 |
| 比較圖表（HW3-2） | **必繳** | HW3-2 |
| Random Mode + Framework 轉換 | **必繳** | HW3-3 |
| Training Tips | **必繳** | HW3-3 |
| Rainbow 整合 | 加分 | Bonus |
| GitHub Repo 連結 | **必繳** | 全部 |
