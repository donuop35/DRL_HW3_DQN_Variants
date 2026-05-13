# EXPERIMENT STORYLINE — DRL HW3: DQN and its Variants

> **文件版本**：v1.0  
> **建立日期**：2026-05-13  
> **作者**：Tony Lo  
> **注意**：本文件規劃實驗敘事方向，實際結果待實驗完成後填入

---

## 敘事主軸

> 「從最簡單的靜態環境出發，逐步挑戰更複雜的隨機環境，透過一系列 DQN 變體的改進，展示深度強化學習的演進歷程。」

---

## Phase 1：Baseline 建立（HW3-1 Static Mode）

### 研究問題
- Naive DQN 是否能在完全靜態的 GridWorld 中穩定收斂？
- Experience Replay Buffer 對學習穩定性有何影響？

### 預期敘事
1. 在 Static Mode 下，環境完全固定，DQN 理論上應能穩定收斂
2. 學習曲線應展示從探索到利用（exploration → exploitation）的轉變
3. Understanding Report 記錄對 Naive DQN 原理的深層理解

### 關鍵指標
- **訓練 Reward 曲線**：是否持續上升並趨於穩定
- **收斂速度**：Episode 數 vs. 平均 Reward
- **最終性能**：收斂後的平均 Reward

### 對應圖表
- [ ] `results/figures/hw3_1_training_curve.png`
- [ ] `results/figures/hw3_1_epsilon_decay.png`

---

## Phase 2：DQN 變體比較（HW3-2 Player Mode）

### 研究問題
- Double DQN 相較 Naive DQN，在 Player Mode 下是否學習更穩定？
- Dueling DQN 的狀態-優勢分解，對學習效率有何改善？
- 三種方法的收斂速度與最終性能如何排序？

### 預期敘事
1. **Naive DQN**：在 Player Mode 下可能因 Q 值過估而不穩定
2. **Double DQN**：透過解耦動作選擇與評估，減少過估偏差
3. **Dueling DQN**：透過分離 V(s) 與 A(s,a)，對無關動作的狀態有更好的估計

### 比較矩陣（待填入）

| 方法 | 收斂 Episode | 最終 Avg Reward | 訓練穩定性 |
|------|------------|----------------|-----------|
| Naive DQN | TBD | TBD | TBD |
| Double DQN | TBD | TBD | TBD |
| Dueling DQN | TBD | TBD | TBD |

### 對應圖表
- [ ] `results/figures/hw3_2_comparison_curves.png`（三種方法同圖比較）
- [ ] `results/figures/hw3_2_double_dqn_curve.png`
- [ ] `results/figures/hw3_2_dueling_dqn_curve.png`

---

## Phase 3：最難環境挑戰（HW3-3 Random Mode）

### 研究問題
- 在完全隨機的環境中，哪些 Training Tips 最有效？
- PyTorch Lightning 的訓練框架如何提升實驗管理效率？
- 各種 Training Tips 組合對學習穩定性的影響？

### 預期敘事
1. **無 Training Tips**：Random Mode 下學習不穩定（baseline）
2. **+ Gradient Clipping**：防止梯度爆炸，訓練更穩定
3. **+ LR Scheduling**：動態調整學習率，改善收斂品質
4. **+ Epsilon Decay 優化**：更智慧的探索策略

### Training Tips 消融實驗設計（Ablation Study）

| 設定 | Gradient Clipping | LR Scheduling | Epsilon Strategy | 預期效果 |
|------|-------------------|---------------|-----------------|---------|
| Baseline | ❌ | ❌ | Linear | 基準線 |
| +GC | ✅ | ❌ | Linear | 穩定性提升 |
| +LR | ❌ | ✅ | Linear | 收斂改善 |
| Full | ✅ | ✅ | Exponential | 最佳性能 |

### 對應圖表
- [ ] `results/figures/hw3_3_ablation_study.png`
- [ ] `results/figures/hw3_3_training_tips_comparison.png`

---

## Phase 4：Rainbow 整合（Bonus）

### 研究問題
- Rainbow DQN 的各元素對 Random Mode GridWorld 的貢獻度如何？
- 整合 PER 後，樣本效率是否顯著提升？
- Multi-step Learning 對訓練速度的影響？

### Rainbow 元素加入順序
1. Double DQN + Dueling（已在 HW3-2 完成）
2. \+ Prioritized Experience Replay (PER)
3. \+ Multi-step Learning (N=3)
4. \+ Noisy Networks（可選）

### 對應圖表
- [ ] `results/figures/bonus_rainbow_ablation.png`
- [ ] `results/figures/bonus_rainbow_vs_naive.png`

---

## 整體故事線圖（Story Arc）

```
環境難度   難
    ↑
    │  Random Mode  ←── HW3-3 (Training Tips) ←── Bonus (Rainbow)
    │
    │  Player Mode  ←── HW3-2 (Double/Dueling DQN)
    │
    │  Static Mode  ←── HW3-1 (Naive DQN)
    │
    └─────────────────────────────────────────────→ 模型複雜度
                       簡單                         複雜
```

---

## 報告撰寫策略

每個 HW 的報告區塊應包含：

1. **實驗目的**：為什麼做這個實驗？
2. **方法說明**：用繁體中文解釋演算法原理
3. **超參數設定**：記錄所有實驗設定
4. **結果圖表**：訓練曲線、比較圖
5. **分析討論**：結果的意義與解讀
6. **結論**：本階段的主要發現
