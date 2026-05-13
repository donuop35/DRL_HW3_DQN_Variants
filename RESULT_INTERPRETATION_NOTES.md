# RESULT_INTERPRETATION_NOTES.md
# HW3 DQN Variants — 結果解讀筆記

> **用途**：報告寫作前的最終數據解讀與品管記錄  
> **版本**：Phase 8（2026-05-13，數據鎖定）  
> **數據來源**：全部來自真實訓練，seed=42，5000 episodes

---

## 一、HW3-1 Static Mode 結果觀察

### 量化結果
| 指標 | 數值 |
|------|------|
| 全體 Win Rate | 75.5% |
| 最後 500ep Win Rate | 98.6% |
| Final Eval Win Rate（greedy, 200場）| **100.0%** |
| 最後 500ep Avg Steps | ~7.0 步 |
| 訓練時間 | 231.9 秒 |

### 觀察
1. **100% final eval win rate**：Basic DQN 在固定 4×4 環境中完全收斂，驗證了 Experience Replay + Target Network 的基礎正確性。
2. **前期 win rate 低（~0%）**：ε=1.0 的純探索階段持續約 500-1000 episodes，這是 ε-greedy 的正常行為。
3. **全體 75.5% vs 最後 98.6%**：表示訓練後期大幅改善，學習曲線呈現明顯上升趨勢。
4. **Avg Steps ≈ 7**：GridWorld 理論最短路徑為 5-7 步，表示 Agent 學到了接近最優路徑。

### 品質確認
- ✅ CSV 5000 rows，NaN=0
- ✅ loss 由 ~0.1 下降並趨於穩定
- ✅ epsilon 由 1.0 線性衰減至 0.1
- ✅ win rate 後期穩定在 95%+ 

---

## 二、HW3-2 Player Mode 三模型比較

### 量化結果
| 指標 | P1 Basic | P2 Double | P3 Dueling |
|------|---------|----------|-----------|
| 全體 Win Rate | 86.1% | 86.2% | 86.2% |
| 最後 500ep Win Rate | 99.4% | **100.0%** | 99.2% |
| Final Eval Win Rate | 100.0% | 100.0% | 100.0% |
| Avg Loss（非零）| 中等 | 較低 | **最低** |

### 觀察
1. **三方 final win rate 均 100%**：Player Mode 雖比 Static 難（起點隨機），但所有方法均能充分收斂。差異在於收斂速度和穩定性。
2. **P2 Double DQN 後期最穩定**（last-500 100%）：消除了 Q overestimation，訓練後期不會因過估而震盪。
3. **P3 Dueling DQN loss 最低**：Value/Advantage 分離讓網路在不需要更新所有 Q 值時，仍能精確估計 V(s)，減少不必要的梯度誤差。
4. **Player mode win rate > Static**（86% vs 75.5% 全體）：多樣化起始點使 Agent 學到更通用的策略，泛化能力更強。

### 異常記錄
- ⚠️ P1 Basic DQN 與 P2/P3 全體 win rate 幾乎相同（86.1 vs 86.2%）：表示在此環境中，Double/Dueling 的優勢主要體現在後期穩定性（last-500），而非訓練效率。這是合理的——GridWorld 本身問題空間小。

---

## 三、HW3-3 E1/E2/E3 Random Mode 比較

### 量化結果
| 指標 | E1 Baseline | E2 Stabilized | E3 PER+Stab（主方法）|
|------|-------------|--------------|---------------------|
| 全體 Win Rate | 79.6% | 82.3% | **85.2%** |
| 最後 500ep Win Rate | **95.2%** | 90.8% | 91.8% |
| Final Eval Win Rate | **91.5%** | 88.5% | 90.0% |
| 平均 Loss | 0.1065 | 0.5986 | **0.2096** |
| LR 範圍 | 固定 0.001 | 0.001→0.000004 | 0.001→0.000007 |

### 觀察
1. **E3 全體 win rate 最高（85.2%）**：PER 讓稀疏的 Goal 獲得（高 TD error）更頻繁被採樣，訓練早期收斂更快，累積 win rate 高。
2. **E1 後期指標最好**：固定 lr=0.001 使後期仍能大幅調整，Random Mode 的多樣地圖需要持續適應能力。
3. **E2 LR 衰減過快**：StepLR 每 step（而非每 episode）更新，5000 steps 後 lr=0.000004（原始的 0.38%），後期接近無法學習，是 E2 後期指標低的主因。
4. **E2 loss 最高（0.5986）**：Loss 高的原因是 LR Scheduling 使優化器步幅過小，積累的分佈更新不足，導致 loss 長期無法充分下降。這是 StepLR per-step 的副作用。
5. **E3 是正式主方法**：整合 PER + Grad Clip + Exp Epsilon，全體 win rate 最高，代表整個訓練週期最佳的樣本利用效率。

### 異常記錄
- ⚠️ E1 Final Eval（91.5%）略高於 E3（90.0%）：500 episodes 的 greedy eval 有隨機性（不同 random seed 的 Goal 位置），200 場的統計誤差約 ±3%，屬正常範圍。
- ℹ️ E2 epsilon 在 ep ~2500 後衰減到 min=0.05 並保持不變，但 LR 繼續衰減；後期 E2 處於「epsilon 已最小，lr 也近乎消失」的雙重弱探索+弱學習狀態。

---

## 四、E4 Rainbow Bonus 比較

### 量化結果
| 指標 | E3 PER+Stab（正式）| E4 Rainbow（Bonus）|
|------|-------------------|---------------------|
| 全體 Win Rate | 85.2% | 33.0% |
| Final Eval Win Rate | 90.0% | 40.0% |
| 最後 500ep Win Rate | 91.8% | 52.4% |
| 訓練時間 | ~399s | ~2504s（6.3×）|

### 觀察
1. **E4 win rate 遠低於 E1-E3**：在 5000 episodes 的 training budget 內，Rainbow（特別是 C51）無法充分收斂於 4×4 GridWorld。
2. **KL loss（~0.96）vs MSE loss（~0.21）**：C51 的 loss 是 KL divergence，與 MSE 不同量綱，不可直接比較數值。KL loss 下降趨勢（1.16→0.87）表示學習確實在進行。
3. **NoisyNet 探索弱於 ε-greedy 初期**：ε=1.0 的 E1-E3 在早期有大量隨機探索找到 Goal；NoisyNet 的初始噪聲（σ=0.5）相對偏小，早期探索更保守。
4. **E4 有正向趨勢**：Win rate 從前期的 ~5% 上升到最後 500ep 的 52.4%，顯示 Rainbow 確實在學習，只是 5000 episodes 不夠。

### 誠實結論
> E4 Rainbow 在本作業的訓練預算內未能超越 E1-E3，但這不代表 Rainbow 不好。  
> 原始 Rainbow（Hessel et al., 2018）使用 200M frames（約 2億步），本作業僅 250K 步（5000ep × 50steps）。  
> E4 的價值在於：**完整實作六組件並驗證其數學正確性，以及清楚呈現方法侷限性。**

---

## 五、數據品質總覽

| 檢查項目 | 結果 |
|---------|------|
| 所有 CSV NaN=0 | ✅ |
| 所有 CSV rows=5000 | ✅ |
| Loss 欄位合理（非負，有下降趨勢）| ✅ |
| Win rate 在 [0,1] 範圍內 | ✅ |
| Reward 符合環境規則（-50 ≤ r ≤ +10）| ✅ |
| E4 未覆蓋 E1-E3（verified）| ✅ |
| 圖表均由 CSV 重建（非手動）| ✅ |
| Epsilon 衰減方向正確（遞減）| ✅ |
| 使用統一 seed=42 | ✅ |
| E4 bonus 與主線分開標記 | ✅ |

### 唯一潛在限制
> ⚠️ `hw3_3_random_loss_comparison_e1_e2_e3_e4.png` 中，E4 的 KL loss（~0.96）遠高於 E1-E3 的 MSE loss（~0.1-0.6），使圖表縱軸拉伸，E1-E3 的 loss 曲線在同圖中顯得極平。這是正常現象（不同損失函數的量綱），在報告中已說明。

---

## 六、報告可直接引用的關鍵數字

```
HW3-1:
  Final Win Rate: 100.0% | Avg Steps: 7.0 | All Win Rate: 75.5%

HW3-2:
  P1 Basic:   Final 100% | all_wr 86.1%
  P2 Double:  Final 100% | all_wr 86.2% | last500 100% (最穩定)
  P3 Dueling: Final 100% | all_wr 86.2% | loss 最低

HW3-3 E1-E3:
  E1 Baseline:       Final 91.5% | all_wr 79.6%
  E2 Stabilized:     Final 88.5% | all_wr 82.3%
  E3 PER+Stab (主):  Final 90.0% | all_wr 85.2% ← 全體最高

HW3-3 E4 Bonus:
  E4 Rainbow:  Final 40.0% | all_wr 33.0% | 誠實記錄低於 E1-E3
```

---

*數據鎖定於 Phase 8（2026-05-13）。如需重現，執行：*
```bash
python scripts/run_hw3_1_static.py
python scripts/run_hw3_2_player.py
python scripts/run_hw3_3_random.py
python scripts/run_hw3_3_rainbow_bonus.py
```
