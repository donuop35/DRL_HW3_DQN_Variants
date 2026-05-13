# FIGURE_MANIFEST.md
# HW3 DQN Variants — 完整圖表清單

> **資料來源聲明**：所有圖表均由真實訓練 CSV 自動生成，不含手動捏造。  
> **生成時間**：2026-05-13  
> **種子**：全部 seed=42  

---

## 圖例

| 欄位 | 說明 |
|------|------|
| Status | ✅ 正式圖 / ⚠️ smoke test |
| Report Section | 對應報告章節 |

---

## HW3-1 Static Mode（S-static）

| File | 對應實驗 | 對應 CSV | 報告章節 | 說明 | Status |
|------|---------|---------|---------|------|--------|
| `hw3_1_static_basic_dqn_reward.png` | S-static | `hw3_1_static_basic_dqn_log.csv` | 主報告 2.2 / 理解報告 §8 | Reward 學習曲線（MA100）| ✅ |
| `hw3_1_static_basic_dqn_win_rate.png` | S-static | 同上 | 主報告 2.2 | Win Rate 學習曲線（MA100）| ✅ |
| `hw3_1_static_basic_dqn_loss.png` | S-static | 同上 | 理解報告 §8 | Training Loss（MA100）| ✅ |
| `hw3_1_static_basic_dqn_steps.png` | S-static | 同上 | 理解報告 §8 | Steps per Episode（MA100）| ✅ |
| `hw3_1_static_basic_dqn_epsilon.png` | S-static | 同上 | 理解報告 §8 | Epsilon Decay Curve | ✅ |

**HW3-1 圖表：5 張，全部正式圖**

---

## HW3-2 Player Mode（P1/P2/P3）

### 個別曲線（P1 Basic）

| File | 對應實驗 | 對應 CSV | 報告章節 | 說明 | Status |
|------|---------|---------|---------|------|--------|
| `hw3_2_player_basic_dqn_reward.png` | P1 | `hw3_2_player_basic_dqn_log.csv` | — | P1 Reward 曲線 | ✅ |
| `hw3_2_player_basic_dqn_win_rate.png` | P1 | 同上 | — | P1 Win Rate 曲線 | ✅ |
| `hw3_2_player_basic_dqn_loss.png` | P1 | 同上 | — | P1 Loss 曲線 | ✅ |
| `hw3_2_player_basic_dqn_steps.png` | P1 | 同上 | — | P1 Steps 曲線 | ✅ |
| `hw3_2_player_basic_dqn_epsilon.png` | P1 | 同上 | — | P1 Epsilon 曲線 | ✅ |

### 個別曲線（P2 Double）

| File | 對應實驗 | 對應 CSV | 報告章節 | 說明 | Status |
|------|---------|---------|---------|------|--------|
| `hw3_2_player_double_dqn_reward.png` | P2 | `hw3_2_player_double_dqn_log.csv` | — | P2 Reward 曲線 | ✅ |
| `hw3_2_player_double_dqn_win_rate.png` | P2 | 同上 | — | P2 Win Rate | ✅ |
| `hw3_2_player_double_dqn_loss.png` | P2 | 同上 | — | P2 Loss | ✅ |
| `hw3_2_player_double_dqn_steps.png` | P2 | 同上 | — | P2 Steps | ✅ |
| `hw3_2_player_double_dqn_epsilon.png` | P2 | 同上 | — | P2 Epsilon | ✅ |

### 個別曲線（P3 Dueling）

| File | 對應實驗 | 對應 CSV | 報告章節 | 說明 | Status |
|------|---------|---------|---------|------|--------|
| `hw3_2_player_dueling_dqn_reward.png` | P3 | `hw3_2_player_dueling_dqn_log.csv` | — | P3 Reward 曲線 | ✅ |
| `hw3_2_player_dueling_dqn_win_rate.png` | P3 | 同上 | — | P3 Win Rate | ✅ |
| `hw3_2_player_dueling_dqn_loss.png` | P3 | 同上 | — | P3 Loss | ✅ |
| `hw3_2_player_dueling_dqn_steps.png` | P3 | 同上 | — | P3 Steps | ✅ |
| `hw3_2_player_dueling_dqn_epsilon.png` | P3 | 同上 | — | P3 Epsilon | ✅ |

### HW3-2 正式比較圖（報告必備）

| File | 對應實驗 | 對應 CSV | 報告章節 | 說明 | Status |
|------|---------|---------|---------|------|--------|
| `hw3_2_player_reward_comparison.png` | P1/P2/P3 | 3個 CSV | 主報告 3.3 / 理解報告 §15 | **三方 Reward 比較（MA100）** | ✅ 正式 |
| `hw3_2_player_win_rate_comparison.png` | P1/P2/P3 | 3個 CSV | 主報告 3.3 | **三方 Win Rate 比較** | ✅ 正式 |
| `hw3_2_player_loss_comparison.png` | P1/P2/P3 | 3個 CSV | 理解報告 §15 | 三方 Loss 比較 | ✅ 正式 |
| `hw3_2_player_steps_comparison.png` | P1/P2/P3 | 3個 CSV | 理解報告 §15 | 三方 Steps 比較 | ✅ 正式 |
| `hw3_2_player_final_metrics_bar.png` | P1/P2/P3 | 3個 CSV | 主報告 3.3 | **最終 Win Rate 柱狀圖** | ✅ 正式 |

**HW3-2 圖表：20 張（15 個別 + 5 比較），比較圖為報告必備**

---

## HW3-3 Random Mode E1/E2/E3（正式主線）

### 個別曲線（E1）

| File | 對應實驗 | 對應 CSV | 說明 | Status |
|------|---------|---------|------|--------|
| `hw3_3_random_e1_baseline_reward.png` | E1 | `hw3_3_random_e1_baseline_log.csv` | E1 Reward | ✅ |
| `hw3_3_random_e1_baseline_win_rate.png` | E1 | 同上 | E1 Win Rate | ✅ |
| `hw3_3_random_e1_baseline_loss.png` | E1 | 同上 | E1 Loss | ✅ |
| `hw3_3_random_e1_baseline_steps.png` | E1 | 同上 | E1 Steps | ✅ |

### 個別曲線（E2）

| File | 對應實驗 | 對應 CSV | 說明 | Status |
|------|---------|---------|------|--------|
| `hw3_3_random_e2_stabilized_reward.png` | E2 | `hw3_3_random_e2_stabilized_log.csv` | E2 Reward | ✅ |
| `hw3_3_random_e2_stabilized_win_rate.png` | E2 | 同上 | E2 Win Rate | ✅ |
| `hw3_3_random_e2_stabilized_loss.png` | E2 | 同上 | E2 Loss | ✅ |
| `hw3_3_random_e2_stabilized_steps.png` | E2 | 同上 | E2 Steps | ✅ |

### 個別曲線（E3）

| File | 對應實驗 | 對應 CSV | 說明 | Status |
|------|---------|---------|------|--------|
| `hw3_3_random_e3_per_stabilized_reward.png` | E3 | `hw3_3_random_e3_per_stabilized_log.csv` | E3 Reward | ✅ |
| `hw3_3_random_e3_per_stabilized_win_rate.png` | E3 | 同上 | E3 Win Rate | ✅ |
| `hw3_3_random_e3_per_stabilized_loss.png` | E3 | 同上 | E3 Loss | ✅ |
| `hw3_3_random_e3_per_stabilized_steps.png` | E3 | 同上 | E3 Steps | ✅ |

### HW3-3 E1-E3 正式比較圖（報告必備）

| File | 對應實驗 | 對應 CSV | 報告章節 | 說明 | Status |
|------|---------|---------|---------|------|--------|
| `hw3_3_random_reward_comparison_e1_e2_e3.png` | E1/E2/E3 | 3個 CSV | 主報告 4.4 / 理解報告 §25 | **E1-E3 Reward 比較** | ✅ 正式 |
| `hw3_3_random_win_rate_comparison_e1_e2_e3.png` | E1/E2/E3 | 3個 CSV | 主報告 4.4 | **E1-E3 Win Rate 比較** | ✅ 正式 |
| `hw3_3_random_loss_comparison_e1_e2_e3.png` | E1/E2/E3 | 3個 CSV | 理解報告 §25 | E1-E3 Loss 比較 | ✅ 正式 |
| `hw3_3_random_steps_comparison_e1_e2_e3.png` | E1/E2/E3 | 3個 CSV | 理解報告 §25 | E1-E3 Steps 比較 | ✅ 正式 |
| `hw3_3_random_final_metrics_e1_e2_e3.png` | E1/E2/E3 | 3個 CSV | 主報告 4.4 | **最終 Win Rate 柱狀圖（E1-E3）** | ✅ 正式 |
| `hw3_3_random_epsilon_decay_comparison.png` | E1/E2/E3 | 3個 CSV | 主報告 4.4 / 理解報告 §22 | **Epsilon Decay 比較（線性 vs 指數）** | ✅ 正式 |
| `hw3_3_random_learning_rate_curve.png` | E2/E3 | 2個 CSV | 主報告 4.4 / 理解報告 §21 | **LR Scheduling 曲線（StepLR）** | ✅ 正式 |

---

## HW3-3 E4 Rainbow Bonus（Bonus，不影響 E1-E3）

### E4 個別曲線

| File | 對應實驗 | 對應 CSV | 說明 | Status |
|------|---------|---------|------|--------|
| `hw3_3_random_e4_rainbow_bonus_reward.png` | E4 | `hw3_3_random_e4_rainbow_bonus_log.csv` | E4 Reward | ✅ Bonus |
| `hw3_3_random_e4_rainbow_bonus_win_rate.png` | E4 | 同上 | E4 Win Rate | ✅ Bonus |
| `hw3_3_random_e4_rainbow_bonus_loss.png` | E4 | 同上 | E4 Loss（KL divergence）| ✅ Bonus |
| `hw3_3_random_e4_rainbow_bonus_steps.png` | E4 | 同上 | E4 Steps | ✅ Bonus |

### E1-E4 總比較圖（Bonus，含 E4）

| File | 對應實驗 | 對應 CSV | 報告章節 | 說明 | Status |
|------|---------|---------|---------|------|--------|
| `hw3_3_random_reward_comparison_e1_e2_e3_e4.png` | E1-E4 | 4個 CSV | 主報告 5.4 | E1-E4 Reward 比較 | ✅ Bonus |
| `hw3_3_random_win_rate_comparison_e1_e2_e3_e4.png` | E1-E4 | 4個 CSV | 主報告 5.4 | **E1-E4 Win Rate 比較** | ✅ Bonus |
| `hw3_3_random_loss_comparison_e1_e2_e3_e4.png` | E1-E4 | 4個 CSV | 主報告 5.4 | E1-E4 Loss 比較（scale 不同）| ✅ Bonus |
| `hw3_3_random_steps_comparison_e1_e2_e3_e4.png` | E1-E4 | 4個 CSV | 主報告 5.4 | E1-E4 Steps 比較 | ✅ Bonus |
| `hw3_3_random_final_metrics_e1_e2_e3_e4.png` | E1-E4 | 4個 CSV | 主報告 5.4 | 最終 Win Rate 柱狀圖（E1-E4）| ✅ Bonus |

---

## 圖表統計彙整

| 類別 | 圖表數 | 必備圖（報告） | Bonus 圖 |
|------|-------|-------------|---------|
| HW3-1 個別 | 5 | 2（reward/win_rate）| — |
| HW3-2 個別 | 15 | — | — |
| HW3-2 比較 | 5 | 5 | — |
| HW3-3 E1-E3 個別 | 12 | — | — |
| HW3-3 E1-E3 比較 | 7 | 7 | — |
| HW3-3 E4 個別 | 4 | — | 4 |
| HW3-3 E1-E4 比較 | 5 | — | 5 |
| **合計** | **53** | **14 正式必備** | **9 Bonus** |

---

*所有圖表路徑均相對於 `results/figures/`*  
*生成程式：`src/plotting/plot_curves.py` + `src/plotting/plot_comparison.py`*
