# REQUIREMENTS TRACEABILITY MATRIX
# 教授需求 → 實作 → 驗收對照表（Phase 2 正式版）

> **文件版本**：v2.0
> **建立日期**：2026-05-13
> **作者**：Tony Lo（via Antigravity）
> **用途**：確保每一項教授需求都有對應的程式、設定、實驗、圖表、報告章節

---

## RTM-01：HW3-1 Naive DQN（30 分）

| # | 教授需求 | 對應程式 | 對應 Config | 對應實驗 | 對應圖表 | 報告章節 | 驗收狀態 |
|---|---------|---------|------------|---------|---------|---------|---------|
| 1.1 | Run Naive DQN code | `src/agents/dqn_agent.py` | `configs/hw3_1_static/basic_dqn_static.yaml` | `hw3_1_static_basic_dqn` | hw3_1_static_basic_dqn_*.png | 理解報告 §8 | ✅ 完成（100% win rate） |
| 1.2 | Experience Replay Buffer | `src/buffers/replay_buffer.py` | 同上 | 同上 | loss/reward 曲線 | 理解報告 §5 | ✅ 完成 |
| 1.3 | **understanding_report.md** | — | — | 同上 | **5 張圖嵌入** | **report/understanding_report.md** | ✅ **完成（10 節，全繁中）** |
| 1.4 | Static Mode 訓練 | `scripts/run_hw3_1_static.py` | 同上 | 同上 | — | — | ✅ 完成（5000ep, 231s） |
| 1.5 | 訓練 Loss 曲線 | `src/plotting/plot_curves.py` | — | 同上 | hw3_1_static_basic_dqn_loss.png | 理解報告 §8.3 | ✅ 完成 |

**HW3-1 小計：5/5 項需求完成 ✅ | Final Win Rate: 100%**

---

## RTM-02：HW3-2 Enhanced DQN Variants（40 分）

| # | 教授需求 | 對應程式 | 對應 Config | 對應實驗 | 對應圖表 | 報告章節 | 驗收狀態 |
|---|---------|---------|------------|---------|---------|---------|---------|
| 2.1 | Double DQN 實作 | `src/agents/dqn_agent.py`（use_double_dqn=true） | `configs/hw3_2_player/double_dqn_player.yaml` | `hw3_2_player_double_dqn` | hw3_2_player_*_comparison.png | 理解報告 §12、主報告 3.1 | ✅ 完成（100% win rate） |
| 2.2 | Dueling DQN 實作 | `src/agents/dqn_agent.py`（use_dueling_dqn=true）+ `src/models/dqn.py:DuelingNet` | `configs/hw3_2_player/dueling_dqn_player.yaml` | `hw3_2_player_dueling_dqn` | 同上 | 理解報告 §13、主報告 3.2 | ✅ 完成（100% win rate） |
| 2.3 | Player Mode 訓練（三組） | `scripts/run_hw3_2_player.py` | 3 個 yaml | P1/P2/P3 | — | — | ✅ 完成（P1/P2/P3 均 100%） |
| 2.4 | 與 Basic DQN 比較 | `src/plotting/plot_comparison.py:plot_hw3_2_comparison` | — | 三組對比 | **hw3_2_player_*_comparison.png（5 張）** | 主報告 3.3 | ✅ 完成 |
| 2.5 | 改進原理說明 | — | — | — | — | 理解報告 §12-13、主報告 3.1-3.2 | ✅ 完成 |
| 2.6 | Double DQN 解決 Overestimation | — | — | — | — | 理解報告 §12 + 公式推導 | ✅ 完成 |
| 2.7 | Dueling DQN 解決表示問題 | — | — | — | — | 理解報告 §13 + 架構圖 | ✅ 完成 |

**HW3-2 小計：7/7 項需求完成 ✅ | 最佳 Double DQN，Final Win Rate: 100%**


---

## RTM-03：HW3-3 Enhanced DQN for Random Mode（30 分）

| # | 教授需求 | 對應程式 | 對應 Config | 對應實驗 | 對應圖表 | 報告章節 | 驗收狀態 |
|---|---------|---------|------------|---------|---------|---------|---------|
| 3.1 | 轉換至 PyTorch Lightning | `src/training/lightning_dqn_module.py`（LightningDQNModule : pl.LightningModule） | e1/e2/e3 yaml（use_lightning=true） | E1, E2, E3 全部 | — | 理解報告 §19、主報告 4.2 | ✅ 完成 |
| 3.2 | Random Mode 訓練（E1/E2/E3） | `scripts/run_hw3_3_random.py` | `configs/hw3_3_random/e{1,2,3}_*.yaml` | E1/E2/E3 | — | — | ✅ 完成 |
| 3.3 | Gradient Clipping | `lightning_dqn_module.py:training_step()`（clip_grad_norm_）| e2/e3（use_gradient_clipping=true, max_norm=1.0）| E2, E3 | hw3_3_random_loss_comparison_e1_e2_e3.png | 理解報告 §20、主報告 4.1 | ✅ 完成 |
| 3.4 | LR Scheduling | `lightning_dqn_module.py:configure_optimizers()`（StepLR） | e2/e3（use_lr_scheduler=true）| E2, E3 | hw3_3_random_learning_rate_curve.png | 理解報告 §21、主報告 4.2 | ✅ 完成 |
| 3.5 | Training Tips 整合 + 消融比較 | 上述 | e1 vs e2 vs e3 | E1→E2→E3 消融 | **hw3_3_random_{reward,win_rate,loss,steps}_comparison_e1_e2_e3.png（4張）** | 主報告 4.3–4.5 | ✅ 完成 |
| 3.6 | PER 實作 | `src/buffers/prioritized_replay_buffer.py`（SumTree + IS weights） | e3（use_per=true, alpha=0.6）| E3 | hw3_3_random_final_metrics_e1_e2_e3.png | 理解報告 §23、主報告 4.5 | ✅ 完成 |
| 3.7 | Epsilon Decay Tuning | `lightning_dqn_module.py:decay_epsilon()`（linear/exponential） | e1(linear)→e2/e3(exp) | E1/E2/E3 | **hw3_3_random_epsilon_decay_comparison.png** | 理解報告 §22 | ✅ 完成 |
| 3.8 | Reward/Loss/Win-Rate 視覺化 | `src/plotting/plot_comparison.py:plot_hw3_3_comparison()` | — | E1, E2, E3 | 7 張比較圖 | 主報告 4.4 | ✅ 完成（7 張） |

**HW3-3 小計：8/8 項需求完成 ✅ | E3 PER+Stabilized 全體 Win Rate 最高（85.2%）**


---

## RTM-04：Rainbow DQN Bonus（加分）

| # | 教授需求（加分項） | 對應程式 | 對應 Config | 對應實驗 | 對應圖表 | 報告章節 | 驗收狀態 |
|---|---------|---------|------------|---------|---------|---------|---------|
| 4.1 | Rainbow 整合 | `src/agents/rainbow_dqn.py` | `configs/hw3_3_random/e4_rainbow.yaml` | E4 | hw3_3_e4_rainbow.png | Ch5 | ⬜ 待實作（Bonus） |
| 4.2 | Double DQN 整合至 Rainbow | 複用 `double_dqn.py` 邏輯 | e4_rainbow.yaml | E4 | — | Ch5.1 | ⬜ |
| 4.3 | Dueling Network 整合 | 複用 `dueling_net.py` | e4_rainbow.yaml | E4 | — | Ch5.1 | ⬜ |
| 4.4 | PER 整合至 Rainbow | 複用 `per_buffer.py` | e4_rainbow.yaml | E4 | — | Ch5.1 | ⬜ |
| 4.5 | Multi-step Learning | `src/buffers/multistep_buffer.py` | e4_rainbow.yaml | E4 | — | Ch5.2 | ⬜ |

**Bonus 小計：5 項需求 → 5 個對應**

---

## RTM-05：全局要求

| # | 全局要求 | 對應文件/設定 | 驗收狀態 |
|---|---------|-------------|---------|
| G1 | GitHub Repo 公開可存取 | `README.md` + repo URL | ⬜ 待 push |
| G2 | README 說明如何重現 | `README.md` | ✅ 已建立 |
| G3 | 可重現性（固定 seed） | `src/utils/seed.py` + all configs | ⬜ 待實作 |
| G4 | 圖表嵌入報告（非只放資料夾） | `report/*.md` | ⬜ 待填入 |
| G5 | 不假造實驗結果 | `scripts/generate_report_assets.py` | ✅ 規則建立 |
| G6 | 所有超參數在 yaml 管理 | `configs/**/*.yaml` | ⬜ 待實作 |
| G7 | 統一 logging schema | `src/utils/logger.py` | ⬜ 待實作 |
| G8 | understanding_report.md 完整 | `report/understanding_report.md` | ⬜ **最高優先** |

---

## RTM-06：OpenSpec Change 追蹤

| Change ID | 涵蓋 RTM 項目 | 狀態 |
|-----------|-------------|------|
| CHG-001 | Repo Bootstrap（G1, G2） | ✅ Applied |
| CHG-002（待建） | 核心基礎設施（G3, G6, G7）+ HW3-1（1.1～1.5） | ⬜ Planned |
| CHG-003（待建） | HW3-2 Double/Dueling DQN（2.1～2.7） | ⬜ Planned |
| CHG-004（待建） | HW3-3 Lightning + Training Tips（3.1～3.8） | ⬜ Planned |
| CHG-005（待建） | Rainbow Bonus E4（4.1～4.5） | ⬜ Planned |
| CHG-006（待建） | 報告整合與圖表嵌入（G4, G8） | ⬜ Planned |

---

## 驗收狀態圖例

| 符號 | 意義 |
|------|------|
| ✅ | 已完成並驗證 |
| ⬜ | 尚未開始 |
| 🔄 | 進行中 |
| ⚠️ | 有問題需處理 |
| ❌ | 失敗或跳過（不應出現） |

---

## 緊急優先順序

基於截止日緊迫性，建議執行順序：

```
P0（立即）：understanding_report.md 骨架填入 + HW3-1 程式碼可執行
P1（今日）：HW3-2 Double + Dueling DQN 實作
P2（今日）：HW3-3 E1-E3 PyTorch Lightning 實作
P3（若有時間）：Bonus E4 Rainbow Pipeline
P4（最後）：報告圖表嵌入與最終潤稿
```
