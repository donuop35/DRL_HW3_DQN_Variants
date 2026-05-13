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
| 1.1 | Run Naive DQN code | `src/agents/naive_dqn.py` | `configs/hw3_1_static/default.yaml` | `hw3_1_static_naive_dqn` | hw3_1_loss_curve.png | Ch2.1 | ⬜ 待實作 |
| 1.2 | Experience Replay Buffer | `src/buffers/replay_buffer.py` | 同上 | 同上 | hw3_1_loss_curve.png | Ch2.2 | ⬜ 待實作 |
| 1.3 | **understanding_report.md** | — | — | 同上 | **嵌入圖表** | **understanding_report.md** | ⬜ **最高優先** |
| 1.4 | Static Mode 訓練 | `scripts/run_hw3_1_static.py` | 同上 | 同上 | — | — | ⬜ 待實作 |
| 1.5 | 訓練 Loss 曲線（隱性） | `src/plotting/plot_utils.py` | — | 同上 | hw3_1_loss_curve.png | Ch2.1 | ⬜ 待實作 |

**HW3-1 小計：5 項需求 → 5 個對應**

---

## RTM-02：HW3-2 Enhanced DQN Variants（40 分）

| # | 教授需求 | 對應程式 | 對應 Config | 對應實驗 | 對應圖表 | 報告章節 | 驗收狀態 |
|---|---------|---------|------------|---------|---------|---------|---------|
| 2.1 | Double DQN 實作 | `src/agents/double_dqn.py` | `configs/hw3_2_player/double_dqn.yaml` | `hw3_2_player_double_dqn` | hw3_2_double_curve.png | Ch3.1 | ⬜ 待實作 |
| 2.2 | Dueling DQN 實作 | `src/agents/dueling_dqn.py` + `src/models/dueling_net.py` | `configs/hw3_2_player/dueling_dqn.yaml` | `hw3_2_player_dueling_dqn` | hw3_2_dueling_curve.png | Ch3.2 | ⬜ 待實作 |
| 2.3 | Player Mode 訓練 | `scripts/run_hw3_2_player.py` | 兩個 config | 兩個實驗 | — | — | ⬜ 待實作 |
| 2.4 | 與 Basic DQN 比較 | `src/plotting/plot_utils.py` | — | `hw3_2_player_naive_dqn`（對照） | **hw3_2_comparison_all.png** | Ch3.3 | ⬜ 待實作 |
| 2.5 | 改進原理說明 | — | — | — | — | Ch3.1/3.2 文字說明 | ⬜ 待撰寫 |
| 2.6 | Double DQN 改善 overestimation（說明） | — | — | — | — | Ch3.1 | ⬜ 待撰寫 |
| 2.7 | Dueling DQN 改善表示（說明） | — | — | — | — | Ch3.2 | ⬜ 待撰寫 |

**HW3-2 小計：7 項需求 → 7 個對應**

---

## RTM-03：HW3-3 Enhanced DQN for Random Mode（30 分）

| # | 教授需求 | 對應程式 | 對應 Config | 對應實驗 | 對應圖表 | 報告章節 | 驗收狀態 |
|---|---------|---------|------------|---------|---------|---------|---------|
| 3.1 | 轉換至 PyTorch Lightning | `src/agents/lightning_dqn.py` + `src/training/lightning_trainer.py` | `configs/hw3_3_random/e2_stabilized.yaml` | E2, E3 | — | Ch4 | ⬜ 待實作 |
| 3.2 | Random Mode 訓練 | `scripts/run_hw3_3_random.py` | e1, e2, e3 configs | E1, E2, E3 | — | — | ⬜ 待實作 |
| 3.3 | Gradient Clipping | `src/training/lightning_trainer.py` | e2_stabilized.yaml | E2, E3 | hw3_3_e2_vs_e1.png | Ch4.2 | ⬜ 待實作 |
| 3.4 | LR Scheduling | `src/training/lightning_trainer.py` | e2_stabilized.yaml | E2, E3 | hw3_3_e2_vs_e1.png | Ch4.2 | ⬜ 待實作 |
| 3.5 | Training Tips 整合（隱性） | 上述 | 上述 | E1→E2 對比 | **hw3_3_ablation.png** | Ch4 消融 | ⬜ 待實作 |
| 3.6 | PER 實作（加分） | `src/buffers/per_buffer.py` | e3_per.yaml | E3 | hw3_3_e3_vs_e2.png | Ch4.3 | ⬜ 待實作 |
| 3.7 | Epsilon Decay Tuning（加分） | `src/utils/epsilon_scheduler.py` | e3_per.yaml | E3 | hw3_3_epsilon_comparison.png | Ch4.3 | ⬜ 待實作 |
| 3.8 | Reward/Loss 視覺化（隱性） | `src/plotting/plot_utils.py` | — | E1, E2, E3 | 各曲線圖 | Ch4 | ⬜ 待實作 |

**HW3-3 小計：8 項需求 → 8 個對應**

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
