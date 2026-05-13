# EXPERIMENT PROTOCOL — DRL HW3: DQN and its Variants

> **文件版本**：v1.0
> **建立日期**：2026-05-13
> **作者**：Tony Lo（via Antigravity）
> **效力**：本文件規定所有正式實驗的執行規範，具約束力

---

## 1. 實驗完整性規則（Integrity Rules）

### 1.1 數據真實性

> **所有正式實驗數據必須來自真實執行，不得手動修改 CSV 數據。**

- ❌ 禁止：手動編輯 `results/csv/*.csv` 中的任何數值
- ❌ 禁止：在未實際執行的情況下建立 `*_log.csv` 檔案
- ❌ 禁止：複製另一個實驗的 CSV 數據並改名
- ✅ 允許：修改 config YAML 重新跑實驗（會覆蓋舊 CSV）
- ✅ 允許：使用 `smoke_test=True` 的 placeholder 圖，但必須標記

### 1.2 圖表生成規則

> **所有報告圖表必須由 `results/csv/` 中的真實 CSV 生成。**

```bash
# 正確做法：從 CSV 生成圖表
python scripts/generate_report_assets.py

# 錯誤做法：手動用 matplotlib 繪製假資料
```

- ❌ 禁止：在報告中嵌入未來自真實 CSV 的圖表
- ❌ 禁止：在報告中只說「詳見 results/figures/xxx.png」（必須嵌入）
- ✅ 允許：在 understanding_report 草稿階段先留 placeholder，但最終版必須換成真實結果

### 1.3 引用規則

> **報告只能引用真實跑出的結果。**

若模型表現不佳，必須誠實呈現並分析原因，不得隱瞞或美化。

---

## 2. 實驗 ID 鎖定規則（Experiment ID Lock Rules）

### 2.1 固定實驗 ID

以下實驗 ID **不得更名、不得刪除、不得重新定義**：

| 實驗 ID | 說明 | Config 路徑 |
|---------|------|------------|
| `hw3_1_static_naive_dqn` | HW3-1 正式實驗 | `configs/hw3_1_static/default.yaml` |
| `hw3_2_player_naive_dqn` | HW3-2 對照組 | `configs/hw3_2_player/naive_dqn.yaml` |
| `hw3_2_player_double_dqn` | HW3-2 Double DQN | `configs/hw3_2_player/double_dqn.yaml` |
| `hw3_2_player_dueling_dqn` | HW3-2 Dueling DQN | `configs/hw3_2_player/dueling_dqn.yaml` |
| `hw3_3_E1_random_baseline` | HW3-3 E1（正式主線） | `configs/hw3_3_random/e1_baseline.yaml` |
| `hw3_3_E2_stabilized` | HW3-3 E2（正式主線） | `configs/hw3_3_random/e2_stabilized.yaml` |
| `hw3_3_E3_per_stabilized` | HW3-3 E3（正式主線） | `configs/hw3_3_random/e3_per.yaml` |
| `hw3_3_E4_rainbow_bonus` | Bonus E4（不影響 E1-E3） | `configs/hw3_3_random/e4_rainbow.yaml` |

### 2.2 E4 隔離規則

> **E4 Rainbow Bonus 不得修改、取代、或破壞 E1～E3 的任何內容。**

- E4 使用獨立的 config（`e4_rainbow.yaml`）
- E4 輸出到獨立的 CSV（`hw3_3_E4_rainbow_bonus_log.csv`）
- E4 圖表在報告中獨立一章（Ch5 Bonus），不插入 Ch4（HW3-3 正式結果）

---

## 3. Config 管理規則

### 3.1 超參數修改流程

若需修改正式實驗的超參數：

1. 在 `configs/` 對應目錄下**建立新的 yaml 檔案**（如 `default_v2.yaml`）
2. 不得直接修改已使用過的 yaml（已有對應 CSV 的 config 不得改動）
3. 以新 yaml 執行新實驗，對比結果後決定是否替換

### 3.2 Seed 一致性

> 所有正式實驗使用 `seed: 42`，不得更改。

若要做 seed sensitivity 分析（非必需），使用 seed 43、44、45，並在報告中說明。

---

## 4. 圖表命名規範

### 4.1 命名規則

```
results/figures/<experiment_id>_<figure_type>.png
```

範例：
```
hw3_1_static_naive_dqn_reward.png
hw3_2_reward_comparison.png          ← 三方比較圖
hw3_3_reward_ablation.png            ← 消融實驗圖
hw3_3_E4_rainbow_bonus_reward.png    ← Bonus 圖
```

### 4.2 禁止的圖表

- Smoke test 圖（有 "SMOKE TEST" 浮水印）不得出現在正式報告中
- 手繪或 Excel 圖表不得出現

---

## 5. Logging 規則

### 5.1 必須記錄的 Episode 欄位

所有實驗必須透過 `ExperimentLogger` 記錄（不得自訂格式）：

```python
from src.utils.logger import ExperimentLogger

with ExperimentLogger(cfg) as logger:
    for ep in range(cfg.training.episodes):
        # ...訓練...
        logger.log_episode(
            episode=ep,
            episode_reward=total_reward,
            episode_steps=steps,
            loss_mean=avg_loss,
            epsilon=epsilon,
            win=reached_goal,
            terminal_state=terminal,
            learning_rate=current_lr,
            buffer_size=buffer_len,
        )
```

### 5.2 CSV 儲存路徑

```
results/csv/<experiment_id>_log.csv
```

### 5.3 禁止的行為

- ❌ 只記錄部分欄位
- ❌ 使用 `print()` 取代 logger
- ❌ 記錄後刪除欄位再報告

---

## 6. 實驗執行流程（Standard Operating Procedure）

### 6.1 新實驗執行 SOP

```bash
# Step 1：確認 smoke test 通過
python scripts/smoke_test.py

# Step 2：執行實驗
python scripts/run_hw3_1_static.py  # 或其他腳本

# Step 3：驗證 CSV 輸出
python -c "
import pandas as pd
df = pd.read_csv('results/csv/hw3_1_static_naive_dqn_log.csv')
print(f'Episodes: {len(df)}')
print(df.tail())
"

# Step 4：生成圖表
python scripts/generate_report_assets.py --experiment hw3_1_static_naive_dqn

# Step 5：嵌入圖表至報告
# 手動在 report/understanding_report.md 和 report/HW3_DQN_Variants_研究型實驗報告.md 中嵌入
```

### 6.2 驗收 Checklist（每個實驗完成後）

- [ ] CSV 存在且行數 = epochs 數（允許 ±10%）
- [ ] 圖表已生成（非 smoke test 版本）
- [ ] 圖表已嵌入對應報告章節
- [ ] config snapshot 已儲存於 `results/checkpoints/<experiment_id>/config_snapshot.csv`

---

## 7. 報告撰寫規則

### 7.1 圖表嵌入格式

```markdown
<!-- 正確：使用 markdown 圖片語法嵌入 -->
![HW3-1 訓練 Loss 曲線](../results/figures/hw3_1_static_naive_dqn_loss.png)
*圖 2.1：HW3-1 Naive DQN 訓練 Loss 曲線（Static Mode，5000 episodes）*

<!-- 錯誤：只放路徑 -->
詳見 results/figures/hw3_1_static_naive_dqn_loss.png
```

### 7.2 數據引用格式

```markdown
<!-- 正確：引用真實數值 -->
最終 100 episodes 的平均勝率為 **XX.X%**（見表 2.1）。

<!-- 錯誤：泛稱無數據支撐 -->
模型表現良好，學習速度很快。
```

---

*本文件由 Antigravity 生成於 2026-05-13，適用於 DRL HW3 所有實驗。*
