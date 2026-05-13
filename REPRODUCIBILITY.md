# REPRODUCIBILITY.md
# HW3 DQN Variants — 完整重現指南

> 遵循 EXPERIMENT_PROTOCOL.md 的數據誠信規則：所有結果均可由本指南重現。

---

## 環境規格

| 項目 | 版本 |
|------|------|
| Python | 3.9 |
| PyTorch | 2.0+ |
| PyTorch Lightning | 2.6.0 |
| NumPy | ≥1.24 |
| Pandas | ≥1.5 |
| Matplotlib | ≥3.7 |
| PyYAML | ≥6.0 |
| OS | macOS 14.x / Ubuntu 22.04（測試環境）|

---

## 安裝

```bash
git clone https://github.com/donuop35/DRL_HW3_DQN_Variants.git
cd DRL_HW3_DQN_Variants
pip install -r requirements.txt
```

驗證安裝：
```bash
python scripts/smoke_test.py
# 預期輸出：11/11 PASSED
```

---

## 固定隨機種子

所有實驗使用 `seed=42`，設定方式：

```python
from src.utils.seeding import set_global_seed
set_global_seed(42)
# 同時設定：random, numpy, torch, torch.cuda
```

YAML config 中均有 `seed: 42`。

---

## 實驗重現指令

### HW3-1 Static Mode

```bash
python scripts/run_hw3_1_static.py
```

**預期輸出檔案**：
- `results/csv/hw3_1_static_basic_dqn_log.csv`（5000 rows）
- `results/figures/hw3_1_static_basic_dqn_{reward,loss,win_rate,steps,epsilon}.png`
- `results/checkpoints/hw3_1_static_basic_dqn/final_model.pt`

**預期關鍵指標**：
- Final Eval Win Rate：~100%
- Last-500ep Win Rate：~98.6%
- 訓練時間：~230 秒

### HW3-2 Player Mode

```bash
python scripts/run_hw3_2_player.py
```

**預期輸出**：
- `results/csv/hw3_2_player_{basic,double,dueling}_dqn_log.csv`（各 5000 rows）
- `results/figures/hw3_2_player_*_comparison.png`（5 張比較圖）

**預期關鍵指標**：
- P1/P2/P3 Final Win Rate：均 ~100%
- P2 Double DQN last-500 win rate：~100%

### HW3-3 Random Mode E1/E2/E3

```bash
python scripts/run_hw3_3_random.py
```

**預期輸出**：
- `results/csv/hw3_3_random_e{1,2,3}_*_log.csv`（各 5000 rows）
- 7 張比較圖（`hw3_3_random_*_comparison_e1_e2_e3.png`）

**預期關鍵指標**：
- E1 Final Win Rate：~91.5%
- E2 Final Win Rate：~88.5%
- E3 Final Win Rate：~90.0%（全體 win rate 最高：~85.2%）

### HW3-3 E4 Rainbow Bonus

```bash
python scripts/run_hw3_3_rainbow_bonus.py
```

**預期輸出**：
- `results/csv/hw3_3_random_e4_rainbow_bonus_log.csv`（5000 rows）
- 4 張 E4 個別圖 + 5 張 E1-E4 比較圖

**預期關鍵指標**：
- Final Win Rate：~40%（C51 在 5000ep 內收斂不足）
- E1-E3 CSV 不被修改（integrity check 通過）

---

## 驗證所有結果

```bash
python - << 'EOF'
import pandas as pd
from pathlib import Path

exps = [
    "hw3_1_static_basic_dqn",
    "hw3_2_player_basic_dqn", "hw3_2_player_double_dqn", "hw3_2_player_dueling_dqn",
    "hw3_3_random_e1_baseline", "hw3_3_random_e2_stabilized",
    "hw3_3_random_e3_per_stabilized", "hw3_3_random_e4_rainbow_bonus",
]
for exp in exps:
    p = Path(f"results/csv/{exp}_log.csv")
    df = pd.read_csv(p)
    print(f"{'✅' if len(df)==5000 else '❌'} {exp}: rows={len(df)}, NaN={df.isna().sum().sum()}")
EOF
```

---

## 已知限制

| 限制 | 說明 |
|------|------|
| E4 Rainbow 在小環境收斂慢 | C51 需要更大 buffer + 更多 episodes 才能超越 E1-E3 |
| E2 LR 衰減過快 | StepLR per-step 使後期 lr~0.000004，是設計取捨而非錯誤 |
| macOS 限定測試 | `torch.backends.mps` 未使用（訓練在 CPU 上）|
| 訓練時間 | E4 Rainbow 約 2500 秒，其他各組 ~108-400 秒 |
| GridWorld 相依 | 使用教授提供的 `Gridworld.py`（原始 starter code 保留不改）|
