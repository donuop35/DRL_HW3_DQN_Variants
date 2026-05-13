# CHG-003: Experiment Harness + Logging Pipeline

**Change ID**: CHG-003
**建立日期**: 2026-05-13
**作者**: Tony Lo (via Antigravity)
**狀態**: In Progress 🔄
**對應分支**: `feat/experiment-harness`

---

## Change Summary

Phase 3：建立統一實驗基礎設施。

包含 config system、logging schema、evaluation metrics、plotting pipeline、
smoke test，確保 HW3-1/2/3 所有實驗共用同一套輸出格式。

---

## Scope

### New Files

**Utils**
- `src/utils/seeding.py`
- `src/utils/config.py`
- `src/utils/logger.py`

**Evaluation**
- `src/evaluation/metrics.py`

**Plotting**
- `src/plotting/plot_curves.py`
- `src/plotting/plot_comparison.py`

**Configs（所有 HW 的 YAML）**
- `configs/hw3_1_static/default.yaml`
- `configs/hw3_2_player/naive_dqn.yaml`
- `configs/hw3_2_player/double_dqn.yaml`
- `configs/hw3_2_player/dueling_dqn.yaml`
- `configs/hw3_3_random/e1_baseline.yaml`
- `configs/hw3_3_random/e2_stabilized.yaml`
- `configs/hw3_3_random/e3_per.yaml`
- `configs/hw3_3_random/e4_rainbow.yaml`

**Protocol & Tests**
- `EXPERIMENT_PROTOCOL.md`
- `scripts/smoke_test.py`

**Init files**
- `src/__init__.py`
- `src/utils/__init__.py`
- `src/evaluation/__init__.py`
- `src/plotting/__init__.py`

---

## Foolproof Rules（本 CHG 強制）

1. 所有 HW 實驗模組必須接入此 harness
2. 不得各自定義輸出格式
3. Smoke test 必須通過後才能進入正式訓練
4. E4 不得修改 E1-E3 configs
