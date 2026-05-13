# CHG-006: HW3-3 Random Mode — E1/E2/E3 Main Experiments

**Change ID**: CHG-006
**建立日期**: 2026-05-13
**作者**: Tony Lo (via Antigravity)
**狀態**: In Progress 🔄

---

## Scope

### New Files
- `src/buffers/prioritized_replay_buffer.py`（PER + SumTree，E3 使用）
- `src/training/lightning_dqn_module.py`（PyTorch Lightning Module，E2/E3 使用）
- `configs/hw3_3_random/e1_random_dqn_baseline.yaml`
- `configs/hw3_3_random/e2_stabilized_dqn.yaml`
- `configs/hw3_3_random/e3_per_dqn_stabilized.yaml`
- `scripts/run_hw3_3_random.py`

### Results (generated)
- `results/csv/hw3_3_random_e{1,2,3}_*.csv`
- 7 comparison figures

### Reports updated
- `report/understanding_report.md` ← HW3-3 正式段落
- `report/HW3_DQN_Variants_研究型實驗報告.md` ← 第 4 節

### Constraints
- 不實作 Rainbow DQN（E4 留給 Phase 7）
- E1/E2 不得使用 PER
- 不修改 HW3-1 / HW3-2 結果
