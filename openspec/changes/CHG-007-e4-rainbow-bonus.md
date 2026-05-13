# CHG-007: E4 Rainbow DQN Bonus — Advanced Pipeline

**Change ID**: CHG-007  
**建立日期**: 2026-05-13  
**狀態**: In Progress 🔄  
**類型**: Bonus Experiment（不影響 E1-E3 正式主線）

---

## Scope（僅新增，不修改任何 E1-E3 檔案）

### New Files
- `src/models/noisy_layers.py`（NoisyLinear）
- `src/models/c51_dueling_dqn.py`（C51 + Dueling Network）
- `src/buffers/nstep_per_buffer.py`（N-step + PER 合體）
- `src/agents/rainbow_dqn_agent.py`（Rainbow orchestration）
- `src/training/lightning_rainbow_module.py`（Lightning wrapper）
- `configs/hw3_3_random/e4_rainbow_dqn_bonus.yaml`
- `scripts/run_hw3_3_rainbow_bonus.py`

### Results (generated, NEW only)
- `results/csv/hw3_3_random_e4_rainbow_bonus_log.csv`
- 4 E4 individual figures
- 5 E1-E4 comparison figures

### Reports (additive only)
- `understanding_report.md` ← append Rainbow bonus 段落
- `HW3_DQN_Variants_研究型實驗報告.md` ← append Section 5 Bonus
- `REQUIREMENTS_TRACEABILITY_MATRIX.md` ← RTM-04 status update

### Constraints
- ❌ 不修改 E1/E2/E3 任何 config / CSV / figure
- ❌ 不讓 E4 取代 E3 的正式主線地位
- ✅ 誠實呈現 E4 結果（不捏造）
