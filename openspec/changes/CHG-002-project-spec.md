# CHG-002: Project Spec + Requirements Analysis

**Change ID**: CHG-002
**建立日期**: 2026-05-13
**作者**: Tony Lo (via Antigravity)
**狀態**: Applied ✅
**對應分支**: `docs/phase2-project-spec`

---

## Change Summary

Phase 2：完整讀取作業需求 + 建立 Project Spec。
逐字閱讀 starter code（51 cells）+ 教授需求，建立可執行規格文件。

---

## Motivation

Phase 1 建立骨架，但缺乏：
1. 基於 starter code 的精確技術規格
2. 鎖定的實驗 ID（E1～E4）和機制標識（S1～S5）
3. 教授需求到實作的完整追蹤矩陣
4. 防呆規則（E4 不得破壞 E1～E3）

---

## Changes Applied

### 文件（規格層）
- ✅ `ASSIGNMENT_REQUIREMENTS.md` v2.0（逐字解析 starter code，含超參數）
- ✅ `EXPERIMENT_STORYLINE.md` v2.0（連貫敘事，非三份斷裂作業）
- ✅ `REQUIREMENTS_TRACEABILITY_MATRIX.md`（教授需求 → 代碼 → 報告對照）

### OpenSpec（規格執行層）
- ✅ `openspec/PROJECT_SPEC.md`（SPEC-00 ～ SPEC-08，含防呆鎖定）

### 鎖定項目
- 機制標識：S1, S2, S3, S4, S5（不得更名）
- 實驗 ID：E1, E2, E3, E4（不得更名，E4 不得破壞 E1～E3）
- Logging Schema（SPEC-05）
- 圖表命名規範（SPEC-06）

---

## 下一步：CHG-003

Phase 3：Experiment Harness 建設
- `src/utils/seed.py`
- `src/utils/config.py`
- `src/utils/logger.py`
- `src/envs/gridworld.py`
- `src/buffers/replay_buffer.py`
- `src/models/dqn_net.py`
- `src/agents/naive_dqn.py`
- `src/training/trainer.py`
- `configs/hw3_1_static/default.yaml`
- `scripts/run_hw3_1_static.py`
