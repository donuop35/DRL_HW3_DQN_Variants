# CHG-004: HW3-1 Static Mode — Basic DQN Implementation

**Change ID**: CHG-004
**建立日期**: 2026-05-13
**作者**: Tony Lo (via Antigravity)
**狀態**: In Progress 🔄

---

## Scope

### New Files
- `src/envs/Gridworld.py` + `src/envs/GridBoard.py`（教授原始檔）
- `src/envs/gridworld.py`（封裝層）
- `src/buffers/replay_buffer.py`（Experience Replay Buffer）
- `src/models/dqn.py`（Q-Network）
- `src/agents/dqn_agent.py`（DQN Agent）
- `src/training/train_dqn.py`（訓練迴圈）
- `configs/hw3_1_static/basic_dqn_static.yaml`
- `scripts/run_hw3_1_static.py`
- `report/understanding_report.md`（完整 HW3-1 段落）

### Foolproof Rules
- 教授 starter code 不得修改（保留於 notebooks/starter/）
- understanding_report.md 不得遺漏
- 所有結果從真實訓練產生，不得使用 smoke test 圖
- E4 不得被此 change 觸及
