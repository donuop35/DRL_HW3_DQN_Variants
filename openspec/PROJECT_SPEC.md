# PROJECT SPEC — DRL HW3: DQN and its Variants
# OpenSpec Project Specification（Phase 2 正式版）

> **文件版本**：v2.0
> **建立日期**：2026-05-13
> **作者**：Tony Lo（via Antigravity）
> **狀態**：Locked ✅（本文件一旦鎖定，所有實作必須遵循）

---

## SPEC-00：研究主線（Research Backbone）

本專案的核心研究主線固定為：

```
Basic DQN
→ [S1] Experience Replay + Sample Reuse
→ [S2] Target Stabilization / Training Stabilization
→ [S3] Overestimation Reduction（Double DQN）
→ [S4] Value–Advantage Decomposition（Dueling DQN）
→ [S5] Prioritized Sampling + Exploration Control（PER + Epsilon Tuning）
→ [ALL] Advanced Rainbow DQN Bonus Pipeline
```

### 機制命名鎖定（不得更名）

| 標識 | 機制名稱 | 論文來源 | 對應實作 |
|------|---------|---------|---------|
| S1 | Experience Replay / Sample Reuse | DeepMind DQN (Mnih 2015) | `ReplayBuffer` |
| S2 | Target Stabilization / Training Stabilization | DeepMind DQN (Mnih 2015) | `TargetNetwork` |
| S3 | Overestimation Reduction | Double DQN (van Hasselt 2016) | `DoubleDQNAgent` |
| S4 | Value–Advantage Decomposition | Dueling DQN (Wang 2016) | `DuelingNet` |
| S5 | Prioritized Sampling + Exploration Control | PER (Schaul 2016) + ε-Greedy | `PERBuffer` + `EpsilonScheduler` |

---

## SPEC-01：環境規格（Environment Specification）

### SPEC-01.1：GridWorld 介面

```python
game = Gridworld(size=4, mode='static'|'player'|'random')

# 狀態表示
state = game.board.render_np()  # shape: (4, 4, 4)
state_flat = state.reshape(1, 64)  # shape: (1, 64)，作為網路輸入

# 動作
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
game.makeMove(action_set[action_idx])

# 獎勵
reward = game.reward()  # +10 (goal), -10 (pit), -1 (step)

# 撞牆檢測（選用）
hit_wall = game.validateMove('Player', move_pos[action_idx]) == 1
```

### SPEC-01.2：三種模式配置

```yaml
# Static Mode（HW3-1）
mode: static
player_pos: [0, 3]
goal_pos: [1, 0]
pit_pos: [0, 1]
wall_pos: [1, 1]

# Player Mode（HW3-2）
mode: player
player_pos: random
goal_pos: [1, 0]
pit_pos: [0, 1]
wall_pos: [1, 1]

# Random Mode（HW3-3）
mode: random
player_pos: random
goal_pos: random
pit_pos: random
wall_pos: random
```

### SPEC-01.3：獎勵規格

```python
REWARD_GOAL = +10    # 到達目標
REWARD_PIT = -10     # 掉入陷阱
REWARD_STEP = -1     # 每步懲罰
REWARD_WALL = -5     # 撞牆懲罰（可選，改良版程式 3.5）
```

---

## SPEC-02：網路架構規格（Model Specification）

### SPEC-02.1：基礎 Q-Network

```python
# Baseline Architecture（對應 starter code 預設）
input_dim = 64      # 4x4x4 flattened
hidden_1 = 150
hidden_2 = 100
output_dim = 4      # 4 actions

model = nn.Sequential(
    nn.Linear(64, 150),
    nn.ReLU(),
    nn.Linear(150, 100),
    nn.ReLU(),
    nn.Linear(100, 4)
)
```

### SPEC-02.2：Dueling Network 架構

```python
class DuelingNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=150, n_actions=4):
        super().__init__()
        # 共享特徵層
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions)
        )

    def forward(self, x):
        feat = self.feature(x)
        V = self.value_stream(feat)
        A = self.advantage_stream(feat)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
```

---

## SPEC-03：超參數規格（Hyperparameter Specification）

### SPEC-03.1：基礎超參數（Starter Code 基準值）

```yaml
# 必須記錄的基礎超參數
network:
  input_dim: 64
  hidden_1: 150
  hidden_2: 100
  output_dim: 4

training:
  epochs: 5000
  max_moves: 50
  batch_size: 200
  gamma: 0.9
  learning_rate: 1.0e-3
  optimizer: Adam
  loss: MSELoss

replay_buffer:
  mem_size: 1000
  batch_size: 200

target_network:
  sync_freq: 500

epsilon:
  start: 1.0
  end: 0.1
  decay: linear  # 1/epochs per episode

seed: 42
```

### SPEC-03.2：HW3-3 Training Tips 超參數

```yaml
# Gradient Clipping
gradient_clip_val: 1.0

# LR Scheduling
lr_scheduler:
  type: StepLR
  step_size: 500
  gamma: 0.9

# Epsilon（Exponential Decay）
epsilon:
  start: 1.0
  end: 0.01
  decay_type: exponential
  decay_rate: 500  # steps

# PER
per:
  alpha: 0.6    # 優先級指數
  beta_start: 0.4   # 重要性採樣初始值
  beta_end: 1.0     # 退火終點
  epsilon: 1.0e-5   # 防止優先級為零
```

---

## SPEC-04：實驗規格（Experiment Specification）

### SPEC-04.1：HW3-1 實驗（Static Mode）

```yaml
experiment_id: hw3_1_static_naive_dqn
hw: HW3-1
mode: static
agent: NaiveDQN  # with S1+S2
epochs: 5000
seed: 42
deliverables:
  - results/csv/hw3_1_training_log.csv
  - results/figures/hw3_1_loss_curve.png
  - results/figures/hw3_1_epsilon_decay.png
  - results/checkpoints/hw3_1_model.pt
  - report/understanding_report.md  # ← 最高優先
```

### SPEC-04.2：HW3-2 實驗（Player Mode）

```yaml
experiments:
  - id: hw3_2_player_naive_dqn
    agent: NaiveDQN
    mode: player
    role: baseline_for_hw3_2

  - id: hw3_2_player_double_dqn
    agent: DoubleDQN  # +S3
    mode: player

  - id: hw3_2_player_dueling_dqn
    agent: DuelingDQN  # +S4
    mode: player

deliverables:
  - results/figures/hw3_2_comparison_all.png  # 三條曲線同圖
  - results/figures/hw3_2_double_dqn_curve.png
  - results/figures/hw3_2_dueling_dqn_curve.png
```

### SPEC-04.3：HW3-3 實驗（Random Mode）— 鎖定不得更改

```yaml
# E1：正式主線 Baseline
experiment_id: hw3_3_E1_random_baseline
agent: NaiveDQN
mode: random
training_tips: none
framework: vanilla_pytorch

# E2：正式主線 Stabilized
experiment_id: hw3_3_E2_stabilized
agent: LightningDQN  # PyTorch Lightning
mode: random
training_tips:
  gradient_clipping: true
  lr_scheduling: true
framework: pytorch_lightning

# E3：正式主線 PER + Stabilization
experiment_id: hw3_3_E3_per_stabilized
agent: LightningDQN  # PyTorch Lightning
mode: random
training_tips:
  gradient_clipping: true
  lr_scheduling: true
  per: true
  epsilon_tuning: exponential_decay
framework: pytorch_lightning

# ==========================================
# E4：BONUS（不得取代或破壞 E1-E3）
# ==========================================
experiment_id: hw3_3_E4_rainbow_bonus
agent: RainbowDQN
mode: random
components:
  - double_dqn     # S3
  - dueling_net    # S4
  - per            # S5
  - multi_step_n3  # N=3
framework: pytorch_lightning
status: bonus_only
```

---

## SPEC-05：Logging Schema（統一記錄格式）

所有實驗必須使用相同的 logging schema：

```python
# CSV 格式（results/csv/<experiment_id>_log.csv）
columns = [
    'episode',        # int：訓練回合數
    'step',           # int：累計步數
    'reward',         # float：本回合總 Reward
    'loss',           # float：本 step 的 loss
    'epsilon',        # float：當前 epsilon 值
    'win',            # bool：是否到達 Goal
    'steps_taken',    # int：本回合用了幾步
    'q_value_mean',   # float：本 step 的平均 Q 值（選填）
    'lr',             # float：當前學習率
    'timestamp'       # str：ISO8601
]

# 評估指標（evaluation）
eval_metrics = {
    'win_rate': float,       # 1000 場測試的勝率
    'avg_reward': float,     # 平均 reward
    'avg_steps': float,      # 平均步數
    'convergence_episode': int  # 收斂所需 episode
}
```

---

## SPEC-06：圖表規格（Plotting Specification）

```python
# 標準圖表要求
FIGURE_FORMAT = 'png'
FIGURE_DPI = 150
FIGURE_SIZE = (10, 7)

# 必要圖表清單
required_figures = {
    # HW3-1
    'hw3_1_loss_curve': 'Loss vs Steps',
    'hw3_1_epsilon_decay': 'Epsilon vs Episode',

    # HW3-2
    'hw3_2_comparison_all': '三方法 Reward 比較（同圖）',
    'hw3_2_q_value_comparison': 'Q 值估計比較',

    # HW3-3
    'hw3_3_e1_loss': 'E1 Baseline Loss Curve',
    'hw3_3_e2_vs_e1': 'E2 vs E1 穩定性比較',
    'hw3_3_e3_vs_e2': 'E3 vs E2 PER 效果',
    'hw3_3_ablation': 'Training Tips 消融實驗',

    # Bonus
    'hw3_3_e4_rainbow': 'E4 Rainbow vs E3',
}

# 防呆規則：圖表必須從真實 CSV 生成，不得手工製作
```

---

## SPEC-07：報告規格（Report Specification）

### SPEC-07.1：understanding_report.md（HW3-1 必繳）

```yaml
file: report/understanding_report.md
language: 繁體中文（英文專業名詞可保留）
required_sections:
  - DQN 原理理解（自己的話）
  - GridWorld 環境分析
  - Starter Code 逐段解析
  - 實驗結果（含嵌入圖表）
  - 個人見解與討論
embedded_figures: true  # 圖表必須嵌入，不可只放路徑
```

### SPEC-07.2：研究型實驗報告

```yaml
file: report/HW3_DQN_Variants_研究型實驗報告.md
language: 繁體中文（英文專業名詞可保留）
required_sections:
  - 摘要
  - 實驗設置（環境、硬體、軟體版本）
  - HW3-1 結果
  - HW3-2 結果與比較
  - HW3-3 E1-E3 結果與消融
  - Bonus E4 Rainbow 結果
  - 結論
  - 參考文獻
embedded_figures: true  # 圖表必須嵌入
```

---

## SPEC-08：防呆鎖定規則（Foolproof Rules）

1. **不得假造實驗結果**：所有圖表由 `scripts/generate_report_assets.py` 從 CSV 生成
2. **E1～E3 不得被 E4 影響**：Bonus 必須在獨立分支實作，不修改主線代碼
3. **理解報告不得省略**：`understanding_report.md` 是正式交付物
4. **圖表必須嵌入報告**：不接受「詳見 results/figures/xxx.png」的參考
5. **機制標識 S1～S5 不得更名**：在所有代碼、設定、報告中統一使用
6. **實驗 ID E1～E4 不得更名**：在所有設定、CSV、圖表、報告中統一使用
7. **Logging Schema 統一**：所有實驗使用 SPEC-05 定義的格式
8. **所有超參數在 yaml 中管理**：不得寫死在 Python 程式碼中
