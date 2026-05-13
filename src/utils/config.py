"""
src/utils/config.py
===================
統一 Config 載入與驗證。

所有實驗超參數必須透過 YAML config 檔案管理，
不得寫死在訓練程式碼中（對應 SPEC-08 防呆規則 8）。

使用範例：
    cfg = load_config("configs/hw3_1_static/default.yaml")
    print(cfg.experiment_id)
    print(cfg.training.episodes)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ──────────────────────────────────────────────
# Dataclass 定義（型別安全的 config 存取）
# ──────────────────────────────────────────────

@dataclass
class NetworkConfig:
    input_dim: int = 64
    hidden_1: int = 150
    hidden_2: int = 100
    output_dim: int = 4


@dataclass
class TrainingConfig:
    episodes: int = 5000
    max_steps_per_episode: int = 50
    gamma: float = 0.9
    learning_rate: float = 1e-3
    batch_size: int = 200
    replay_capacity: int = 1000
    target_update_frequency: int = 500
    # Gradient clipping
    use_gradient_clipping: bool = False
    max_grad_norm: float = 1.0
    # LR scheduler
    use_lr_scheduler: bool = False
    lr_scheduler_type: str = "StepLR"       # StepLR | CosineAnnealingLR
    lr_scheduler_step_size: int = 500
    lr_scheduler_gamma: float = 0.9
    # Lightning
    use_lightning: bool = False


@dataclass
class EpsilonConfig:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_type: str = "linear"      # linear | exponential
    epsilon_decay_steps: int = 5000


@dataclass
class AlgorithmConfig:
    use_target_network: bool = True
    use_double_dqn: bool = False
    use_dueling_dqn: bool = False
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_epsilon: float = 1e-5
    use_n_step: bool = False
    n_step: int = 3
    use_noisy_net: bool = False
    use_distributional: bool = False
    c51_atoms: int = 51
    c51_v_min: float = -10.0
    c51_v_max: float = 10.0


@dataclass
class ExperimentConfig:
    """頂層 config 物件，涵蓋所有超參數。"""
    experiment_id: str = "default"
    hw_part: str = "hw3_1"                  # hw3_1 | hw3_2 | hw3_3
    mode: str = "static"                    # static | player | random
    algorithm: str = "NaiveDQN"
    seed: int = 42

    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    epsilon: EpsilonConfig = field(default_factory=EpsilonConfig)
    algorithm_flags: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    # 輸出路徑（自動從 experiment_id 推導，可被 yaml 覆寫）
    log_dir: str = ""
    checkpoint_dir: str = ""
    figures_dir: str = ""

    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = f"results/csv"
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"results/checkpoints/{self.experiment_id}"
        if not self.figures_dir:
            self.figures_dir = f"results/figures"


# ──────────────────────────────────────────────
# 載入函式
# ──────────────────────────────────────────────

def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """將巢狀 dict 攤平為 dot-notation dict。"""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    從 YAML 檔案載入 ExperimentConfig。

    Args:
        config_path: YAML 設定檔路徑

    Returns:
        ExperimentConfig 物件

    Raises:
        FileNotFoundError: 若 config 檔案不存在
        ValueError: 若 config 格式有誤
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg = ExperimentConfig(
        experiment_id=raw.get("experiment_id", "default"),
        hw_part=raw.get("hw_part", "hw3_1"),
        mode=raw.get("mode", "static"),
        algorithm=raw.get("algorithm", "NaiveDQN"),
        seed=raw.get("seed", 42),
    )

    # 解析各子區塊
    net_raw = raw.get("network", {})
    cfg.network = NetworkConfig(
        input_dim=net_raw.get("input_dim", 64),
        hidden_1=net_raw.get("hidden_1", 150),
        hidden_2=net_raw.get("hidden_2", 100),
        output_dim=net_raw.get("output_dim", 4),
    )

    tr_raw = raw.get("training", {})
    cfg.training = TrainingConfig(
        episodes=tr_raw.get("episodes", 5000),
        max_steps_per_episode=tr_raw.get("max_steps_per_episode", 50),
        gamma=tr_raw.get("gamma", 0.9),
        learning_rate=tr_raw.get("learning_rate", 1e-3),
        batch_size=tr_raw.get("batch_size", 200),
        replay_capacity=tr_raw.get("replay_capacity", 1000),
        target_update_frequency=tr_raw.get("target_update_frequency", 500),
        use_gradient_clipping=tr_raw.get("use_gradient_clipping", False),
        max_grad_norm=tr_raw.get("max_grad_norm", 1.0),
        use_lr_scheduler=tr_raw.get("use_lr_scheduler", False),
        lr_scheduler_type=tr_raw.get("lr_scheduler_type", "StepLR"),
        lr_scheduler_step_size=tr_raw.get("lr_scheduler_step_size", 500),
        lr_scheduler_gamma=tr_raw.get("lr_scheduler_gamma", 0.9),
        use_lightning=tr_raw.get("use_lightning", False),
    )

    ep_raw = raw.get("epsilon", {})
    cfg.epsilon = EpsilonConfig(
        epsilon_start=ep_raw.get("epsilon_start", 1.0),
        epsilon_end=ep_raw.get("epsilon_end", 0.1),
        epsilon_decay_type=ep_raw.get("epsilon_decay_type", "linear"),
        epsilon_decay_steps=ep_raw.get("epsilon_decay_steps", 5000),
    )

    al_raw = raw.get("algorithm_flags", {})
    cfg.algorithm_flags = AlgorithmConfig(
        use_target_network=al_raw.get("use_target_network", True),
        use_double_dqn=al_raw.get("use_double_dqn", False),
        use_dueling_dqn=al_raw.get("use_dueling_dqn", False),
        use_per=al_raw.get("use_per", False),
        per_alpha=al_raw.get("per_alpha", 0.6),
        per_beta_start=al_raw.get("per_beta_start", 0.4),
        per_beta_end=al_raw.get("per_beta_end", 1.0),
        per_epsilon=al_raw.get("per_epsilon", 1e-5),
        use_n_step=al_raw.get("use_n_step", False),
        n_step=al_raw.get("n_step", 3),
        use_noisy_net=al_raw.get("use_noisy_net", False),
        use_distributional=al_raw.get("use_distributional", False),
        c51_atoms=al_raw.get("c51_atoms", 51),
        c51_v_min=al_raw.get("c51_v_min", -10.0),
        c51_v_max=al_raw.get("c51_v_max", 10.0),
    )

    # 路徑覆寫
    cfg.log_dir = raw.get("log_dir", f"results/csv")
    cfg.checkpoint_dir = raw.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_id}")
    cfg.figures_dir = raw.get("figures_dir", "results/figures")

    return cfg


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """將 ExperimentConfig 序列化為 dict（用於記錄到 CSV header）。"""
    return {
        "experiment_id": cfg.experiment_id,
        "hw_part": cfg.hw_part,
        "mode": cfg.mode,
        "algorithm": cfg.algorithm,
        "seed": cfg.seed,
        "episodes": cfg.training.episodes,
        "max_steps_per_episode": cfg.training.max_steps_per_episode,
        "gamma": cfg.training.gamma,
        "learning_rate": cfg.training.learning_rate,
        "batch_size": cfg.training.batch_size,
        "replay_capacity": cfg.training.replay_capacity,
        "target_update_frequency": cfg.training.target_update_frequency,
        "use_gradient_clipping": cfg.training.use_gradient_clipping,
        "max_grad_norm": cfg.training.max_grad_norm,
        "use_lr_scheduler": cfg.training.use_lr_scheduler,
        "use_lightning": cfg.training.use_lightning,
        "epsilon_start": cfg.epsilon.epsilon_start,
        "epsilon_end": cfg.epsilon.epsilon_end,
        "epsilon_decay_type": cfg.epsilon.epsilon_decay_type,
        "use_target_network": cfg.algorithm_flags.use_target_network,
        "use_double_dqn": cfg.algorithm_flags.use_double_dqn,
        "use_dueling_dqn": cfg.algorithm_flags.use_dueling_dqn,
        "use_per": cfg.algorithm_flags.use_per,
        "use_n_step": cfg.algorithm_flags.use_n_step,
        "n_step": cfg.algorithm_flags.n_step,
        "use_noisy_net": cfg.algorithm_flags.use_noisy_net,
        "use_distributional": cfg.algorithm_flags.use_distributional,
    }
