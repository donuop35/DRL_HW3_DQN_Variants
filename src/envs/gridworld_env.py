"""
src/envs/gridworld_env.py
=========================
GridWorld 環境封裝層（命名為 gridworld_env 以避免與 Gridworld.py 大小寫衝突）。

封裝教授提供的原始 Gridworld.py / GridBoard.py，
提供統一的 reset() / step() API 供所有 DQN agents 使用。

三種模式：
    - static : 所有物件位置固定（HW3-1）
    - player : Player 隨機，其餘固定（HW3-2）
    - random : 所有物件隨機（HW3-3）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# 確保 Gridworld.py / GridBoard.py 可被 import（絕對路徑）
_ENV_DIR = str(Path(__file__).resolve().parent)
if _ENV_DIR not in sys.path:
    sys.path.insert(0, _ENV_DIR)

from Gridworld import Gridworld as _Gridworld  # noqa

# ──────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────

ACTION_SET   = {0: "u", 1: "d", 2: "l", 3: "r"}
ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

REWARD_GOAL  =  10.0
REWARD_PIT   = -10.0
REWARD_STEP  =  -1.0

STATE_DIM   = 64
N_ACTIONS   = 4
NOISE_SCALE = 0.01   # 對應 starter code /100


class GridworldEnv:
    """
    GridWorld 環境封裝器。

    使用範例::

        env = GridworldEnv(mode="static")
        state = env.reset()                  # shape (64,) float32
        next_state, reward, done, info = env.step(0)   # action=0 (up)
    """

    def __init__(
        self,
        mode: str = "static",
        size: int = 4,
        noise_scale: float = NOISE_SCALE,
    ) -> None:
        self.mode        = mode
        self.size        = size
        self.noise_scale = noise_scale
        self._game: _Gridworld = None
        self.reset()

    def reset(self) -> np.ndarray:
        """重設環境，回傳初始狀態 shape (64,) float32。"""
        self._game = _Gridworld(size=self.size, mode=self.mode)
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """執行動作，回傳 (next_state, reward, done, info)。"""
        self._game.makeMove(ACTION_SET[action])
        next_state = self._get_state()
        reward     = float(self._game.reward())
        done       = (reward != REWARD_STEP)
        terminal   = ("goal" if reward == REWARD_GOAL
                      else ("pit" if reward <= REWARD_PIT else "step"))
        return next_state, reward, done, {"terminal_state": terminal}

    def display(self) -> str:
        return str(self._game.display())

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def _get_state(self) -> np.ndarray:
        raw = self._game.board.render_np().reshape(64).astype("float32")
        if self.noise_scale > 0:
            raw += (np.random.rand(64) * self.noise_scale).astype("float32")
        return raw
