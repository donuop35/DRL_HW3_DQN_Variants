"""
src/utils/seeding.py
====================
統一隨機種子管理。

所有實驗必須在最開頭呼叫 set_global_seed(seed)，
確保 Python / NumPy / PyTorch 的可重現性。
"""

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """
    固定所有隨機種子，確保實驗可重現。

    Args:
        seed: 隨機種子（預設 42，對應 SPEC-03.1）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 確保 cuDNN 使用確定性演算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 環境變數（某些 numpy / scipy 操作）
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seeding] Global seed set to {seed}")


def get_seed_from_config(cfg: dict) -> int:
    """
    從 config dict 中取得 seed，若無則回傳預設值 42。

    Args:
        cfg: 已載入的 config dict

    Returns:
        seed (int)
    """
    return int(cfg.get("seed", 42))
