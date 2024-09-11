import numpy as np
import os
import random


def init_random_seeds(seed: int = 42):
    """Seed all random generators and enforce deterministic algorithms to \
        guarantee reproducible results (may limit performance).

    Args:
        seed (int): The seed shared by all RNGs.
    """
    seed = seed % 2**32  # some only accept 32bit seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
