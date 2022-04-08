import os
import torch
import numpy as np
import random
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
