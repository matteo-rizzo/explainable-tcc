import re

import numpy as np
import torch

# --- Device (cpu or cuda:n) ---

DEVICE_TYPE = "cuda:0"


def get_device() -> torch.device:
    if DEVICE_TYPE == "cpu":
        print("\n Running on device 'cpu' \n")
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", DEVICE_TYPE):
        if not torch.cuda.is_available():
            print("\n WARNING: running on cpu since device {} is not available \n".format(DEVICE_TYPE))
            return torch.device("cpu")

        print("\n Running on device '{}' \n".format(DEVICE_TYPE))
        return torch.device(DEVICE_TYPE)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(DEVICE_TYPE))


DEVICE = get_device()


# --- Determinism (for reproducibility) ---

def make_deterministic(random_seed: int = 0):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.benchmark = False


# The ids of the sequences to be monitored at training time (e.g., ["0", "1", "2"])
TEST_VIS_IMG = []