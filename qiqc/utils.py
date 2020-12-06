import importlib
import os
import random
import shutil
import sys
from pathlib import Path
import torch
import numpy as np
import prompter

def load_module(filename):
    assert isinstance(filename, Path)
    name = filename.stem
    # pathlib.PosixPath convert into str format
    spec = importlib.util.spec_from_file_location(name, str(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[mod.__name__] = mod
    return mod

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


