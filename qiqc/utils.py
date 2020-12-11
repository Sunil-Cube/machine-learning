import importlib
import os
import random
import shutil
import sys
from pathlib import Path
import torch
import numpy as np
import prompter

from _qiqc.utils import *  # NOQA

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

def rmtree_after_confirmation(path, force=False):
    if Path(path).exists():
        if not force and not prompter.yesno('Overwrite %s?' % path):
            sys.exit(0)
        else:
            shutil.rmtree(path)

def pad_sequence(xs, length, padding_value=0):
    assert isinstance(xs, list)
    n_padding = length - len(xs)
    return np.array(xs + [padding_value] * n_padding, 'i')[:length]

class Pipeline(object):

    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x
