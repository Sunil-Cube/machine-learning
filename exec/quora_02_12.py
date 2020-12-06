import argparse
import json
import math
import os
import sys
import unidecode
import random
import re
import time
import yaml
from abc import ABCMeta, abstractmethod
from collections import defaultdict, Counter
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nltk
import gensim
import sklearn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors
from gensim.models import Word2Vec, Doc2Vec, FastText
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import sys
sys.path.append('/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/git_manage')

import qiqc
from qiqc.utils import set_seed, load_module

def load_qiqc(n_rows=None):
    train_df = pd.read_csv('/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/quora/train.csv', nrows=n_rows)
    submit_df = pd.read_csv('/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/quora/test.csv', nrows=n_rows)
    n_labels = {
        0: (train_df.target == 0).sum(),
        1: (train_df.target == 1).sum(),
    }
    train_df['target'] = train_df.target.astype('f')
    train_df['weights'] = train_df.target.apply(lambda t: 1 / n_labels[t])

    return train_df, submit_df

def build_datasets(train_df, submit_df, holdout=False, seed=0):
    submit_dataset = QIQCDataset(submit_df)
    if holdout:
        # Train : Test split for holdout training
        splitter = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=seed)
        train_indices, test_indices = list(splitter.split(
            train_df, train_df.target))[0]

        train_indices.sort(), test_indices.sort()

        train_dataset = QIQCDataset(
            train_df.iloc[train_indices].reset_index(drop=True))
        test_dataset = QIQCDataset(
            train_df.iloc[test_indices].reset_index(drop=True))
    else:
        train_dataset = QIQCDataset(train_df)
        test_dataset = QIQCDataset(train_df.head(0))

    return train_dataset, test_dataset, submit_dataset

class QIQCDataset(object):
    def __init__(self, df):
        self.df = df

    @property
    def tokens(self):
        return self.df.tokens.values

    @tokens.setter
    def tokens(self, tokens):
        self.df['tokens'] = tokens

    @property
    def positives(self):
        return self.df[self.df.target == 1]

    @property
    def negatives(self):
        return self.df[self.df.target == 0]




#############################################################


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfile', '-m', type=Path, required=True)
    _args, others = parser.parse_known_args(args)
    modules = load_module(_args.modelfile)
    config = modules.ExperimentConfigBuilder().build(args=[])
    print(config)
    # start = time.time()
    # set_seed(config.seed)
    #
    # train_df, submit_df = load_qiqc(n_rows=config.n_rows)
    # datasets = build_datasets(train_df, submit_df, config.holdout, config.seed)
    #
    # train_dataset, test_dataset, submit_dataset = datasets
    #
    # print('Tokenize texts...')
    # preprocessor = Preprocessor()

if __name__ == '__main__':
    main()
