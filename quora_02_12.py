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

modules = """

class ExperimentConfigBuilder(ExperimentConfigBuilderBase):

    default_config = dict(
        test=False,
        device=0,
        maxlen=72,
        vocab_mincount=5,
        scale_batchsize=[],
        validate_from=4,
    )

    @property
    def modules(self):
        return [
            TextNormalizer,
            TextTokenizer,
            WordEmbeddingFeaturizer,
            WordExtraFeaturizer,
            SentenceExtraFeaturizer,
            Embedding,
            Encoder,
            Aggregator,
            MLP,
        ]


def build_model(config, embedding_matrix, n_sentence_extra_features):
    embedding = Embedding(config, embedding_matrix)
    encoder = Encoder(config, embedding.out_size)
    aggregator = Aggregator(config)
    mlp = MLP(config, encoder.out_size + n_sentence_extra_features)
    out = nn.Linear(config.mlp_n_hiddens[-1], 1)
    lossfunc = nn.BCEWithLogitsLoss()

    return BinaryClassifier(
        embedding=embedding,
        encoder=encoder,
        aggregator=aggregator,
        mlp=mlp,
        out=out,
        lossfunc=lossfunc,
    )


# =======  Preprocessing modules  =======

# class TextNormalizer(TextNormalizerPresets):
#     pass
# 
# 
# class TextTokenizer(TextTokenizerPresets):
#     pass
# 
# 
# class WordEmbeddingFeaturizer(WordEmbeddingFeaturizerPresets):
#     pass
# 
# 
# class WordExtraFeaturizer(WordExtraFeaturizerPresets):
# 
#     default_config = dict(
#         word_extra_features=['idf', 'unk'],
#     )
# 
# 
# class SentenceExtraFeaturizer(SentenceExtraFeaturizerPresets):
# 
#     default_config = dict(
#         sentence_extra_features=['char', 'word'],
#     )
# 
# 
class Preprocessor(PreprocessorPresets):

    embedding_sampling = 400

    def build_word_features(self, word_embedding_featurizer,
                            embedding_matrices, word_extra_features):
        embedding = np.stack(list(embedding_matrices.values()))

        # Concat embedding
        embedding = np.concatenate(embedding, axis=1)
        vocab = word_embedding_featurizer.vocab
        embedding[vocab.lfq & vocab.unk] = 0

        # Embedding random sampling
        n_embed = embedding.shape[1]
        n_select = self.embedding_sampling
        idx = np.random.permutation(n_embed)[:n_select]
        embedding = embedding[:, idx]

        word_features = np.concatenate(
            [embedding, word_extra_features], axis=1)
        return word_features

# =======  Training modules  =======

# class Embedding(EmbeddingPresets):
#     pass
# 
# 
# class Encoder(EncoderPresets):
#     pass
# 
# 
# class Aggregator(AggregatorPresets):
#     pass
# 
# 
# class MLP(MLPPresets):
#     pass
# 
# 
# class Ensembler(EnsemblerPresets):
#     pass

"""
class ExperimentConfigBuilderBase(metaclass=ABCMeta):

    default_config = None

    def add_args(self, parser):
        parser.add_argument('--modelfile', '-m', type=Path)
        parser.add_argument('--outdir-top', type=Path, default=Path('results'))
        parser.add_argument('--outdir-bottom', type=str, default='default')
        parser.add_argument('--device', '-g', type=int)
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--logging', action='store_true')
        parser.add_argument('--n-rows', type=int)

        parser.add_argument('--seed', type=int, default=1029)
        parser.add_argument('--optuna-trials', type=int)
        parser.add_argument('--gridsearch', action='store_true')
        parser.add_argument('--holdout', action='store_true')
        parser.add_argument('--cv', type=int, default=5)
        parser.add_argument('--cv-part', type=int)
        parser.add_argument('--processes', type=int, default=2)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--batchsize', type=int, default=512)
        parser.add_argument('--batchsize-valid', type=int, default=1024)
        parser.add_argument('--scale-batchsize', type=int, nargs='+',
                            default=[])
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--validate-from', type=int)
        parser.add_argument('--pos-weight', type=float, default=1.)
        parser.add_argument('--maxlen', type=float, default=72)
        parser.add_argument('--vocab-mincount', type=float, default=5)
        parser.add_argument('--ensembler-n-snapshots', type=int, default=1)

    @abstractmethod
    def modules(self):
        raise NotImplementedError()

    def build(self, args=None):
        assert self.default_config is not None
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        parser.set_defaults(**self.default_config)

        # for module in self.modules:
        #     module.add_args(parser)
        # config, extra_config = parser.parse_known_args(args)
        #
        # for module in self.modules:
        #     if hasattr(module, 'add_extra_args'):
        #         module.add_extra_args(parser, config)

        # if config.test:
        #     parser.set_defaults(**dict(
        #         n_rows=500,
        #         batchsize=64,
        #         validate_from=0,
        #         epochs=3,
        #         cv_part=2,
        #         ensembler_test_size=1.,
        #     ))

        config = parser.parse_args(args)
        if config.modelfile is not None:
            config.outdir = config.outdir_top / config.modelfile.stem \
                / config.outdir_bottom
        else:
            config.outdir = Path('.')

        return config

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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
exec(modules)
config = ExperimentConfigBuilder().build(args=[])
print(config)
start = time.time()
set_seed(config.seed)

train_df, submit_df = load_qiqc(n_rows=config.n_rows)
datasets = build_datasets(train_df, submit_df, config.holdout, config.seed)

train_dataset, test_dataset, submit_dataset = datasets

print('Tokenize texts...')
preprocessor = Preprocessor()

