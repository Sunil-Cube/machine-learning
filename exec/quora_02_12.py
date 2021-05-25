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

from qiqc.preprocessing.modules import load_pretrained_vectors
from qiqc.training import classification_metrics, ClassificationResult



#import pyximport; pyximport.install() #one way to build cpython


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

    def build(self, device):
        self._X = self.tids
        self.X = torch.Tensor(self._X).type(torch.long).to(device)
        if 'target' in self.df:
            self._t = self.df.target[:, None]
            self._W = self.df.weights
            self.t = torch.Tensor(self._t).type(torch.float).to(device)
            self.W = torch.Tensor(self._W).type(torch.float).to(device)
        if hasattr(self, '_X2'):
            self.X2 = torch.Tensor(self._X2).type(torch.float).to(device)
        else:
            self._X2 = np.zeros((self._X.shape[0], 1), 'f')
            self.X2 = torch.Tensor(self._X2).type(torch.float).to(device)

    def build_labeled_dataset(self, indices):
        return torch.utils.data.TensorDataset(
            self.X[indices], self.X2[indices],
            self.t[indices], self.W[indices])




#############################################################

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfile', '-m', type=Path, required=True)
    _args, others = parser.parse_known_args(args)
    modules = load_module(_args.modelfile) # load all file we can check using (step into code)
    config = modules.ExperimentConfigBuilder().build(args=[])
    print(config.outdir, config.outdir)
    #qiqc.utils.rmtree_after_confirmation(config.outdir, config.test) # pending
    train(config, modules)


def train(config, modules):
    print(config)
    start = time.time()
    set_seed(config.seed)
    config.outdir.mkdir(parents=True, exist_ok=True)
    build_model = modules.build_model

    Preprocessor = modules.Preprocessor
    TextNormalizer = modules.TextNormalizer
    TextTokenizer = modules.TextTokenizer
    WordEmbeddingFeaturizer = modules.WordEmbeddingFeaturizer
    WordExtraFeaturizer = modules.WordExtraFeaturizer
    SentenceExtraFeaturizer = modules.SentenceExtraFeaturizer

    #train_df, submit_df = load_qiqc(n_rows=config.n_rows)
    train_df, submit_df = load_qiqc(n_rows=50)
    datasets = build_datasets(train_df, submit_df, config.holdout, config.seed)

    train_dataset, test_dataset, submit_dataset = datasets

    # print('Tokenize texts...')
    preprocessor = Preprocessor()
    normalizer = TextNormalizer(config)
    tokenizer = TextTokenizer(config)
    train_dataset.tokens, test_dataset.tokens, submit_dataset.tokens = \
        preprocessor.tokenize(datasets, normalizer, tokenizer)

    print('Build vocabulary...')
    vocab = preprocessor.build_vocab(datasets, config)

    print('Build token ids...')
    train_dataset.tids, test_dataset.tids, submit_dataset.tids = \
        preprocessor.build_tokenids(datasets, vocab, config)

    print('Build sentence extra features...')
    sentence_extra_featurizer = SentenceExtraFeaturizer(config)

    train_dataset._X2, test_dataset._X2, submit_dataset._X2 = \
        preprocessor.build_sentence_features(datasets, sentence_extra_featurizer)

    print(train_dataset._X2)

    #in def build(self, device) convert array to tensor
    [d.build(config.device) for d in datasets]

    print('Load pretrained vectors...')
    pretrained_vectors = load_pretrained_vectors(
        config.use_pretrained_vectors, vocab.token2id, test=config.test)

    print('Build word embedding matrix...')
    word_embedding_featurizer = WordEmbeddingFeaturizer(config, vocab)
    embedding_matrices = preprocessor.build_embedding_matrices(
        datasets, word_embedding_featurizer, vocab, pretrained_vectors)

    print('Build word extra features...')
    word_extra_featurizer = WordExtraFeaturizer(config, vocab)
    word_extra_features = word_extra_featurizer(vocab)

    print('Build models...')
    word_features_cv = [
        preprocessor.build_word_features(
            word_embedding_featurizer, embedding_matrices, word_extra_features)
        for i in range(config.cv)]

    models = [
        build_model(
            config, word_features, sentence_extra_featurizer.n_dims
        ) for word_features in word_features_cv]

    print('Start training...')
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=config.cv, shuffle=True, random_state=config.seed)

    train_results, valid_results = [], []
    best_models = []

    for i_cv, (train_indices, valid_indices) in enumerate(
            splitter.split(train_dataset.df, train_dataset.df.target)):
        if config.cv_part is not None and i_cv >= config.cv_part:
            break

        train_tensor = train_dataset.build_labeled_dataset(train_indices)
        valid_tensor = train_dataset.build_labeled_dataset(valid_indices)
        valid_iter = DataLoader(
            valid_tensor, batch_size=config.batchsize_valid)

        model = models.pop(0)
        model = model.to_device(config.device)
        model_snapshots = []
        optimizer = torch.optim.Adam(model.parameters(), config.lr)
        train_result = ClassificationResult('train', config.outdir, str(i_cv))
        valid_result = ClassificationResult('valid', config.outdir, str(i_cv))

        batchsize = config.batchsize

        for epoch in range(config.epochs):
            if epoch in config.scale_batchsize:
                batchsize *= 2
                print('Batchsize: {}'.format(batchsize))
            epoch_start = time.time()
            sampler = None
            train_iter = DataLoader(
                train_tensor, sampler=sampler, drop_last=True,
                batch_size=batchsize, shuffle=sampler is None)
            _summary = []

            for i, batch in enumerate(tqdm(train_iter, desc='train', leave=False)):
                model.train()
                optimizer.zero_grad()
                loss, output = model.calc_loss(*batch)
                loss.backward()
                optimizer.step()
                train_result.add_record(**output)

            train_result.calc_score(epoch, 'train')
            _summary.append(train_result.summary.iloc[-1])

            # Validation loop
            if epoch >= config.validate_from:
                for i, batch in enumerate(
                        tqdm(valid_iter, desc='valid', leave=False)):
                    model.eval()
                    loss, output = model.calc_loss(*batch)
                    valid_result.add_record(**output)
                valid_result.calc_score(epoch, 'validation')
                _summary.append(valid_result.summary.iloc[-1])

                _model = deepcopy(model)
                _model.threshold = valid_result.summary.threshold[epoch]
                model_snapshots.append(_model)

            summary = pd.DataFrame(_summary).set_index('name')
            epoch_time = time.time() - epoch_start
            pbar = '#' * (i_cv + 1) + '-' * (config.cv - 1 - i_cv)

            tqdm.write('{} cv: {} / {}, epoch {}, time:{}'.format(pbar,i_cv,config.cv,epoch,epoch_time))
            tqdm.write(str(summary))

        train_results.append(train_result)
        valid_results.append(valid_result)
        best_indices = valid_result.summary.fbeta.argsort()[::-1]
        best_models.extend([model_snapshots[i] for i in
                            best_indices[:config.ensembler_n_snapshots]])





if __name__ == '__main__':
    main()
