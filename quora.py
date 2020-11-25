# !/usr/bin/env python
# coding: utf-8

import numpy as np  # linear algebra
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import torch
import torchtext
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from torch import LongTensor
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils import data
from torch.utils.data import TensorDataset, Subset
from torchtext.vocab import Vectors

# Reproducing same results
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Cuda algorithms
torch.backends.cudnn.deterministic = True

# not used yet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Config path
os.chdir("/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/quora")
os.listdir("/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/quora")
own_dir_path = '/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/quora'
if not os.path.exists(own_dir_path):
    os.makedirs(own_dir_path)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 800)

LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

train_df = pd.read_csv('train.csv')

# Split data into testing and training and convert to csv or excel files

# train,test = train_test_split(train_df, test_size=0.05, random_state=0)
# save the data
# train.to_csv(own_dir_path +'/train.csv',index=False)
# test.to_csv(own_dir_path +'/test.csv',index=False)

print("train of columns name :::", train_df.columns)
train_df['question_text'] = train_df['question_text'].astype(str)
train_df['question_text'] = train_df['question_text'].str.lower()


def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    return clean_special_chars(data, punct)


train_df['question_text'] = train_df['question_text'].apply(lambda x: preprocess(x))

# torch_text with train data
question_text_field = torchtext.data.Field(lower=True, tokenize='spacy', sequential=True, batch_first=True)
target_field = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
qid_field = torchtext.data.Field()

train_fields = [
    ('qid', None),
    ('question_text', question_text_field),
    ('target', target_field)
]

kfold_train_tabular_dataset = torchtext.data.TabularDataset(path=own_dir_path + '/train.csv',
                                                            format='csv',
                                                            fields=train_fields,
                                                            skip_header=True,
                                                            # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                                                            )

print("kfold_train_tabular_dataset", kfold_train_tabular_dataset)

# it's require create vocab using train tabular dataset
MAX_VOCAB_SIZE = 18000  # 25000

# question_text_field.build_vocab(kfold_train_tabular_dataset, max_size = MAX_VOCAB_SIZE, min_freq=1)
question_text_field.build_vocab(kfold_train_tabular_dataset, max_size=MAX_VOCAB_SIZE)
vocab = question_text_field.vocab

EMBEDDING_FILE = '/home/sunil/Downloads/glove.6B/glove.6B.300d.txt'

vocab.load_vectors([
    Vectors(EMBEDDING_FILE, cache=own_dir_path),
    # vocab.Vectors(torch_fasttext_path, cache='.')
])

print("kfold train tabular dataset :::::", kfold_train_tabular_dataset[0].__dict__.keys())

print("Most common text field ", question_text_field.vocab.freqs.most_common(10))

print("length of kfold train tabular dataset :::", len(kfold_train_tabular_dataset))

print("comment text field index to string ::::", question_text_field.vocab.itos[:10])

print("comment text field shape :::", question_text_field.vocab.vectors.shape)

print("Unique tokens in text vocabulary :::", len(question_text_field.vocab))


class TorchtextSubset(Subset):
    def __init__(self, dataset, indices):
        super(TorchtextSubset, self).__init__(dataset, indices)
        self.fields = self.dataset.fields
        self.sort_key = self.dataset.sort_key


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).to(device)
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()

        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embed_size)
        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.atten1 = Attention(LSTM_UNITS * 2, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS)
        self.relu = nn.ReLU()

        self.linear_out = nn.Linear(LSTM_UNITS, 1)

    def forward(self, x, x_len):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        h_embedding = pack_padded_sequence(h_embedding, x_len, batch_first=True)
        out1, (h_n, c_n) = self.lstm1(h_embedding)
        h_embedding, lengths = pad_packed_sequence(out1, batch_first=True)

        h_embedding, _ = self.atten1(h_embedding, lengths)
        conc = self.relu(self.linear1(h_embedding))

        result = self.linear_out(conc)
        return result


# kfold split data into 2 phase train , validation part
N_SPLITS = 2
SEED = 10

kfold = KFold(n_splits=N_SPLITS, shuffle=False, random_state=SEED)
idx_splits = list(kfold.split(range(len(kfold_train_tabular_dataset))))


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# save and load model functions
def save(m, info):
    torch.save(info, own_dir_path + '/best_model.info')
    torch.save(m, own_dir_path + '/best_model.m')


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def get_val_score(model, val_loader, loss_fn):
    val_loader.init_epoch()

    # deactivating dropout layers
    model.eval()

    # initialize every epoch
    val_loss, val_acc = 0, 0

    # deactivates autograd
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # retrieve text and no. of words
            data_len_question_text = LongTensor(list(map(len, data.question_text)))
            question = data.question_text
            question = question.to(device)
            my_target = data.target.to(device).float()

            x_batch = question
            y_batch = my_target

            # convert to 1d tensor
            y_pred = model(x_batch, data_len_question_text).squeeze(1)

            # compute loss and accuracy
            loss = loss_fn(y_pred, y_batch)
            acc = binary_accuracy(y_pred, y_batch)

            # keep track of loss and accuracy
            val_loss += loss.item()
            val_acc += acc.item()

        return val_acc / len(val_loader), val_loss / len(val_loader)


def train_model(model, lr, batch_size, n_epochs, loss_fn):
    min_loss = float('inf')
    for i, (train_idx, val_idx) in enumerate(idx_splits):

        train_ds = TorchtextSubset(kfold_train_tabular_dataset, train_idx)
        val_ds = TorchtextSubset(kfold_train_tabular_dataset, val_idx)

        train_loader, val_loader = torchtext.data.BucketIterator.splits(
            [train_ds, val_ds], batch_sizes=[batch_size, batch_size], device=device,
            sort_key=lambda x: len(x.question_text),
            sort_within_batch=True, repeat=False)

        param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
        optimizer = torch.optim.Adam(param_lrs, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
        step = 0

        for epoch in range(n_epochs):
            train_loader.init_epoch()

            start_time = time.time()

            # initialize every epoch
            epoch_loss, epoch_acc = 0, 0

            for i, data in enumerate(train_loader):
                step += 1

                # resets the gradients after every batch
                optimizer.zero_grad()

                # retrieve text and no. of words
                data_len_question_text = LongTensor(list(map(len, data.question_text)))
                question = data.question_text
                question = question.to(device)
                my_target = data.target.to(device).float()
                x_batch = question
                y_batch = my_target

                # convert to 1D tensor
                y_pred = model(x_batch, data_len_question_text).squeeze(1)

                # compute loss
                loss = loss_fn(y_pred, y_batch)

                # compute accuracy
                acc = binary_accuracy(y_pred, y_batch)

                # backpropage the loss and compute the gradients
                loss.backward()

                # update the weights
                optimizer.step()
                scheduler.step()

                # loss and accuracy
                epoch_loss += loss.item()
                epoch_acc += acc.item()

                if step % len(train_loader) == 0:

                    # evaluate the model
                    val_acc, val_loss = get_val_score(model, val_loader, loss_fn)

                    print("current val_loss", val_loss, "last min_loss", min_loss)

                    # save the best model
                    if val_loss < min_loss:
                        save(m=model, info={'epoch': epoch, 'val_loss': val_loss})
                        min_loss = val_loss

                    print(
                        'val_acc', val_acc, 'val_loss', val_loss, 'train_acc', epoch_acc / len(train_loader),
                        'train_loss',
                                                                               epoch_loss / len(train_loader))

            elapsed_time = time.time() - start_time
            print(
                'Epoch {}/{} \t loss={:.4f} \t accouracy={} \t time={:.2f}s  '.format(epoch + 1, n_epochs,
                                                                                      epoch_loss / len(train_loader),
                                                                                      epoch_acc / len(train_loader),
                                                                                      elapsed_time))

    return model


# embedding_matrix it's passed in to embedding layers
embedding_matrix = question_text_field.vocab.vectors
print("embedding_matrix ::::", embedding_matrix)
print("embedding_matrix shape::::", embedding_matrix.shape)

# model init
model = NeuralNet(embedding_matrix)

# set batch size
lr = 0.001
BATCH_SIZE = 64
n_epochs = 1

# train model
# model = train_model(model, lr, BATCH_SIZE, n_epochs, loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
# print()

# load best model
model = torch.load(own_dir_path + '/best_model.m')
info = torch.load(own_dir_path + '/best_model.info')

# deactivating dropout layers
model.eval();
print("model info:::", info)

test_fields = [
    ('qid', qid_field),
    ('question_text', question_text_field)
]

kfold_test_tabular_dataset = torchtext.data.TabularDataset(own_dir_path + '/test.csv',
                                                           format='csv',
                                                           fields=test_fields, skip_header=True,
                                                           ## if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                                                           )

print("kfold test ::::", kfold_test_tabular_dataset[0].__dict__.keys())
print("test-- dataset length of kfold tabular dataset:::::", len(kfold_test_tabular_dataset))

# IMP :: run build vocab than check qid_field.vocab.itos
qid_field.build_vocab(kfold_test_tabular_dataset)

def test_model(model):
    test_loader = torchtext.data.BucketIterator(dataset=kfold_test_tabular_dataset,
                                                batch_size=BATCH_SIZE,
                                                # sort_key=lambda x: len(x.question_text),
                                                train=False,
                                                sort_within_batch=False,
                                                shuffle=False, sort=False)

    all_test_preds = []
    q_id = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data_len_question_text = LongTensor(list(map(len, data.question_text)))
            question = data.question_text
            question = question.to(device)
            x_batch = (question)

            y_pred = sigmoid(model(x_batch, data_len_question_text).numpy()).tolist()
            q_id += data.qid.view(-1).data.numpy().tolist()
            all_test_preds += y_pred

    submission = pd.DataFrame.from_dict({
        'id': [qid_field.vocab.itos[i] for i in q_id],
        'prediction': [i[0] for i in all_test_preds],
    })

    return submission


submission = test_model(model)
print()
submission.to_csv(own_dir_path + "/submission.csv", index=False)

# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/

# inference
import spacy

nlp = spacy.load('en')


def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  # tokenize the sentence
    indexed = [question_text_field.vocab.stoi[t] for t in tokenized]  # convert to integer sequence
    length = [len(indexed)]  # compute no. of words
    tensor = torch.LongTensor(indexed).to(device)  # convert to tensor
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)  # convert to tensor
    prediction = model(tensor, length_tensor)  # prediction
    return prediction.item()


# make predictions
print(predict(model, "Are there any sports that you don't like?"))
print(predict(model, "Why Indian girls go crazy about marrying Shri. Rahul Gandhi ji?"))
print(predict(model, "How much money was Modi and BJP paid for hushing up the 2G scam?"))
