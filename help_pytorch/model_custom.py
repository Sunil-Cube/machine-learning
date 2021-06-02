import torch
from torch import nn

import torch.nn.functional as F
from torch.optim import Adam

# uniform random number
# r1= 2
# r2 = 5
# t1 = (r1 - r2) * torch.rand(10, 4)
# print(t1.type())
# print(t1.long())

token_ids = 10
embedding_fix_size = 800
encoder_n_hidden = 128

# embedding = nn.Embedding(token_ids, embedding_fix_size)
# print(embedding)

# embedding_pre = nn.Embedding.from_pretrained(torch.Tensor(token_ids, embedding_fix_size), freeze=True)
# x = torch.LongTensor(token_ids, embedding_fix_size).random_(token_ids)
# output = embedding_pre(x)
#
# print("embedding_pre-output",output.shape)
#
#
# embedding_out_size = 300
# lstm = nn.LSTM(embedding_out_size, encoder_n_hidden)
# output, _ = lstm(output)
#
# print("lstm-output",output.shape)
#
# linear = nn.Linear(encoder_n_hidden, 1)
# output = linear(output[-1])
#
# print("liner-output",output.shape)
#
# activation = nn.Sigmoid()
# output = activation(output)
#
# print("activation-output",output.shape)


# class CustomModel(nn.Module):
#     def __init__(self, vocab_size=10,
#           embedding_dim=300,
#           hidden_size=128):
#         super().__init__()
#         self.encoder = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_size)
#         self.linear = nn.Linear(hidden_size, 1)
#         self.activation = nn.Sigmoid()
#
#     def forward(self, x):
#         output = self.encoder(x)
#         output, _ = self.lstm(output)
#         output = output[-1] # Keep last output only
#         output = self.linear(output)
#         output = self.activation(output)
#         return output


# x = torch.LongTensor(token_ids, embedding_fix_size).random_(token_ids)
# model = CustomModel()
# o1= model.forward(x)

# print("----output1",o1.shape)


my_token_ids = 10
my_embedding_fix_size = 300
my_encoder_n_hidden = 128


class MyModel(nn.Module):
    def __init__(self, my_token_ids,
                 my_embedding_fix_size,
                 my_encoder_n_hidden):
        super().__init__()

        self.embedding = nn.Embedding(
            my_token_ids,
            my_embedding_fix_size
        )

        self.rnn = nn.GRU(my_embedding_fix_size, my_encoder_n_hidden,  bidirectional=True)

        self.seq1 = nn.Sequential(
            nn.Linear(

            ),
            nn.ReLU(),
            nn.Linear(

            ),nn.ReLU())


    def forward(self, x):
        features = self.embedding(x)
        features, _ = self.rnn(features)
        features = self.seq1(features)
        return features


# x = torch.LongTensor(my_token_ids, my_embedding_fix_size).random_(my_token_ids)
# model = MyModel(my_token_ids, my_embedding_fix_size, my_encoder_n_hidden)
# o1= model.forward(x)
# print(o1.shape)

x = torch.LongTensor(my_token_ids, my_embedding_fix_size).random_(my_token_ids)
model = MyModel(my_token_ids, my_embedding_fix_size, my_encoder_n_hidden)
print(model)

o1 = model.forward(x)
# print(o1)
