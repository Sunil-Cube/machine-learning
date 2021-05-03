import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

import netron


# dummy_input = torch.zeros(1, 3)
#
#
# class LinearInLinear(nn.Module):
#     def __init__(self):
#         super(LinearInLinear, self).__init__()
#         self.l = nn.Linear(3, 5)
#
#     def forward(self, x):
#         return self.l(x)
#
#
# m = LinearInLinear()
# o = m(dummy_input)
#
# onnx_path = "/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/git_manage/help_pytorch/runs/onnx/"
# torch.onnx.export(m, dummy_input, onnx_path+"LinearInLinear.onnx")
#
# class MultipleInput(nn.Module):
#     def __init__(self):
#         super(MultipleInput, self).__init__()
#         self.Linear_1 = nn.Linear(3, 5)
#
#
#     def forward(self, x, y):
#         return self.Linear_1(x+y)
#
# m1 = MultipleInput()
#
# onnx_path = "/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/git_manage/help_pytorch/runs/onnx/"
# torch.onnx.export(m1, (torch.zeros(1, 3), torch.zeros(1, 3)), onnx_path+"MultipleInput.onnx")
#
#
# dummy_input = torch.zeros(1, 3)
#
# class MultipleOutput_shared(nn.Module):
#     def __init__(self):
#         super(MultipleOutput_shared, self).__init__()
#         self.Linear_1 = nn.Linear(3, 5)
#
#     def forward(self, x):
#         return self.Linear_1(x), self.Linear_1(x)
#
# m2 = MultipleOutput_shared()
#
# onnx_path = "/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/git_manage/help_pytorch/runs/onnx/"
# torch.onnx.export(m2, dummy_input, onnx_path+"MultipleOutput_shared.onnx")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(
            n_categories +
            input_size +
            hidden_size,
            hidden_size)
        self.i2o = nn.Linear(
            n_categories +
            input_size +
            hidden_size,
            output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden, input

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_letters = 100
n_hidden = 128
n_categories = 10
rnn = RNN(n_letters, n_hidden, n_categories)
cat = torch.Tensor(1, n_categories)
dummy_input = torch.Tensor(1, n_letters)
hidden = torch.Tensor(1, n_hidden)


out, hidden, input = rnn(cat, dummy_input, hidden)

onnx_path = "/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/git_manage/help_pytorch/runs/onnx/"
torch.onnx.export(rnn, (cat, dummy_input, hidden), onnx_path+"rnn.onnx")
