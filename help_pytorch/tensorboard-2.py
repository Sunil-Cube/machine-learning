# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from torch.autograd import Variable
# from tensorboardX import SummaryWriter
#
# dummy_input = (torch.zeros(1, 3),)
#
#
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         # self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual
#         out = F.relu(out)
#         return out
#
#
# dummy_input = torch.rand(1, 3, 224, 224)
#
# with SummaryWriter(comment='basicblock') as w:
#     model = BasicBlock(3, 3)
#     w.add_graph(model, (dummy_input, ), verbose=True)
#



import torch
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),      #(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  #output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

dummy_input = torch.rand(13, 1, 28, 28)
model = LeNet()
with SummaryWriter(comment='Net', log_dir='runs/output') as w:
    w.add_graph(model, (dummy_input, ))