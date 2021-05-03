import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



# dummy_input = (torch.zeros(1, 3),)
#
#
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#
#     def forward(self, x):
#         return x * 2
#
#
# model = SimpleModel()
# dummy_input = (torch.zeros(1, 2, 3),)
#
# with SummaryWriter(comment='constantModel') as w:
#     w.add_graph(model, dummy_input, True)


#==================================================================================

writer = SummaryWriter('runs/mnist1')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()

############## TENSORBOARD ########################
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
#writer.close()
#sys.exit()
###################################################

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############## TENSORBOARD ########################
writer.add_graph(model, (example_data.reshape(-1, 28 * 28),), True)

writer.close()



# Train the model
# running_loss = 0.0
# running_correct = 0
# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # origin shape: [100, 1, 28, 28]
#         # resized: [100, 784]
#         images = images.reshape(-1, 28 * 28).to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         _, predicted = torch.max(outputs.data, 1)
#         running_correct += (predicted == labels).sum().item()
#         if (i + 1) % 100 == 0:
#             print('Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
#             ############## TENSORBOARD ########################
#             writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
#             running_accuracy = running_correct / 100 / predicted.size(0)
#             writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
#             running_correct = 0
#             running_loss = 0.0
#             ###################################################
#
#
# writer.close()