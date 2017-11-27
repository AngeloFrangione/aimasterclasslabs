import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 3, 5)
        self.conv3 = nn.Conv2d(3, 4, 5)
        self.conv4 = nn.Conv2d(5, 5, 5)
        self.fc0 = nn.Linear(28*28, 20*20)
        self.fc1 = nn.Linear(20*20, 27)

    def forward(self, x):
        F.max_pool2d(self.conv1, 5)
        F.max_pool2d(self.conv2, 5)
        x = self.fc0(x.view(x.size(0), -1))
        F.relu(x)
        x = self.fc1(x.view(x.size(0), -1))
        F.max_pool2d(self.conv3, 5)
        F.max_pool2d(self.conv4, 5)
        return F.log_softmax(x)
