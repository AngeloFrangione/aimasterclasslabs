import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(28*28, 20*20)
        self.fc1 = nn.Linear(20*20, 27)

    def forward(self, x):
        x = self.fc0(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.fc1(x.view(x.size(0), -1))
        return F.log_softmax(x)
