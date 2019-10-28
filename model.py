import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43  # GTSRB as 43 classes


class STN(nn.Module):
    def __init__(self, in_channels):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=7)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(8, 10, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(160, 32)
        self.fc2 = nn.Linear(32, 3 * 2)

        # Initialize the weights/bias with identity transformation
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.conv1(x)
        xs = self.pool1(xs)
        xs = F.leaky_relu(xs, inplace=True)
        xs = self.conv2(xs)
        xs = self.pool2(xs)
        xs = F.leaky_relu(xs, inplace=True)
        xs = xs.view(-1, 160)
        xs = self.fc1(xs)
        xs = F.leaky_relu(xs, inplace=True)
        theta = self.fc2(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)
        self.stn = STN(3)

    def forward(self, x):
        x = self.stn(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
