import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self) -> None:
        super(Block, self).__init__()

        # 1x1, 3x3, 1x1 conv block, 32, 32, 64 filters, same pad
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x

class ValueHead(nn.Module):
    def __init__(self) -> None:
        super(ValueHead, self).__init__()

        # 1x1 conv, 1 filter, no pad
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()
        # 32 units FC, ReLU
        self.fc1 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        # 1 unit FC, tanh
        self.fc2 = nn.Linear(32, 1)
        self.tanh1 = nn.Tanh()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.tanh1(x)
        return x

class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super(PolicyHead, self).__init__()

        # 1x1 conv, 16 filters, no pad
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()
        # 294 units FC, softmax
        self.fc1 = nn.Linear(294, 294)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.softmax1(x)
        return x

class DeepMctsModel(nn.Module):
    def __init__(self) -> None:
        super(DeepMctsModel, self).__init__()

        # input board: 7x7x7 tensor
        # 3x3 conv, 64 filters, no pad
        self.conv1 = nn.Conv2d(in_channels=7*7*7, out_channels=64, kernel_size=3, padding=0)
        # 9 conv blocks
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.block4 = Block()
        self.block5 = Block()
        self.block6 = Block()
        self.block7 = Block()
        self.block8 = Block()
        self.block9 = Block()
        # value and policy heads
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()
        

    def forward(self, x):
        # using skip connections
        x = self.conv1(x)
        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x) + x
        x = self.block4(x) + x
        x = self.block5(x) + x
        x = self.block6(x) + x
        x = self.block7(x) + x
        x = self.block8(x) + x
        x = self.block9(x) + x
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy

