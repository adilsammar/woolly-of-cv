import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

GROUP_SIZE = 2

class Block(nn.Module):
    def __init__(self, input_size, output_size, padding=1, norm='bn', usepool=True):
        super(Block, self).__init__()
        self.usepool = usepool
        self.conv1 = nn.Conv2d(input_size, output_size, 3, padding=padding)
        if norm=='bn':
            self.n1 = nn.BatchNorm2d(output_size)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, output_size)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, output_size)
        self.conv2 = nn.Conv2d(output_size, output_size, 3, padding=padding)
        if norm=='bn':
            self.n2 = nn.BatchNorm2d(output_size)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, output_size)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, output_size)
        self.conv3 = nn.Conv2d(output_size, output_size, 3, padding=padding)
        if norm=='bn':
            self.n3 = nn.BatchNorm2d(output_size)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(GROUP_SIZE, output_size)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, output_size)
        if usepool:
            self.pool = nn.MaxPool2d(2, 2)
        
    def __call__(self, x, layers=3, last=False):
        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)
        if layers >= 2:
            x = self.conv2(x)
            x = self.n2(x)
            x = F.relu(x)
        if layers >= 3:
            x = self.conv3(x)
            x = self.n3(x)
            x = F.relu(x)
        if self.usepool:
            x = self.pool(x)
        return x

class Net(nn.Module):
    def __init__(self, base_channels=4, layers=3, drop=0.01, norm='bn'):
        super(Net, self).__init__()
        
        self.base_channels = base_channels
        self.drop = drop
        self.no_layers = layers

        # Conv 
        self.block1 = Block(1, self.base_channels, norm=norm)
        self.dropout1 = nn.Dropout(self.drop)
        self.block2 = Block(self.base_channels, self.base_channels*2, norm=norm)
        self.dropout2 = nn.Dropout(self.drop)
        self.block3 = Block(self.base_channels*2, self.base_channels*2, norm=norm, usepool=False)
        self.dropout3 = nn.Dropout(self.drop)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Conv2d(self.base_channels*2, 10, 1)

    def forward(self, x, dropout=True):
        # Conv Layer
        x = self.block1(x, layers=self.no_layers)
        if dropout:
            x = self.dropout1(x)
        x = self.block2(x, layers=self.no_layers)
        if dropout:
            x = self.dropout2(x)
        x = self.block3(x, layers=self.no_layers, last=True)

        # Output Layer
        x = self.gap(x)
        x = self.flat(x)
        x = x.view(-1, 10)

        # Output Layer
        return F.log_softmax(x, dim=1)