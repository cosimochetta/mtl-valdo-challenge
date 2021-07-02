import torch
import torch.nn as nn

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.relu(self.batch_norm(self.conv(x)))
        return out

class MultipleConv(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(SingleConv(channels, channels))
        self.multiple_conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.multiple_conv(x)
        return out

class InputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SingleConv(in_channels, out_channels)

    def forward(self, x):
        out = self.conv(x)
        x16 = x.repeat(1, 16, 1, 1, 1)
        return torch.add(x16, out)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.down_conv = SingleConv(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.convs = MultipleConv(out_channels, depth)
        

    def forward(self, x):
        down = self.down_conv(x)
        out = self.convs(down)
        return torch.add(out, down)



class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()

        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm3d(out_channels // 2)
        self.relu = nn.PReLU(out_channels // 2)
        self.convs = MultipleConv(out_channels, depth)
        

    def forward(self, x, x_skip):
        x_up = self.relu(self.batch_norm(self.up_conv(x)))
        x_cat = torch.cat((x_up, x_skip), 1)
        out = self.convs(x_cat)
        return torch.add(out, x_cat)

class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SingleConv(in_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        out = self.sigmoid(x)
        return out

