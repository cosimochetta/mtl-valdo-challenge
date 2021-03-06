""" Full assembly of the parts to form the complete network """
from torchsummary import summary
import torch.nn.functional as F
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        print(f"X3: {x3.shape}")
        x4 = self.down3(x3)
        print(f"X4: {x4.shape}")
        #x5 = self.down4(x4)
        x = self.up1(x4, x3)
        print(f"X1 up: {x.shape}")
        x = self.up2(x, x2)
        print(f"X2 up: {x.shape}")
        x = self.up3(x, x1)
        print(f"X3 up: {x.shape}")
        #x = self.up4(x, x1)
        seg_logits = self.outc(x)
        return seg_logits
    
def test_unet():
    net = UNet(3, 2)
    summary(net, (3, 32, 32))
    
if __name__ == "__main__":
    test_unet()

