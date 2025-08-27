import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
import os


def AttentionInfo(index, spAttMask = None, chAttMask = None, show = False, saveName = None):
    if spAttMask != None:
        spMin = spAttMask.min()
        spMax = spAttMask.max()
        print(f'Spatial Attention {index} Min-Max: {spMin}  -  {spMax} ')
    if chAttMask != None:
        chMin = chAttMask.min()
        chMax = chAttMask.max()
        print(f'Channel Attention {index} Min-Max: {chMin}  -  {chMax} ')
    if show:
        to_pil_image = ToPILImage()
        mask = to_pil_image(spAttMask[0].cpu().squeeze(0))
        mask.show()
    if saveName != None: 
        to_pil_image = ToPILImage()
        mask = to_pil_image(spAttMask[0].cpu().squeeze(0))
        mask.save(os.path.join('./AttentionMasks', f"{saveName}.jpg"))
class ChannelAttention(nn.Module):
    def __init__(self, inChannels, redRatio):
        super(ChannelAttention, self).__init__()
        # The pooling layers output a 1x1xC tensor, meaning a single value for each channel
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.maxPool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(nn.Conv2d(inChannels, inChannels // redRatio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inChannels // redRatio, inChannels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avgPool(x))
        max = self.mlp(self.maxPool(x))

        add = avg + max
        x = self.sigmoid(add)

        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernelSize=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernelSize, stride=1, padding=(kernelSize-1)//2, bias=False)

    def forward(self, x, show=False):
        avg = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True).values
        x = torch.cat([avg, max], dim=1)
        x = self.conv(x)
        x = torch.sigmoid(x)
        if show:
            to_pil_image = ToPILImage()
            print(f"shape of x: {x.shape}")
            # Show the last image in the batch
            last_img = x[-1, 0].cpu().squeeze(0)
            to_pil_image(last_img).show()
        return x
    

class CBAM(nn.Module):
    def __init__(self, inChannels, redRatio, kernelSize=7, show=False):
        super(CBAM, self).__init__()
        self.show = show
        self.channelAttention = ChannelAttention(inChannels, redRatio)
        self.spatialAttention = SpatialAttention(kernelSize)

    def forward(self, x):
        chAtt = self.channelAttention(x)
        x = chAtt * x
        spAtt = self.spatialAttention(x, self.show)
        x = spAtt * x

        return x