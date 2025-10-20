import torch.nn as nn
from utils.CbamForYolo import CBAM

class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbam = CBAM(inChannels=320, redRatio=16, kernelSize=7, show=False)
        self.avg = nn.AdaptiveAvgPool2d((16, 16))
        self.max = nn.AdaptiveMaxPool2d((16, 16))
        self.conv = nn.Conv2d(320, 5, 1)

    def forward(self, latent):
        cbamLatent = self.cbam(latent)
        return self.conv(self.avg(cbamLatent) + self.max(cbamLatent))

class EncoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels, downscalingFactor=2):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                inChannels,
                outChannels,
                kernel_size=3,
                stride=downscalingFactor,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels, upscalingFactor=2):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                inChannels,
                outChannels * (upscalingFactor**2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscalingFactor),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1)
        # Projection if inChannels != outChannels
        self.proj = nn.Conv2d(inChannels, outChannels, kernel_size=1) if inChannels != outChannels else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return self.relu(out)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # CBAM for final skip connection
        self.cbam = CBAM(inChannels=16, redRatio=8, kernelSize=7, show=False)

        # Encoder
        self.enc1 = EncoderBlock(1, 16, downscalingFactor=1)
        self.enc2 = EncoderBlock(16, 32, downscalingFactor=2)
        self.enc3 = EncoderBlock(32, 64, downscalingFactor=2)
        self.enc4 = EncoderBlock(64, 128, downscalingFactor=2)
        self.enc5 = ResidualBlock(128, 320)

        

        # Decoder
        self.dec0 = ResidualBlock(320, 128)
        self.dec1 = DecoderBlock(128, 64, upscalingFactor=2)
        self.dec2 = DecoderBlock(64, 32, upscalingFactor=2)
        self.dec3 = DecoderBlock(32, 16, upscalingFactor=2)
        self.dec4 = DecoderBlock(16, 8, upscalingFactor=2)
        self.dec5 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

        # Detection head
        self.detection = DetectionHead()

    def forward(self, x):
        x = self.enc1(x)
        skip1 = x
        x = self.enc2(x)
        skip2 = x
        x = self.enc3(x)
        skip3 = x
        x = self.enc4(x)
        skip4 = x

        LatentImage = self.enc5(x)
        #LatentImage = self.dropout_latent(LatentImage)

        x = self.dec0(LatentImage) + skip4
        x = self.dec1(x) + skip3
        x = self.dec2(x) + skip2
        x = self.dec3(x) + skip1
        x = self.dec4(x)
        generated_image = self.dec5(x)

        bboxes = self.detection(LatentImage)

        return generated_image, bboxes
