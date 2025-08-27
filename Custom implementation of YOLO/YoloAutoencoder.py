import torch
import torch.nn as nn
from CbamForYolo import CBAM, SpatialAttention

class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbam = CBAM(inChannels=512, redRatio=16, kernelSize=7, show=True)
        self.conv = nn.Conv2d(512, 5, 1)
        self.convert = nn.Conv2d(64, 512, kernel_size=8, stride=8)

    #latent --> 512 channels
    #decX --> 64 channels
    
    def forward(self, latent, decX):
        #Converting decX to the same size and number of channels as latent
        decX = self.convert(decX) #Convolution to match the number of channels
        


        combined = self.cbam(latent) + self.cbam(decX) #Combining the Latent and 3rd decoder layer output
        #print(f"Alpha: {self.alpha.item()} - Beta: {self.beta.item()}") #Printing the alpha and beta values for debugging
        #out = self.cbam(combined) #Applying CBAM attention
        out = self.conv(combined) #Convolution to get the final detection output
        return out
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        
        # Encoder
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #CBAM(inChannels=32, redRatio=4, kernelSize=7)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
                              
     
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #CBAM(inChannels=256, redRatio=16, kernelSize=7)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        ) 


        # detection head
        self.detection = DetectionHead()




    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        
        skip1 = x
        
        x = self.enc2(x)
        
        x = self.enc3(x)
        skip2 = x
        
        x = self.enc4(x)

        x = self.enc5(x)
        LatentImage = x


        # Decoder
        x = self.dec1(x)

        x = self.dec2(x) + skip2

        x = self.dec3(x)

        # Detection head
        bboxes = self.detection(LatentImage, x)

        x = self.dec4(x) + skip1
        
        x = self.dec5(x)
        
        generated_image = self.dec6(x)
        

        return generated_image, bboxes
