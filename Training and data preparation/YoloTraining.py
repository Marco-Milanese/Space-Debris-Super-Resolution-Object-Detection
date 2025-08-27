from YoloDataLoader import SpaceDebrisDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from YoloAutoencoder import Autoencoder
import os
#from datetime import datetime
from YoloLoss import YoloLoss
from lossLogger import logLosses
from tqdm import tqdm

# Select the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Training and Validation datasets
TrainingSet = SpaceDebrisDataset('./data/trainTest.csv', './data/LowResTrain1ch', './data/Train1ch')
TrainLen = len(TrainingSet)
print(f'Training set size: {TrainLen}')

ValSet = SpaceDebrisDataset('./data/valTest.csv', './data/LowResVal1ch', './data/Val1ch')
ValLen = len(ValSet)
print(f'Validation set size: {ValLen}')

batch_size = 16 # batch size chosen as 2^7, good for Colab GPU memory
epochs = 10

# Load the datasets into dataloaders
trainDataLoader = DataLoader(TrainingSet, batch_size, shuffle=True)
valDataLoader = DataLoader(ValSet, batch_size, shuffle=True)
model = Autoencoder().to(device)

# Load the pre-trained model if available
if os.path.exists('./YoloAutoencoderTest.pth'):
    print("Loading pre-trained model\n")
    model.load_state_dict(torch.load('./YoloAutoencoderTest.pth', map_location="cpu"))
    model.to(device)
else:
    print('No pre-trained model\n')
    
# Defining the loss function and optimizer
YoloLoss = YoloLoss()
MseLoss = nn.MSELoss()
lambdaSR = 500
# Adam optimizer with the specified betas and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

for epoch in tqdm(range(epochs)):
    # Training phase
    batchNumber = 0
    trainingLosses = torch.zeros(3, device=device)
    validationLosses = torch.zeros(3, device=device)
    model.train()
    for data in tqdm(trainDataLoader):
        batchNumber += 1
        totalBatch = len(trainDataLoader)

        # Pushing the data to the available device
        lowResImages, hiResImages, bboxes= data 
        lowResImages = lowResImages.to(device) 
        hiResImages = hiResImages.to(device)
        bboxes = bboxes.to(device)
        # Forward pass and loss calculation
        outputs = model(lowResImages)

        yoloLoss = YoloLoss(outputs[1], bboxes)
        mseLoss = MseLoss(outputs[0], hiResImages)
        trainLoss = yoloLoss + lambdaSR * mseLoss

        # Saving the losses for logging
        trainingLosses[0] += yoloLoss
        trainingLosses[1] += mseLoss
        trainingLosses[2] += trainLoss

        # Backward pass and optimization
        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()

    # Validation phase to prevent overfitting
    model.eval()
    with torch.no_grad():
        for data in tqdm(valDataLoader):
            lowResImages, hiResImages, bboxes= data
            lowResImages = lowResImages.to(device)
            hiResImages = hiResImages.to(device)
            bboxes = bboxes.to(device)
            # Forward pass
            outputs = model(lowResImages)

            yoloLoss = YoloLoss(outputs[1], bboxes)
            mseLoss = MseLoss(outputs[0], hiResImages)
            valLoss = yoloLoss + lambdaSR * mseLoss
            # Saving the losses for logging
            validationLosses[0] += yoloLoss
            validationLosses[1] += mseLoss
            validationLosses[2] += valLoss 

    #torch.save(model.state_dict(), 'YoloAutoencoderTest.pth')
    #logLosses(epoch+1, trainingLosses/len(trainDataLoader), validationLosses/len(valDataLoader))
    
    """if (epoch+1) % 5 == 0:
        current_time = datetime.now().strftime("%A, %d %B %Y")
        os.system('git add .')
        os.system(f'git commit . -m "AutoSave of the model during training - Epoch {epoch+1}/{epochs} - {current_time}"')
        os.system('git push -u origin main')
    """

    print(f"Epoch [{epoch+1}/{epochs}], Average Training Loss: YOLO: {trainingLosses[0].item()/len(trainDataLoader):.4f} Detection: {(trainingLosses[1].item()*lambdaSR)/len(trainDataLoader):.4f},\n Average Validation Loss: {validationLosses[2]/len(valDataLoader):.4f}")