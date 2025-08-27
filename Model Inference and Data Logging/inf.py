import torch
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from YoloAutoencoder import Autoencoder
from YoloDataLoader import SpaceDebrisDataset
from torchvision.ops import nms
import os

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("YoloAutoencoder.pth", map_location=device))
model.eval()

# Load dataset (adjust paths if needed)
dataset = SpaceDebrisDataset(
    csv_file="./data/test.csv",  # Update this if needed
    lowResDirectory="./data/LowResTest1ch",
    hiResDirectory="./data/Test1ch"
)

# Parameters
gridSize = 16
confThreshold = 0.5
iouThreshold = 0.4



def runInference(index=0):
    lowResImage, _, _ = dataset[index]
    
    inputTensor = lowResImage.unsqueeze(0).to(device)  # Add batch dimension
    print(f"shape: {lowResImage.shape()}")
    with torch.no_grad():
        generatedImage, rawPredictions = model(inputTensor)

    rawPredictions = rawPredictions.squeeze(0).permute(1, 2, 0).reshape(-1, 5)  # [256, 5]

    boxes = []
    confidences = []

    cellSize = 1 / gridSize
    for i, prediction in enumerate(rawPredictions):
        confidence, xCell, yCell, width, height = prediction.tolist()
        if confidence < confThreshold:
            continue

        cellX = i % gridSize
        cellY = i // gridSize

        xCenter = (xCell + cellX) * cellSize
        yCenter = (yCell + cellY) * cellSize

        absX = xCenter * 512
        absY = yCenter * 512
        absW = width * 512
        absH = height * 512

        x1 = absX - absW / 2
        y1 = absY - absH / 2
        x2 = absX + absW / 2
        y2 = absY + absH / 2

        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)

    if not boxes:
        print(f"No boxes detected above threshold in image {index}.")
        return

    boxesTensor = torch.tensor(boxes)
    scoresTensor = torch.tensor(confidences)
    keepIndices = nms(boxesTensor, scoresTensor, iouThreshold)
    filteredBoxes = boxesTensor[keepIndices]

    # Visualization
    imageArray = generatedImage.squeeze().clamp(0,1).cpu().numpy()
    
    print(f"image min:", generatedImage.min().item())
    print(f"image max:", generatedImage.max().item())
    to_pil_image = ToPILImage()
    to_pil_image(generatedImage.squeeze().clamp(0,1)).show()
    
    fig, ax = plt.subplots(1)
    ax.imshow(imageArray, cmap="gray")

    for box in filteredBoxes:
        x1, y1, x2, y2 = box
        boxWidth = x2 - x1
        boxHeight = y2 - y1
        rect = patches.Rectangle((x1, y1), boxWidth, boxHeight, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    plt.title(f"Predicted Bounding Boxes - Image {index}")
    plt.axis("off")
    plt.show()

# Run on the first image
runInference(index=1998)
