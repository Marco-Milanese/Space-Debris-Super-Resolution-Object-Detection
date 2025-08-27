import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from YoloAutoencoderV2 import Autoencoder
from torchvision.ops import nms
to_pil_image = ToPILImage()


def Inference(imagePath, modelPath="YoloAutoencoder.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferenceModel = Autoencoder().to(device)
    inferenceModel.load_state_dict(torch.load(modelPath, map_location=device))
    inferenceModel.eval()

    # Convert the image to a tensor an add a batch dimension
    inputImage = ToTensor()(Image.open(imagePath)).unsqueeze(0).to(device)
    with torch.no_grad():
        generatedImage, rawPredictions = inferenceModel(inputImage)

    # The shape of the raw predictions is [5, 16, 16]
    # Remove batch dimension and reshape to [256, 5]
    rawPredictions = rawPredictions.squeeze(0).permute(1, 2, 0).reshape(-1, 5)

    return generatedImage, rawPredictions


def Visualize(imagePath, modelPath="YoloAutoencoderV2.pth"):
    # Parameters
    gridSize = 16
    confThreshold = 0.5
    iouThreshold = 0.4

    generatedImage, rawPredictions = Inference(imagePath, modelPath)

    imageSize, _ = to_pil_image(generatedImage.squeeze()).size
    boxes = []
    confidences = []
    cellSize = 1 / gridSize

    for i, prediction in enumerate(rawPredictions):
        confidence, xCell, yCell, width, height = prediction
        if confidence < confThreshold:
            continue

        cellX = i % gridSize
        cellY = i // gridSize

        xCenter = (xCell + cellX) * cellSize
        yCenter = (yCell + cellY) * cellSize

        absX = xCenter * imageSize
        absY = yCenter * imageSize
        absW = width * imageSize
        absH = height * imageSize

        x1 = absX - absW / 2
        y1 = absY - absH / 2
        x2 = absX + absW / 2
        y2 = absY + absH / 2

        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)

    if not boxes:
        print("No boxes above the threshold detected.")
        return

    boxesTensor = torch.tensor(boxes)
    scoresTensor = torch.tensor(confidences)
    keepIndices = nms(boxesTensor, scoresTensor, iouThreshold)
    filteredBoxes = boxesTensor[keepIndices]

    # Visualization
    imageArray = generatedImage.squeeze().clamp(0,1).cpu().numpy()
    
    print(f"image min:", generatedImage.min().item())
    print(f"image max:", generatedImage.max().item())
    to_pil_image(generatedImage.squeeze().clamp(0,1)).show()
    
    fig, ax = plt.subplots(1)
    ax.imshow(imageArray, cmap="gray")

    for box in filteredBoxes:
        x1, y1, x2, y2 = box
        boxWidth = x2 - x1
        boxHeight = y2 - y1
        rect = patches.Rectangle((x1, y1), boxWidth, boxHeight, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    plt.title("Predicted Bounding Boxes")
    plt.axis("off")
    plt.show()

#Visualize("data/LowResTest1ch/107.jpg")