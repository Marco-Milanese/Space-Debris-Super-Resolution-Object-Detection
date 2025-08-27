import json
import os
from YoloAutoencoderInference import Inference
import datetime
from torchvision.transforms import ToPILImage
import pandas as pd
import matplotlib.pyplot as plt
import torch


def logLosses(epoch, trainingLosses, validationLosses):
    
    logFile = {}
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    dirPath = f"./TrainingLogs/Training{date}"
    os.makedirs(dirPath, exist_ok=True)
    logFilePath = os.path.join(dirPath, "trainingLogs.json")
    # Load existing log file if it exists, create an empty one otherwise
    if os.path.exists(logFilePath):
        with open(logFilePath, 'r') as f:
            logFile = json.load(f)
    else:
        with open(logFilePath, 'w') as f:
            pass

    # Log the losses to the JSON file
    logFile[str(epoch)] = {
        "Training":{
            "Detection": trainingLosses[0].item(),
            "Reconstruction": trainingLosses[1].item(),
            "Total": trainingLosses[2].item()
        }
    }

    logFile[str(epoch)] = {
        "Validation":{
            "Detection": validationLosses[0].item(),
            "Reconstruction": validationLosses[1].item(),
            "Total": validationLosses[2].item()
        }
        
    }

    with open(logFilePath, 'w') as f:
        json.dump(logFile, f, indent=4)

    # get and save the model reconstruction and predicted bboxes for test image 134.jpg
    generatedImage, bboxes = Inference("./data/Test1ch/134.jpg")
    to_pil_image = ToPILImage()
    generatedImage = to_pil_image(generatedImage.squeeze().clamp(0,1))
    generatedImage.save(os.path.join(dirPath, f"epoch{epoch}.jpg"))
    torch.save(bboxes, os.path.join(dirPath, f"bboxesEpoch{epoch}.pth"))

    

def lossGraphs(logPath):

    # Load JSON
    with open(logPath, "r") as f:
        logs = json.load(f)

    # Flatten the nested structure
    flattened = []
    for epoch, data in logs.items():
        row = {"epoch": int(epoch)}
        for section in ["Training", "Validation"]:
            for key, value in data[section].items():
                row[f"{section} {key}"] = value
        flattened.append(row)

    # Create DataFrame
    df = pd.DataFrame(flattened).sort_values("epoch").set_index("epoch")

    # Plot total training and validation loss
    plt.plot(df["Training Total"], label="Trainining Total Loss")
    plt.plot(df["Validation Total"], label="Validation Total Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Time")
    plt.grid(True)
    plt.show()

    # Plot total training detection and reconstruction loss
    plt.plot(df["Training Detection"], label="Trainining Detection Loss")
    plt.plot(df["Training Reconstruction"], label="Trainining Reconstruction Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Detection & Reconstruction Loss Over Time")
    plt.grid(True)
    plt.show()

    # Plot total validation detection and reconstruction loss
    plt.plot(df["Validation Detection"], label="Validation Detection Loss")
    plt.plot(df["Validation Reconstruction"], label="Validation Reconstruction Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Detection & Reconstruction Loss Over Time")
    plt.grid(True)
    plt.show()
