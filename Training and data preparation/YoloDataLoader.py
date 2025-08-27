import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from ast import literal_eval
from math import floor


class SpaceDebrisDataset(Dataset):
    def __init__(self, csv_file, lowResDirectory, hiResDirectory, S = 16):
        # Load the dataset from the CSV file
        self.csv = pd.read_csv(csv_file)
        self.lowResDirectory = lowResDirectory
        self.hiResDirectory = hiResDirectory
        self.S = S
    
    def __len__(self):
        # The length of the dataset is the number of rows in the CSV file
        return len(self.csv)
    
    def toCellCoord(self, coords):
        #coords is a [c, x, y, w, h] tuple relative to the image size
        # we need to convert it to the [c, x, y, w, h, n] where n is the cell number and x, y are relative to the cell.
        # n is the number of the cell counted from the top left, starting from 0 to SxS-1, so n = floor(x/cellSize) + (floor(y/cellSize)) * S

        S = self.S
        cellSize = 1 / S
        x = coords[1]
        y = coords[2]
        n = floor(x/cellSize) + (floor(y/cellSize)) * S
        relCoords = [coords[0], (coords[1] % cellSize)/cellSize, (coords[2] % cellSize)/cellSize, coords[3], coords[4], n]

        return relCoords
    
    def __getitem__(self, index):
        # Generate the paths for the hi/low resolution images
        lowResPath = os.path.join(self.lowResDirectory, str(self.csv.iloc[index, 0]) + '.jpg')
        hiResPath = os.path.join(self.hiResDirectory, str(self.csv.iloc[index, 0]) + '.jpg')
        # Load the images as tensors from the paths
        lowResImage = torchvision.transforms.ToTensor()(Image.open(lowResPath))
        hiResImage = torchvision.transforms.ToTensor()(Image.open(hiResPath))
        # Load the bboxes from the CSV file
        bboxes = literal_eval(self.csv.iloc[index, 1])
        # Convert the bboxes from absolute values to relative values
        newBboxes = []
        for bbox in bboxes:
            bbox = [bbox[0] / 512, bbox[1] / 512, bbox[2] / 512, bbox[3] / 512]
            # Conversion to [x, y, w, h] format
            newBbox = [1, (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2, (bbox[1] - bbox[0]), (bbox[3] - bbox[2])]
            newBboxes.append(self.toCellCoord(newBbox))

        zeros = torch.tensor([0, 0, 0, 0, 0, 0])
        for i in range(7 - len(newBboxes)):
            newBboxes.append(zeros)
        
        # Convert the bboxes to a tensor    
        newBboxes = torch.tensor(newBboxes)
        
        return lowResImage, hiResImage, newBboxes
    
    

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_sample(dataset, index=0, hiRes=False):
    """
    Visualizes a sample from the SpaceDebrisDataset with bounding boxes.

    Args:
        dataset (Dataset): An instance of SpaceDebrisDataset.
        index (int): Index of the sample to visualize.
     hiRes (bool): If True, shows high-res image; otherwise, low-res.
    """
    # Get the sample
    low_res_img, high_res_img, bboxes= dataset[index]
    img_tensor = high_res_img if hiRes else low_res_img

    # Convert tensor image to PIL format
    img = torchvision.transforms.ToPILImage()(img_tensor)
    img_width, img_height = img.size

    # Create a matplotlib figure
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')

    # Draw each bounding box
    for box in bboxes:
        _, x_center, y_center, width, height = box
        x = (x_center - width / 2) * img_width
        y = (y_center - height / 2) * img_height
        w = width * img_width
        h = height * img_height
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.title(f"{'High' if hiRes else 'Low'} Resolution Image {index}")
    plt.show()


trainingSet = SpaceDebrisDataset('./data/train.csv', './data/LowResTrain1ch', './data/Train1ch')
bboxes = trainingSet[0]