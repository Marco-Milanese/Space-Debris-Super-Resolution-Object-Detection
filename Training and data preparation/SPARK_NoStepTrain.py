from utils.YoloDataLoader import SpaceDebrisDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model.YoloAutoencoderV1_5 import Autoencoder  # use your latest model file
import os
from datetime import datetime
from utils.YoloLoss import YoloLoss
from utils.lossLoggerWeights import logLosses
from tqdm import tqdm
import pytorch_msssim
import math
import numpy as np
import random

TRAINING_LOGS_PATH = 'FULL_SPARKProgressiveTrainingLog.csv'
ckpt_path = './YoloAutoencoderFULL_SPARK.pth'
best_ckpt_path = './YoloAutoencoderFULL_BEST_SPARK.pth'

BATCH_SIZE = 64

# ----------------------------
# Global perf knobs for T4
# ----------------------------
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
except Exception:
    pass

# ----------------------------
# Repro (optional)
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ----------------------------
# Device
# ----------------------------
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# ----------------------------
# T4-optimized loader helper
# ----------------------------
def create_loader_T4(dataset, batch_size=256, is_train=True):
    num_workers = max(2, os.cpu_count() // 2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )

# ----------------------------
# DEFINE YOUR DATASETS
# We train on the full dataset at once (no progressive stepping)
# ----------------------------

dataset_full = SpaceDebrisDataset('./data/SPARK_train.csv',  './data/LowResTrain', './data/train')
val_dataset  = SpaceDebrisDataset('./data/SPARK_val.csv', './data/LowResVal',   './data/val')

print(f"Training set size (full): {len(dataset_full)}")
print(f"Validation set size: {len(val_dataset)}")

# ----------------------------
# Create your loaders with T4-optimized wrapper
# ----------------------------

train_loader = create_loader_T4(dataset_full, batch_size=BATCH_SIZE, is_train=True)
val_loader   = create_loader_T4(val_dataset,  batch_size=BATCH_SIZE, is_train=False)

# ----------------------------
# Training horizons / rules
# ----------------------------
max_epochs = 200
MIN_DELTA_FACTOR   = 0.998  # require >=0.2% improvement to count as progress
EARLY_STOP_PATIENCE= 10     # stop if no progress for this many epochs

# ----------------------------
# Model
# ----------------------------
model = Autoencoder().to(device)
# channels_last improves conv perf on T4
model = model.to(memory_format=torch.channels_last)

# torch.compile (PyTorch 2.x); safe fallback
try:
    model = torch.compile(model)
    print("torch.compile: enabled")
except Exception as e:
    print(f"torch.compile: disabled ({e})")

# ----------------------------
# Losses and Weights (Kendall trick)
# ----------------------------
YoloLossFn = YoloLoss()
MseLoss = nn.MSELoss()

def init_weight(lambda_init, device):
    return torch.nn.Parameter(torch.tensor([-torch.log(torch.tensor(lambda_init, dtype=torch.float32))], device=device))

YoloWeight = init_weight(1, device)
MSEWeight  = init_weight(100, device)
SSIMWeight = init_weight(10, device)

# ----------------------------
# Optimizer + AMP
# ----------------------------
optimizer = torch.optim.Adam(
    list(model.parameters()) + [YoloWeight, MSEWeight, SSIMWeight],
    lr=0.0015, betas=(0.9, 0.99)
)

scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

# ----------------------------
# Checkpointing
# ----------------------------

best_val_loss_all = math.inf
early_stop_counter = 0

if os.path.exists(ckpt_path):
    print("Loading pre-trained model & weights\n")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        YoloWeight.data = checkpoint['weights']['yolo'].to(device)
        MSEWeight.data  = checkpoint['weights']['mse'].to(device)
        SSIMWeight.data = checkpoint['weights']['ssim'].to(device)
    except Exception:
        # older checkpoints may have different structure; ignore if missing
        pass
    model.to(device)
else:
    print('No pre-trained model\n')

# ----------------------------
# Training Loop (AMP + channels_last + non_blocking xfers)
# ----------------------------

epoch = 0
while epoch < max_epochs:
    trainingLosses = torch.zeros(4, device=device)
    validationLosses = torch.zeros(4, device=device)

    model.train()
    for data in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
        lowResImages, hiResImages, bboxes = data

        # Non-blocking H2D + channels_last
        lowResImages = lowResImages.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        hiResImages  = hiResImages.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        bboxes       = bboxes.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_cuda):
            outputs = model(lowResImages)
            yoloLoss = YoloLossFn(outputs[1], bboxes)
            mseLoss  = MseLoss(outputs[0], hiResImages)
            ssimLoss = 1 - pytorch_msssim.ms_ssim(outputs[0], hiResImages, data_range=1.0, size_average=True)

            # Weighted losses (Kendall uncertainty)
            WeightedYoloLoss = torch.exp(-YoloWeight) * yoloLoss + YoloWeight * 0.5
            WeightedMSELoss  = torch.exp(-MSEWeight)  * mseLoss  + MSEWeight  * 0.5
            WeightedSSIMLoss = torch.exp(-SSIMWeight) * ssimLoss + SSIMWeight * 0.5
            trainLoss = WeightedYoloLoss + WeightedMSELoss + WeightedSSIMLoss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(trainLoss).backward()

        # Unscale before clipping; then clip for stability with AMP
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Log raw task losses + total
        trainingLosses[0] += yoloLoss.detach()
        trainingLosses[1] += mseLoss.detach()
        trainingLosses[2] += ssimLoss.detach()
        trainingLosses[3] = trainingLosses[3] + trainLoss.detach()

    # Validation
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
            lowResImages, hiResImages, bboxes = data
            lowResImages = lowResImages.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            hiResImages  = hiResImages.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            bboxes       = bboxes.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(lowResImages)
                yoloLoss = YoloLossFn(outputs[1], bboxes)
                mseLoss  = MseLoss(outputs[0], hiResImages)
                ssimLoss = 1 - pytorch_msssim.ms_ssim(outputs[0], hiResImages, data_range=1.0, size_average=True)

                WeightedYoloLoss = torch.exp(-YoloWeight) * yoloLoss + YoloWeight * 0.5
                WeightedMSELoss  = torch.exp(-MSEWeight)  * mseLoss  + MSEWeight  * 0.5
                WeightedSSIMLoss = torch.exp(-SSIMWeight) * ssimLoss + SSIMWeight * 0.5
                valLoss = WeightedYoloLoss + WeightedMSELoss + WeightedSSIMLoss

            validationLosses[0] += yoloLoss.detach()
            validationLosses[1] += mseLoss.detach()
            validationLosses[2] += ssimLoss.detach()
            validationLosses[3] = validationLosses[3] + valLoss.detach()

    # Averages for logging / decisions
    train_avg = (trainingLosses/len(train_loader)).detach()
    val_avg   = (validationLosses/len(val_loader)).detach()
    val_total_avg = (val_avg[0] + val_avg[1] + val_avg[2]).item()
    train_total_avg = (train_avg[0] + train_avg[1] + train_avg[2]).item()

    # Save checkpoint (model + weights)
    torch.save({
        'model_state_dict': model.state_dict(),
        'weights': {
            'yolo': YoloWeight.detach().cpu(),
            'mse' : MSEWeight.detach().cpu(),
            'ssim': SSIMWeight.detach().cpu()
        },
        'epoch': epoch
    }, ckpt_path)

    # Track global best & save best checkpoint
    if val_total_avg < best_val_loss_all * MIN_DELTA_FACTOR:
        best_val_loss_all = val_total_avg
        early_stop_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'weights': {
                'yolo': YoloWeight.detach().cpu(),
                'mse' : MSEWeight.detach().cpu(),
                'ssim': SSIMWeight.detach().cpu()
            },
            'epoch': epoch,
            'best_val': best_val_loss_all
        }, best_ckpt_path)
    else:
        early_stop_counter += 1

    # Log losses + effective weights (exp(-log_var))
    effective_weights = torch.stack([
        torch.exp(-YoloWeight.detach()),
        torch.exp(-MSEWeight.detach()),
        torch.exp(-SSIMWeight.detach())
    ])
    logLosses(
        train_avg,
        val_avg,
        effective_weights.cpu(),
        TRAINING_LOGS_PATH
    )

    # Autosave commit (optional)
    if (epoch+1) % 2 == 0:
        current_time = datetime.now().strftime("%A, %d %B %Y")
        os.system('git add .')
        os.system(f'git commit . -m "AutoSave - Epoch {epoch+1}/{max_epochs} - {current_time}"')
        os.system('git push -u origin main')

    # Console output
    print(
        f"Epoch [{epoch+1} early: {early_stop_counter} | Train YOLO: {train_avg[0].item():.4f} | MSE: {train_avg[1].item():.4f} | SSIM: {train_avg[2].item():.4f} | Train-Val : {train_total_avg - val_total_avg:.4f}"
    )

    # Early stopping on full dataset
    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"\n[Early Stop] No sufficient improvement for {EARLY_STOP_PATIENCE} epochs. Stopping training.")
        break

    # Reduce fragmentation a bit
    if use_cuda and (epoch + 1) % 3 == 0:
        torch.cuda.empty_cache()
    epoch += 1

print(f"Training complete. Best global validation total loss: {best_val_loss_all:.6f}")
print(f"Best checkpoint saved to: {best_ckpt_path}")
