import pandas as pd
import os
import matplotlib.pyplot as plt

def logLosses(trainingLosses, validationLosses, lossWeights, logFilePath):
    """
    Logs the training and validation losses for each epoch to a CSV file using pandas.
    Columns: Epoch, Train_YOLO, Train_MSE, Train_SSIM, Train_Total, Val_YOLO, Val_MSE, Val_SSIM, Val_Total
    The epoch number is read from the last entry and incremented, starting from 1 if the file does not exist.
    """
    
    # Prepare the new row as a dictionary
    new_row = {
        'YOLO_Weight': float(lossWeights[0]),
        'MSE_Weight': float(lossWeights[1]),
        'SSIM_Weight': float(lossWeights[2]),
        'Train_YOLO': float(trainingLosses[0]),
        'Train_MSE': float(trainingLosses[1]),
        'Train_SSIM': float(trainingLosses[2]),
        'Train_Total': float(trainingLosses[3]),
        'Val_YOLO': float(validationLosses[0]),
        'Val_MSE': float(validationLosses[1]),
        'Val_SSIM': float(validationLosses[2]),
        'Val_Total': float(validationLosses[3])
    }

    if os.path.isfile(logFilePath):
        # Load existing log
        df = pd.read_csv(logFilePath)
        last_epoch = df['Epoch'].max()
        epoch = last_epoch + 1
    else:
        # Create new DataFrame with proper columns
        df = pd.DataFrame(columns=[
            'Epoch', 'YOLO_Weight', 'MSE_Weight', 'SSIM_Weight', 'Train_YOLO', 'Train_MSE', 'Train_SSIM', 'Train_Total',
            'Val_YOLO', 'Val_MSE', 'Val_SSIM', 'Val_Total'
        ])
        epoch = 1

    # Add the epoch number to the new row
    new_row['Epoch'] = epoch

    # Append the new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save back to CSV
    df.to_csv(logFilePath, index=False)

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def LossVisualizationWeighted(logFilePath='sample_training_log.csv'):
    """
    Generates a figure with three subplots:
    1. Training Loss per Epoch (stacked bars + weighted total loss curve + unweighted total loss curve)
    2. Validation Loss per Epoch (stacked bars + weighted total loss curve + unweighted total loss curve)
    3. Learned weights (YOLO, MSE, SSIM) per Epoch
    """

    # Load CSV
    df = pd.read_csv(logFilePath)
    epochs = df['Epoch']

    # ------------------------
    # Training losses
    # ------------------------
    train_yolo = df['Train_YOLO'] * df['YOLO_Weight']
    train_mse  = df['Train_MSE']  * df['MSE_Weight']
    train_ssim = df['Train_SSIM'] * df['SSIM_Weight']
    train_total_weighted = train_yolo + train_mse + train_ssim

    # Raw unweighted sum
    train_total_unweighted = df['Train_YOLO'] + df['Train_MSE'] + df['Train_SSIM']

    # ------------------------
    # Validation losses
    # ------------------------
    val_yolo = df['Val_YOLO'] * df['YOLO_Weight']
    val_mse  = df['Val_MSE']  * df['MSE_Weight']
    val_ssim = df['Val_SSIM'] * df['SSIM_Weight']
    val_total_weighted = val_yolo + val_mse + val_ssim

    # Raw unweighted sum
    val_total_unweighted = df['Val_YOLO'] + df['Val_MSE'] + df['Val_SSIM']

    # ------------------------
    # Plotting
    # ------------------------
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=False)

    # --- Training subplot ---
    ax = axes[0]
    """
    ax.bar(epochs, train_yolo, label='YOLO Loss', color='skyblue', bottom=0)
    ax.bar(epochs, train_mse, label='MSE Loss', color='orange', bottom=train_yolo)
    ax.bar(epochs, train_ssim, label='SSIM Loss', color='green', bottom=train_yolo + train_mse)
    ax.plot(epochs, train_total_weighted, label='Weighted Total Loss', color='red', marker='o', linewidth=2)
    """
    ax.plot(epochs, train_total_unweighted, label='Loss over training epochs', color='red', marker='o', linewidth=2)
    ax.set_title('Training Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Validation subplot ---
    ax = axes[1]
    ax.bar(epochs, val_yolo, label='YOLO Loss', color='skyblue', bottom=0)
    ax.bar(epochs, val_mse, label='MSE Loss', color='orange', bottom=val_yolo)
    ax.bar(epochs, val_ssim, label='SSIM Loss', color='green', bottom=val_yolo + val_mse)
    ax.plot(epochs, val_total_weighted, label='Weighted Total Loss', color='red', marker='o', linewidth=2)
    ax.plot(epochs, val_total_unweighted, label='Unweighted Total Loss', color='black', linestyle='--', marker='x', linewidth=2)
    ax.set_title('Validation Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Weights subplot ---
    ax = axes[2]
    ax.plot(epochs, df['YOLO_Weight'], label='YOLO Weight', color='skyblue', marker='o', linewidth=2)
    ax.plot(epochs, df['MSE_Weight'], label='MSE Weight', color='orange', marker='o', linewidth=2)
    ax.plot(epochs, df['SSIM_Weight'], label='SSIM Weight', color='green', marker='o', linewidth=2)
    ax.set_title('Learned Weights per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight (λ)')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

LossVisualizationWeighted('FULL_DEBRISProgressiveTrainingLog.csv')