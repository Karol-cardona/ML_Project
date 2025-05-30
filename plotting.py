import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter  # pip install scipy

def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies,
                        title_loss='Loss Curve', title_acc='Accuracy Curve'):
    """
    Plot smoothed learning curves for loss and accuracy.
    A moving average filter is applied to reduce the fluctuation in the curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss curves.
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    if val_losses is not None:
        ax1.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(title_loss, fontsize=14)
    ax1.set_xlim(0, 100)
    ax1.legend(fontsize=10)

    # Plot accuracy curves.
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
    if val_accuracies is not None:
        ax2.plot(val_accuracies, label='Validation Accuracy', color='orange', linewidth=2)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(title_acc, fontsize=14)
    ax2.set_xlim(0, 100)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_cup_regression(train_mse, train_mee, val_mse, val_mee,
                        title_mse='CUP MSE Curve', title_mee='CUP MEE Curve',
                        model=None, csv_path=None):

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

    # Plot MSE
    ax1.plot(train_mse, label='Training Loss', color='blue')
    if val_mse is not None:
        ax1.plot(val_mse, label='Validation MSE', color='orange', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss / MSE', fontsize=12)
    ax1.set_title(title_mse, fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 25)
    ax1.legend(fontsize=10)

    # Plot MEE
    ax2.plot(train_mee, label='Training Loss', color='blue')
    if val_mee is not None:
        ax2.plot(val_mee, label='Validation MEE', color='green', linewidth=2)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss / MEE', fontsize=12)
    ax2.set_title(title_mee, fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 10)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

