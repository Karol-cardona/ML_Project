import matplotlib.pyplot as plt
import numpy as np
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
    ax1.set_xlim(0, 50)
    ax1.legend(fontsize=10)

    # Plot accuracy curves.
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
    if val_accuracies is not None:
        ax2.plot(val_accuracies, label='Validation Accuracy', color='orange', linewidth=2)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(title_acc, fontsize=14)
    ax2.set_xlim(0, 50)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_cup_regression(train_losses,
                        title_loss='CUP Loss Curve'):
    """
    Traccia side-by-side la Loss (con regolarizzazione) e il vero MSE
    per training e, se disponibili, validation.
    """
    fig, ax1 = plt.subplots(1,1, figsize=(7,5))

    # Loss plot
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.set_title(title_loss)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xlim(0, 50)
    ax1.legend()

    plt.tight_layout()
    plt.show()
