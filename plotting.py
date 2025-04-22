import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies,
                        title_loss='Loss Curve', title_acc='Accuracy Curve'):
    """
    Plot smoothed learning curves for loss and accuracy.
    A moving average filter is applied to reduce the fluctuation in the curves.
    """
    def smooth(data, window=5):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    smoothing_window = 5
    smooth_train_loss = smooth(train_losses, window=smoothing_window)
    smooth_val_loss = smooth(val_losses, window=smoothing_window) if val_losses else None
    smooth_train_acc = smooth(train_accuracies, window=smoothing_window)
    smooth_val_acc = smooth(val_accuracies, window=smoothing_window) if val_accuracies else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss curves.
    ax1.plot(smooth_train_loss, label='Training Loss', color='blue', linewidth=2)
    if smooth_val_loss is not None:
        ax1.plot(smooth_val_loss, label='Validation Loss', color='orange', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(title_loss, fontsize=14)
    ax1.legend(fontsize=10)

    # Plot accuracy curves.
    ax2.plot(smooth_train_acc, label='Training Accuracy', color='blue', linewidth=2)
    if smooth_val_acc is not None:
        ax2.plot(smooth_val_acc, label='Validation Accuracy', color='orange', linewidth=2)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(title_acc, fontsize=14)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()
