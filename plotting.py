import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter  # pip install scipy

def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies,
                        title_loss='Loss Curve', title_acc='Accuracy Curve',
                        smoothing='moving',  # 'moving', 'exp', or 'savgol'
                        window=11,           # per moving e savgol: odd integer
                        alpha=0.1,           # per exp smoothing
                        polyorder=3):        # per savgol
    """
    Plot smoothed learning curves for loss and accuracy.
    A moving average filter is applied to reduce the fluctuation in the curves.
    """

    def smooth_moving(data, w):
        # moving average with padding to preserve length
        kernel = np.ones(w) / w
        return np.convolve(data, kernel, mode='same')

    def smooth_exp(data, a):
        # exponential weighted moving average
        s = np.zeros_like(data, dtype=float)
        s[0] = data[0]
        for i in range(1, len(data)):
            s[i] = a * data[i] + (1 - a) * s[i-1]
        return s

    def apply_smoothing(data):
        # apply the chosen smoothing method
        if smoothing == 'moving':
            return smooth_moving(data, window)
        elif smoothing == 'exp':
            return smooth_exp(data, alpha)
        elif smoothing == 'savgol':
            # ensure window is odd and greater than polyorder
            w = window if window % 2 == 1 else window+1
            return savgol_filter(data, window_length=w, polyorder=polyorder, mode='interp')
        else:
            # no smoothing
            return np.array(data)

    # generate smoothed series
    smooth_train_loss = apply_smoothing(train_losses)
    smooth_val_loss = apply_smoothing(val_losses) if val_losses else None
    smooth_train_acc = apply_smoothing(train_accuracies)
    smooth_val_acc = apply_smoothing(val_accuracies) if val_accuracies else None

    # create subplots for loss and accuracy
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
