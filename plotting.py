import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies,
                        title_loss='Training vs Validation Loss',
                        title_acc='Training vs Validation Accuracy'):
    """
    Plot learning curves for loss and accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        ax1.plot(val_losses, label='Validation Loss', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(title_loss)
    ax1.legend()

    # Plot accuracy
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    if val_accuracies:
        ax2.plot(val_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(title_acc)
    ax2.legend()

    plt.tight_layout()
    plt.show()
