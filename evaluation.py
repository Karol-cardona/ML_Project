import numpy as np

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the trained model on test data.
    Computes the loss and accuracy.
    """
    predictions = model.predict(test_data)
    test_loss = model.compute_loss(predictions, test_labels)
    if test_labels.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
    else:
        pred_classes = (predictions >= 0.5).astype(int)
        true_classes = test_labels
    accuracy = np.mean(pred_classes == true_classes)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return test_loss, accuracy
