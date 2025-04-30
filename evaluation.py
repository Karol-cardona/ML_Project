import numpy as np

def mean_euclidean_error(y_pred, y_true):
    """Mean Euclidean Error (MEE) su un array di shape (n_samples, n_outputs)."""
    return np.mean(np.linalg.norm(y_pred - y_true, axis=1))

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

def evaluate_regression(model, X_test, y_test):
    """
    Evaluate a trained regression model on test data.
    """
    # Forward pass
    preds = model.predict(X_test)

    # Loss (may include L1/L2 penalties)
    test_loss = model.compute_loss(preds, y_test)

    # MSE and MAE
    mse = np.mean((preds - y_test) ** 2)
    mae = np.mean(np.abs(preds - y_test))

    # R² score
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    print(f"Test Loss: {test_loss:.6f}")
    print(f"MSE:       {mse:.6f}")
    print(f"MAE:       {mae:.6f}")
    print(f"R²:        {r2:.4f}")

    return test_loss, mse, mae, r2
