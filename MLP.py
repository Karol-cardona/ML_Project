import numpy as np
import matplotlib.pyplot as plt
import csv
import os

class MatrixMLP:
    """
    Matrix-based implementation of a Multilayer Perceptron (MLP).
    Supports various activation functions, weight initializations, and optimizers.
    """
    def __init__(self, n_inputs, n_hidden, n_outputs,
                 activation='relu', output_activation='softmax',
                 learning_rate=0.01, reg_lambda=0.0, l1_lambda=0.0,
                 dropout_rate=0.2, momentum=0.9, nesterov=True,
                 optimizer='sgd', lr_decay=0.0, batch_size=32,
                 weight_init='range', init_range=(-0.7, 0.7),
                 early_stopping=True, patience=10, validation_data=None):
        """
        Initialize the MLP with specified hyperparameters.
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden if isinstance(n_hidden, list) else [n_hidden]
        self.n_outputs = n_outputs
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.reg_lambda = reg_lambda
        self.l1_lambda = l1_lambda
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.init_range = init_range
        self.early_stopping = early_stopping
        self.patience = patience

        # Initialize lists for tracking training and validation metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_loss = np.inf
        self.no_improvement_count = 0
        self.best_weights = None
        self.best_biases = None

        # For reproducibility, set a fixed random seed
        np.random.seed(1)

        # Build the network layer sizes list
        layer_sizes = [self.n_inputs] + self.n_hidden + [self.n_outputs]

        # Initialize weights and biases for each layer
        self.weights = self._initialize_weights(layer_sizes)
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

        # Ensure the output layer has the correct number of outputs
        if self.weights[-1].shape[1] != self.n_outputs:
            self.weights[-1] = np.random.randn(self.weights[-1].shape[0], self.n_outputs)
            if self.weight_init == 'orthogonal':
                u, _, v = np.linalg.svd(self.weights[-1], full_matrices=False)
                self.weights[-1] = u if u.shape == self.weights[-1].shape else v

        # Initialize optimizer-specific variables (for Adam or SGD with momentum)
        if self.optimizer == 'adam':
            self.adam_m = [np.zeros_like(w) for w in self.weights]
            self.adam_v = [np.zeros_like(w) for w in self.weights]
            self.adam_t = 0
        elif self.optimizer == 'sgd' and self.momentum > 0:
            self.momentums = [np.zeros_like(w) for w in self.weights]

    def _initialize_weights(self, layer_sizes):
        """
        Initialize weights using the selected method.
        Supported methods: 'xavier', 'he', 'glorot', 'orthogonal', 'range'.
        """
        weights = []
        for i in range(len(layer_sizes) - 1):
            shape = (layer_sizes[i], layer_sizes[i+1])
            if self.weight_init == 'xavier':
                weight = np.random.randn(*shape) * np.sqrt(1. / layer_sizes[i])
            elif self.weight_init == 'he':
                weight = np.random.randn(*shape) * np.sqrt(2. / layer_sizes[i])
            elif self.weight_init == 'glorot':
                limit = np.sqrt(6. / (layer_sizes[i] + layer_sizes[i+1]))
                weight = np.random.uniform(-limit, limit, shape)
            elif self.weight_init == 'orthogonal':
                temp = np.random.randn(*shape)
                u, _, v = np.linalg.svd(temp, full_matrices=False)
                weight = u if u.shape == shape else v
            elif self.weight_init == 'range':
                low, high = self.init_range
                weight = np.random.uniform(low, high, shape)
            else:
                # Default to glorot initialization
                limit = np.sqrt(6. / (layer_sizes[i] + layer_sizes[i+1]))
                weight = np.random.uniform(-limit, limit, shape)
            weights.append(weight)
        return weights

    def _activation(self, x):
        """Apply the activation function for hidden layers."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == 'elu':
            return np.where(x > 0, x, 0.01 * (np.exp(x) - 1))
        else:
            return np.maximum(0, x)

    def _activation_derivative(self, x):
        """Compute derivative of the activation function for backpropagation."""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        elif self.activation == 'elu':
            return np.where(x > 0, 1, 0.01 * np.exp(x))
        else:
            return (x > 0).astype(float)

    def _output_activation(self, x):
        """Apply the activation function for the output layer."""
        if self.output_activation == 'linear':
            return x
        elif self.output_activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            # Default to sigmoid (useful for binary classification)
            return 1/(1 + np.exp(-x))

    def forward(self, X, training=True):
        """
        Perform a forward pass through the network.
        Returns the final output and a cache with intermediate values.
        """
        activations = [X]
        pre_activations = []

        # Loop through layers
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            # Apply activation and dropout in hidden layers
            if i < len(self.weights) - 1:
                a = self._activation(z)
                if training:
                    a = self._apply_dropout(a)
                activations.append(a)
            else:
                a = self._output_activation(z) if self.output_activation in ['softmax', 'sigmoid'] else z
                activations.append(a)

        cache = {'activations': activations, 'pre_activations': pre_activations}
        return activations[-1], cache

    def _apply_dropout(self, layer_output):
        """
        Apply dropout regularization to the layer's output.
        """
        if self.dropout_rate > 0:
            mask = np.random.rand(*layer_output.shape) > self.dropout_rate
            return layer_output * mask / (1 - self.dropout_rate)
        return layer_output

    def compute_loss(self, y_pred, y_true):
        """
        Compute the loss (with regularization).
        Uses binary cross-entropy for one-sigmoid output,
        cross-entropy for softmax outputs, or MSE otherwise.
        """
        m = y_true.shape[0]
        l2_loss = self.reg_lambda * sum(np.sum(w**2) for w in self.weights) / (2 * m)
        l1_loss = self.l1_lambda * sum(np.sum(np.abs(w)) for w in self.weights) / (2 * m)
        eps = 1e-15

        if self.output_activation == 'sigmoid' and self.n_outputs == 1:
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            bce = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
            return bce + l2_loss + l1_loss
        elif self.output_activation == 'softmax' and self.n_outputs > 1:
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            cce = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
            return cce + l2_loss + l1_loss
        else:
            mse = np.mean((y_pred - y_true)**2) / 2
            return mse + l2_loss + l1_loss

    def mean_euclidean_error(self, y_true, y_pred):
        """
        Compute Mean Euclidean Error (MEE) for evaluation.
        """
        return np.mean(np.sqrt(np.sum((y_true - y_pred)**2, axis=1)))

    def compute_metrics(self, y_pred, y_true):
        """
        Compute performance metrics: accuracy, MSE, and MEE.
        """
        if self.n_outputs > 1:
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        else:
            accuracy = np.mean((y_pred >= 0.5).astype(int) == y_true)
        mse = np.mean((y_pred - y_true)**2)
        mee = self.mean_euclidean_error(y_true, y_pred)
        return {'accuracy': accuracy, 'mse': mse, 'mee': mee}

    def backward(self, cache, y_true):
        """
        Backpropagate errors and update weights using the chosen optimizer.
        """
        activations = cache['activations']
        pre_activations = cache['pre_activations']
        m = y_true.shape[0]
        # Start at the output layer
        delta = activations[-1] - y_true
        grad_weights = []
        grad_biases = []

        for i in reversed(range(len(self.weights))):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_w += (self.reg_lambda / m) * self.weights[i]  # L2 regularization
            if self.l1_lambda > 0:
                grad_w += (self.l1_lambda / m) * np.sign(self.weights[i])
            grad_b = np.mean(delta, axis=0, keepdims=True)
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)
            if i > 0:
                d_act = self._activation_derivative(pre_activations[i-1])
                delta = np.dot(delta, self.weights[i].T) * d_act

        # Update weights and biases
        for i in range(len(self.weights)):
            if self.optimizer == 'sgd':
                if self.momentum > 0:
                    if self.nesterov:
                        v_prev = self.momentums[i].copy()
                        self.momentums[i] = self.momentum * self.momentums[i] - self.learning_rate * grad_weights[i]
                        self.weights[i] += -self.momentum * v_prev + (1 + self.momentum) * self.momentums[i]
                    else:
                        self.momentums[i] = self.momentum * self.momentums[i] - self.learning_rate * grad_weights[i]
                        self.weights[i] += self.momentums[i]
                else:
                    self.weights[i] -= self.learning_rate * grad_weights[i]
            elif self.optimizer == 'adam':
                self.adam_t += 1
                beta1, beta2 = self.momentum, 0.999
                self.adam_m[i] = beta1 * self.adam_m[i] + (1 - beta1) * grad_weights[i]
                self.adam_v[i] = beta2 * self.adam_v[i] + (1 - beta2) * (grad_weights[i]**2)
                m_hat = self.adam_m[i] / (1 - beta1**self.adam_t)
                v_hat = self.adam_v[i] / (1 - beta2**self.adam_t)
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            # Bias update (common for both optimizers)
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def train(self, X_train, y_train, epochs=100, verbose=True,
              validation_data=None, csv_log_path=None):
        """
        Train the network using mini-batch or full-batch gradient descent.
        Supports learning rate decay and early stopping.
        """
        n_samples = X_train.shape[0]
        self.best_loss = np.inf
        self.no_improvement_count = 0

        if validation_data is not None:
            X_val, y_val = validation_data
        else:
            X_val, y_val = None, None

        if csv_log_path is not None:
            self._initialize_csv(csv_log_path)

        for epoch in range(epochs):
            # Update learning rate if decay is set
            if self.lr_decay > 0:
                self.learning_rate = self.initial_lr / (1 + self.lr_decay * epoch)

            # Shuffle training data if using mini-batches
            if self.batch_size and self.batch_size > 0:
                indices = np.random.permutation(n_samples)
                X_train = X_train[indices]
                y_train = y_train[indices]
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                num_batches = int(np.ceil(n_samples / self.batch_size))
                for batch in range(num_batches):
                    start = batch * self.batch_size
                    end = start + self.batch_size
                    X_batch = X_train[start:end]
                    y_batch = y_train[start:end]
                    y_pred, cache = self.forward(X_batch, training=True)
                    batch_loss = self.compute_loss(y_pred, y_batch)
                    epoch_loss += batch_loss
                    metrics = self.compute_metrics(y_pred, y_batch)
                    epoch_accuracy += metrics['accuracy']
                    self.backward(cache, y_batch)
                epoch_loss /= num_batches
                epoch_accuracy /= num_batches
            else:
                # Full-batch training
                y_pred, cache = self.forward(X_train, training=True)
                epoch_loss = self.compute_loss(y_pred, y_train)
                metrics = self.compute_metrics(y_pred, y_train)
                epoch_accuracy = metrics['accuracy']
                self.backward(cache, y_train)

            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)

            if X_val is not None:
                y_val_pred, _ = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_metrics = self.compute_metrics(y_val_pred, y_val)
                val_accuracy = val_metrics['accuracy']
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

                # Early stopping check
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.no_improvement_count = 0
                    self.best_weights = [w.copy() for w in self.weights]
                    self.best_biases = [b.copy() for b in self.biases]
                else:
                    self.no_improvement_count += 1
                    if self.early_stopping and self.no_improvement_count >= self.patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch+1}.")
                        self.weights = [w.copy() for w in self.best_weights]
                        self.biases = [b.copy() for b in self.best_biases]
                        break

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}")

            if csv_log_path is not None:
                self._append_csv_log(csv_log_path, epoch+1, epoch_loss, epoch_accuracy,
                                     val_loss if X_val is not None else None,
                                     val_accuracy if X_val is not None else None)

    def predict(self, X):
        """
        Predict outputs for input data X.
        """
        predictions, _ = self.forward(X, training=False)
        return predictions

    def _initialize_csv(self, csv_path):
        """Initialize CSV log file if it does not exist."""
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])

    def _append_csv_log(self, csv_path, epoch, train_loss, train_acc, val_loss, val_acc):
        """Append epoch results to the CSV log."""
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                             f"{val_loss:.6f}" if val_loss is not None else "",
                             f"{val_acc:.4f}" if val_acc is not None else ""])

def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies,
                        title_loss='Loss Curve', title_acc='Accuracy Curve'):
    """
    Plot training and validation loss and accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Plot loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    if val_losses:
        ax1.plot(val_losses, label='Validation Loss', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(title_loss)
    ax1.legend()
    # Plot accuracy curves
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    if val_accuracies:
        ax2.plot(val_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(title_acc)
    ax2.legend()
    plt.tight_layout()
    plt.show()
