import csv
import numpy as np
import os

class MatrixMLP:
    """
    Matrix-based implementation of a Multilayer Perceptron (MLP).
    This simulator supports various activation functions, weight initializations,
    and optimizers. It uses mini-batch gradient descent with optional early stopping.
    """
    def __init__(self, n_inputs, n_hidden, n_outputs,
                 activation='relu', output_activation='softmax',
                 learning_rate=0.01, reg_lambda=0.0, l1_lambda=0.0,
                 dropout_rate=0.2, momentum=0.9, nesterov=True,
                 optimizer='sgd', lr_decay=0.0, batch_size=32,
                 weight_init='range', init_range=(-0.7, 0.7),
                 early_stopping=True, patience=10, validation_data=None):
        """
        Initialize the MLP with the specified hyperparameters.
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

        # Lists for tracking performance over epochs.
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_loss = np.inf
        self.no_improvement_count = 0
        self.best_weights = None
        self.best_biases = None

        # Set random seed for reproducibility.
        np.random.seed(1)

        # Build the network architecture.
        layer_sizes = [self.n_inputs] + self.n_hidden + [self.n_outputs]

        # Initialize weights and biases.
        self.weights = self._initialize_weights(layer_sizes)
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

        # Check that output layer weights have the correct shape.
        if self.weights[-1].shape[1] != self.n_outputs:
            self.weights[-1] = np.random.randn(self.weights[-1].shape[0], self.n_outputs)
            if self.weight_init == 'orthogonal':
                u, _, v = np.linalg.svd(self.weights[-1], full_matrices=False)
                self.weights[-1] = u if u.shape == self.weights[-1].shape else v

        # Optimizer-specific initialization.
        if self.optimizer == 'adam':
            self.adam_m = [np.zeros_like(w) for w in self.weights]
            self.adam_v = [np.zeros_like(w) for w in self.weights]
            self.adam_t = 0
        elif self.optimizer == 'sgd' and self.momentum > 0:
            self.momentums = [np.zeros_like(w) for w in self.weights]

    def _init_weight_matrix(self, n_in, n_out):
        """
        Initialize weights based on the chosen method.
        Supported methods: 'gaussian' 'xavier', 'he', 'glorot', 'orthogonal', 'range'.
        """
        if self.weight_init == 'gaussian':
            mu, sigma = self.init_range
            return np.random.randn(n_in, n_out) * sigma + mu
        elif self.weight_init == 'xavier':
            return np.random.randn(n_in, n_out) * np.sqrt(1. / n_in)
        elif self.weight_init == 'he':
            return np.random.randn(n_in, n_out) * np.sqrt(2. / n_in)
        elif self.weight_init == 'glorot':
            limit = np.sqrt(6. / (n_in + n_out))
            return np.random.uniform(-limit, limit, (n_in, n_out))
        elif self.weight_init == 'orthogonal':
            temp = np.random.randn(n_in, n_out)
            u, _, v = np.linalg.svd(temp, full_matrices=False)
            return u if u.shape == (n_in, n_out) else v
        elif self.weight_init == 'range':
            low, high = self.init_range
            return np.random.uniform(low, high, (n_in, n_out))
        else:
            limit = np.sqrt(6. / (n_in + n_out))
            return np.random.uniform(-limit, limit, (n_in, n_out))

    def _initialize_weights(self, layer_sizes):
        return [self._init_weight_matrix(layer_sizes[i], layer_sizes[i+1])
                for i in range(len(layer_sizes)-1)]

    def _activation(self, x):
        """Apply the activation function for hidden layers."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1/(1+np.exp(-np.clip(x,-500,500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'leaky_relu':
            return np.where(x>0, x, 0.01*x)
        elif self.activation == 'elu':
            return np.where(x>0, x, 0.01*(np.exp(np.clip(x,-500,500))-1))
        else:
            return np.maximum(0, x)

    def _activation_derivative(self, x):
        """Compute the derivative of the activation function."""
        if self.activation == 'relu':
            return (x>0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1/(1+np.exp(-np.clip(x,-500,500)))
            return sig*(1-sig)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'leaky_relu':
            return np.where(x>0,1,0.01)
        elif self.activation == 'elu':
            return np.where(x>0,1,0.01*np.exp(np.clip(x,-500,500)))
        else:
            return (x>0).astype(float)

    def _output_activation(self, x):
        """Apply the activation for the output layer."""
        if self.output_activation == 'linear':
            return x
        elif self.output_activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            return 1/(1+np.exp(-np.clip(x,-500,500)))

    def forward(self, X, training=True):
        """
        Perform a forward pass through the network.
        Returns the final output and a cache of intermediate results.
        """
        activations = [X]
        pre_activations = []
        for i, W in enumerate(self.weights):
            z = np.dot(activations[-1], W) + self.biases[i]
            pre_activations.append(z)
            if i < len(self.weights)-1:
                a = self._activation(z)
                if training and self.dropout_rate > 0:
                    a = self._apply_dropout(a)
                activations.append(a)
            else:
                a = self._output_activation(z)
                activations.append(a)

        cache = {'activations': activations, 'pre_activations': pre_activations}
        return activations[-1], cache

    def _apply_dropout(self, layer_output):
        """
       Apply dropout regularization to the output of a layer.
       """
        mask = np.random.rand(*layer_output.shape) > self.dropout_rate
        return layer_output * mask / (1-self.dropout_rate)

    def compute_loss(self, y_pred, y_true):
        """
        Compute the loss with L1 and L2 regularization.
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
        elif self.output_activation == 'softmax':
            y_pred_clipped = np.clip(y_pred, eps, 1)
            cce = -np.mean(np.sum(y_true*np.log(y_pred_clipped), axis=1))
            return cce + l2_loss + l1_loss
        else:
            mse = np.mean((y_pred-y_true)**2)/2
            return mse + l2_loss + l1_loss

    def compute_metrics(self, y_pred, y_true):
        """
        Compute key metrics: accuracy, MSE, and MEE.
        """
        if self.n_outputs > 1:
            accuracy = np.mean(np.argmax(y_pred, axis=1)==np.argmax(y_true, axis=1))
        else:
            accuracy = np.mean((y_pred>=0.5).astype(int)==y_true)
        mse = np.mean((y_pred-y_true)**2)
        return {'accuracy': accuracy, 'mse': mse}

    def backward(self, cache, y_true):
        """
        Backpropagation: compute gradients and update weights.
        """
        activations = cache['activations']
        pre_activations = cache['pre_activations']
        m = y_true.shape[0]
        # Start at the output layer
        delta = activations[-1] - y_true
        grad_weights = []
        grad_biases = []

        for i in reversed(range(len(self.weights))):
            grad_w = np.dot(activations[i].T, delta)/m
            grad_w += (self.reg_lambda/m)*self.weights[i]
            if self.l1_lambda > 0:
                grad_w += (self.l1_lambda/m)*np.sign(self.weights[i])
            grad_b = np.mean(delta, axis=0, keepdims=True)
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)
            if i > 0:
                d_act = self._activation_derivative(pre_activations[i-1])
                delta = np.dot(delta, self.weights[i].T) * d_act
        # GRADIENT CLIPPING (norm-based)
        max_norm = 5.0
        # calcola norma complessiva
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grad_weights))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            grad_weights = [g * scale for g in grad_weights]

        self._update_params(grad_weights, grad_biases)

    # Update weights and biases
    def _update_params(self, grad_weights, grad_biases):
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
                self.adam_m[i] = beta1 * self.adam_m[i] + (1-beta1) * grad_weights[i]
                self.adam_v[i] = beta2 * self.adam_v[i] + (1-beta2) * (grad_weights[i]**2)
                m_hat = self.adam_m[i] / (1 - beta1**self.adam_t)
                v_hat = self.adam_v[i] / (1 - beta2**self.adam_t)
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.biases[i] -= self.learning_rate * grad_biases[i]

            clip_val = 5.0
            self.weights[i] = np.clip(self.weights[i], -clip_val, clip_val)

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
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = int(np.ceil(n_samples / self.batch_size))
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = min(start + self.batch_size, n_samples)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                y_pred, cache = self.forward(X_batch, training=True)
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss * (end - start) / n_samples
                metrics = self.compute_metrics(y_pred, y_batch)
                epoch_accuracy += metrics['accuracy'] * (end - start) / n_samples
                self.backward(cache, y_batch)
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)

            if X_val is not None:
                y_val_pred, _ = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_acc = self.compute_metrics(y_val_pred, y_val)['accuracy']
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

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
                            print(f"Early stopping at epoch {epoch+1}.")
                        self.weights = [w.copy() for w in self.best_weights]
                        self.biases = [b.copy() for b in self.best_biases]
                        break

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                if csv_log_path is not None:
                    self._append_csv_log(csv_log_path, epoch+1, epoch_loss, epoch_accuracy, val_loss, val_acc)

    def predict(self, X):
        """
        Predict the output for given input X.
        """
        preds, _ = self.forward(X, training=False)
        return preds

    def _initialize_csv(self, csv_path):
        """Initialize the CSV log file if it does not exist."""
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])

    def _append_csv_log(self, csv_path, epoch, train_loss, train_acc, val_loss, val_acc):
        """Append epoch results to the CSV log file."""
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                             f"{val_loss:.6f}" if val_loss is not None else "",
                             f"{val_acc:.4f}" if val_acc is not None else ""])