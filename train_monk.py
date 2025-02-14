import os
import csv
import ast
import itertools
import numpy as np

from monk_loader import load_monk_data, one_hot_encode
from MLP import MatrixMLP, plot_learning_curve
from evaluation import evaluate_model

# Define the folder where output files will be saved
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Updated hyperparameter grid for MONK problems
param_grid = {
    'n_hidden': [[3], [5], [3, 3]],
    'learning_rate': [0.01, 0.05],
    'reg_lambda': [0.0, 0.001],
    'dropout_rate': [0.0],
    'activation': ['sigmoid', 'tanh'],
    'optimizer': ['sgd', 'adam'],
    'lr_decay': [0.0, 0.001],
    'batch_size': [16, 32]
}

def params_to_tuple(params):
    """
    Convert the parameter dictionary to a tuple for caching.
    """
    n_hidden = tuple(params['n_hidden']) if isinstance(params['n_hidden'], list) else params['n_hidden']
    return (n_hidden, params['learning_rate'], params['reg_lambda'],
            params['dropout_rate'], params['activation'],
            params['optimizer'], params['lr_decay'], params['batch_size'])

def ensure_csv_exists(csv_path):
    """
    Create a CSV file with headers if it does not exist.
    """
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["n_hidden", "learning_rate", "reg_lambda", "dropout_rate",
                             "activation", "optimizer", "lr_decay", "batch_size", "val_loss"])

def read_csv_cache(csv_path):
    """
    Read existing grid search results from the CSV file.
    Returns a dictionary mapping parameter tuples to validation loss.
    """
    cache = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_hidden_val = row["n_hidden"].strip()
                n_hidden = ast.literal_eval(n_hidden_val) if n_hidden_val.startswith('[') else int(n_hidden_val)
                learning_rate = float(row["learning_rate"])
                reg_lambda = float(row["reg_lambda"])
                dropout_rate = float(row["dropout_rate"])
                activation = row["activation"]
                optimizer = row["optimizer"]
                lr_decay = float(row["lr_decay"])
                batch_size = int(row["batch_size"])
                val_loss = float(row["val_loss"])
                param_key = (tuple(n_hidden) if isinstance(n_hidden, list) else n_hidden,
                             learning_rate, reg_lambda, dropout_rate,
                             activation, optimizer, lr_decay, batch_size)
                cache[param_key] = val_loss
    return cache

def append_to_csv(csv_path, params, val_loss):
    """
    Append a new grid search result to the CSV file.
    """
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([params['n_hidden'], params['learning_rate'], params['reg_lambda'],
                         params['dropout_rate'], params['activation'], params['optimizer'],
                         params['lr_decay'], params['batch_size'], val_loss])

def grid_search(X_train, y_train, X_val, y_val, param_grid, epochs=50, csv_path=None):
    """
    Perform grid search over hyperparameters.
    Saves each evaluated parameter combination and its validation loss in a CSV file.
    Returns the best hyperparameters and corresponding validation loss.
    """
    # Save CSV inside the output directory
    if csv_path is None:
        csv_path = os.path.join(output_dir, "monk_grid_search.csv")
    ensure_csv_exists(csv_path)
    cache = read_csv_cache(csv_path)

    # Generate all hyperparameter combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_params = None
    best_loss = np.inf

    for params in combinations:
        param_key = params_to_tuple(params)
        # Skip if this combination was already evaluated
        if param_key in cache:
            current_loss = cache[param_key]
            print(f"Skipping parameters {params}, already evaluated with validation loss: {current_loss:.4f}")
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params
            continue

        # Train the model with the current hyperparameters
        print(f"Evaluating parameters: {params}")
        model = MatrixMLP(
            n_inputs=X_train.shape[1],
            n_hidden=params['n_hidden'],
            n_outputs=y_train.shape[1],
            activation=params['activation'],
            output_activation='sigmoid',  # Use sigmoid for binary classification
            learning_rate=params['learning_rate'],
            reg_lambda=params['reg_lambda'],
            dropout_rate=params['dropout_rate'],
            optimizer=params['optimizer'],
            lr_decay=params['lr_decay'],
            batch_size=params['batch_size'],
            early_stopping=True,
            patience=15
        )
        model.train(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=False)
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        current_loss = model.compute_loss(y_val_pred, y_val)
        print(f"Validation loss: {current_loss:.4f}")

        # Save the result in the CSV cache
        append_to_csv(csv_path, params, current_loss)
        cache[param_key] = current_loss

        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params

    # Save the best parameters in a separate file inside the output directory
    best_params_file = os.path.join(output_dir, "grid_search_best_params.txt")
    with open(best_params_file, "w") as f:
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Validation Loss: {best_loss:.6f}\n")
    print(f"Best hyperparameters saved in '{best_params_file}'")
    return best_params, best_loss

if __name__ == "__main__":
    train_file = './Monk/monks-1.train'
    test_file = './Monk/monks-1.test'

    # Load training and testing data with one-hot encoding enforced
    X_train_full, y_train_full = load_monk_data(train_file, one_hot=True)
    X_test, y_test = load_monk_data(test_file, one_hot=True)

    # Ensure labels are in one-hot format (if not already)
    if y_train_full.ndim == 1:
        y_train_full = one_hot_encode(y_train_full)
    if y_test.ndim == 1:
        y_test = one_hot_encode(y_test)

    print(f"Training data shape: {X_train_full.shape}, {y_train_full.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    # Create an 80/20 training/validation split
    split_index = int(0.8 * len(y_train_full))
    X_train = X_train_full[:split_index]
    y_train = y_train_full[:split_index]
    X_val = X_train_full[split_index:]
    y_val = y_train_full[split_index:]
    print(f"Train split: {X_train.shape}, Validation split: {X_val.shape}")

    # Perform grid search to find the best hyperparameters; CSV is saved in the output folder.
    best_params, best_val_loss = grid_search(X_train, y_train, X_val, y_val,
                                             param_grid, epochs=50)
    print("Best Hyperparameters:", best_params)
    print("Best Validation Loss:", best_val_loss)

    # Train the final model using the best hyperparameters on the full training set.
    # Save the training log in the output folder.
    final_log_csv = os.path.join(output_dir, "monk_final_training_log.csv")
    final_model = MatrixMLP(
        n_inputs=X_train_full.shape[1],
        n_hidden=best_params['n_hidden'],
        n_outputs=y_train_full.shape[1],
        activation=best_params['activation'],
        output_activation='sigmoid',
        learning_rate=best_params['learning_rate'],
        reg_lambda=best_params['reg_lambda'],
        dropout_rate=best_params['dropout_rate'],
        optimizer=best_params['optimizer'],
        lr_decay=best_params['lr_decay'],
        batch_size=best_params['batch_size'],
        early_stopping=False,  # Disable early stopping for final training
        patience=15
    )

    print("Training final model with best hyperparameters...")
    final_model.train(X_train_full, y_train_full, epochs=50,
                      validation_data=(X_val, y_val),
                      verbose=True, csv_log_path=final_log_csv)

    print("Evaluating final model on test set:")
    evaluate_model(final_model, X_test, y_test)

    # Optionally, plot the learning curves
    plot_learning_curve(final_model.train_losses, final_model.val_losses,
                        final_model.train_accuracies, final_model.val_accuracies,
                        title_loss='MONK-1 Loss Curve', title_acc='MONK-1 Accuracy Curve')
