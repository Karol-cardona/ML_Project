import os
import numpy as np
import random

from CUP_loader import load_cup_data
from MLP import MatrixMLP
from plotting import plot_cup_regression
from evaluation import evaluate_regression, mean_euclidean_error
from search_utils import random_search, create_focused_param_grid, grid_search

# Define the hyperparameter space for the initial random search
initial_param_space = {
    'n_hidden': [[60, 60], [80, 60], [100,70], [120, 80],[150, 100, 50], [200, 200], [50, 50, 50]],
    'learning_rate': list(np.logspace(-4, -3, 8)),
    'reg_lambda': [0.0, 1e-5, 1e-4, 1e-2, 5e-4, 1e-3, 2e-3],
    'l1_lambda': [0.0, 1e-4, 5e-4, 1e-3, 1e-5],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3],
    'momentum': [0.7, 0.8, 0.9],
    'activation': ['sigmoid', 'relu', 'leaky_relu', 'elu'],
    'optimizer': ['sgd', 'adam', 'adamw', 'quickprop', 'rprop_prev_grad'],
    'lr_decay': [0.0,  1e-4, 1e-3],
    'batch_size': [8, 16, 32, 64, 128],
    'weight_init': ['xavier', 'he', 'glorot', 'orthogonal','range'],
    'init_range': [(-0.05, 0.05)],
    'patience': [50, 100, 150]
}

def train_and_evaluate_model(X_train, y_train, X_val, y_val, params, epochs=50):
    """
    Build, train, and evaluate a MatrixMLP on a train/validation split for regression.
    """
    model = MatrixMLP(
        n_inputs=X_train.shape[1],
        n_hidden=params['n_hidden'],
        n_outputs=y_train.shape[1],
        activation=params['activation'],
        output_activation='linear',
        learning_rate=params['learning_rate'],
        reg_lambda=params['reg_lambda'],
        l1_lambda=params.get('l1_lambda',0.0),
        dropout_rate=params['dropout_rate'],
        momentum=params.get('momentum', 0.9),
        optimizer=params['optimizer'],
        lr_decay=params['lr_decay'],
        batch_size=params.get('batch_size', X_train.shape[0]),
        weight_init=params.get('weight_init', 'range'),
        init_range=params.get('init_range', (0.0, 0.05)),
        early_stopping=True,
        patience=params.get('patience', 100)
    )

    # Train without verbose output
    model.train(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=False
    )

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_loss = model.compute_loss(y_val_pred, y_val)
    val_mee  = mean_euclidean_error(y_val_pred, y_val)

    return val_loss, val_mee, model


def kfold_split(X, y, k=5, shuffle=True, random_state=None):
    """
    Generate indices for k-fold cross-validation.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    # Determine fold sizes (evenly distributed)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    # Build the list of splits
    current = 0
    splits = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_idx, val_idx))
        current = stop
    return splits

# Define a utility for an 80/20 split without using scikit-learn
def train_val_split(X, y, val_fraction=0.2, shuffle=True, random_state=None):
    """
    Split arrays X and y into training and validation sets by a given fraction.
    Returns: X_train, X_val, y_train, y_val
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    n_val = int(np.floor(val_fraction * n_samples))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    # 1) Configuration
    base_dir = os.path.join("results", "cup")
    os.makedirs(base_dir, exist_ok=True)

    # Paths
    train_file = f"./Cup/ML-CUP24-TR.csv"
    test_file = f"./Cup/ML-CUP24-TS.csv"
    GLOBAL_BEST_FILE = os.path.join(base_dir, "grid_search_best_params.txt")

    # 2) Load data
    ids_train, X_train_full, y_train_full = load_cup_data(train_file, has_targets=True)
    ids_test, X_test = load_cup_data(test_file, has_targets=False)

    print(f"Train full: {X_train_full.shape}, {y_train_full.shape}")
    print(f"Blind Test: {X_test.shape}")

    # 3) K-Fold Cross-Validation
    k = 5
    folds = kfold_split(X_train_full, y_train_full, k=k, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold}/{k} ===")
        X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

        # 4a) Perform random search to identify promising configurations
        print("Starting random search...")
        random_results = random_search(
            X_train, y_train,
            X_val, y_val,
            initial_param_space,
            n_trials=150,
            epochs=100,
            train_eval_fn=train_and_evaluate_model,
            result_dir=base_dir
        )
        top_results = random_results[:5]

        print("\nTop 5 from random search:")
        for i,(p,l,a) in enumerate(top_results,1):
            print(f" {i}) loss={l:.4f}, acc={a:.4f} -> {p}")

        # 4b) Build a focused grid around the top-5 results
        focused_param_grid = create_focused_param_grid(
            top_results,
            n_top=10,
            max_combinations=50)

        # 4c) Conduct grid search on the narrowed parameter grid
        best_params, best_val_loss, best_val_acc = grid_search(
            X_train, y_train,
            X_val, y_val,
            focused_param_grid,
            epochs=120,
            train_eval_fn=train_and_evaluate_model,
            result_dir=base_dir
        )

        # 5) Final training pass on the current fold with the best parameters
        final_csv = os.path.join(base_dir, f"cup_fold{fold}_log.csv")
        print("\nTraining finale con i migliori iperparametri:")
        final_model = MatrixMLP(
            n_inputs=X_train.shape[1],
            n_hidden=best_params['n_hidden'],
            n_outputs=y_train.shape[1],
            activation = best_params['activation'],
            output_activation='linear',
            learning_rate=best_params['learning_rate'],
            reg_lambda=best_params['reg_lambda'],
            l1_lambda = best_params['l1_lambda'],
            dropout_rate = best_params['dropout_rate'],
            momentum = best_params['momentum'],
            optimizer = best_params['optimizer'],
            lr_decay = best_params['lr_decay'],
            batch_size=best_params.get('batch_size',32),
            weight_init=best_params.get('weight_init','xavier'),
            init_range=best_params.get('init_range',(-0.05, 0.05)),
            early_stopping=True,
            patience=best_params.get('patience',100)
        )

        final_model.train(
            X_train, y_train,
            epochs=200,
            validation_data=(X_val, y_val),
            verbose=True,
            csv_log_path=final_csv
        )

        # Evaluate and record performance on the validation split mse e mee
        loss, mse, mee, r2 = evaluate_regression(final_model, X_val, y_val)
        fold_results.append((loss, mse, mee, r2))
        print(f"Fold {fold} → Val Loss: {loss:.4f}, Val MSE: {mse:.4f}, MEE: {mee:.4f}, R²: {r2:.4f}")

    # 6) Compute and display average cross-validation performance
    avg_loss = np.mean([res[0] for res in fold_results])
    avg_mse = np.mean([res[1] for res in fold_results])
    print(f"\n=== CV Results ===\nAvg Loss: {avg_loss:.4f}\nAvg MSE: {avg_mse:.4f}")

    # 7) Final training on full set with an 80/20 split for validation
    # Split the full training data into 80% train and 20% validation
    X_train_final, X_val_final, y_train_final, y_val_final = train_val_split(
        X_train_full, y_train_full,
        val_fraction=0.2,
        random_state=42,
        shuffle=True
    )
    print(f"Final train: {X_train_final.shape}, Final val: {X_val_final.shape}")


    # 8) Final training on full set
    final_full_csv = os.path.join(base_dir, "cup_final_log.csv")
    final_model = MatrixMLP(
        n_inputs=X_train_full.shape[1],
        n_hidden=best_params['n_hidden'],
        n_outputs=y_train_full.shape[1],
        activation = best_params['activation'],
        output_activation = 'linear',
        learning_rate = best_params['learning_rate'],
        reg_lambda = best_params['reg_lambda'],
        l1_lambda = best_params['l1_lambda'],
        dropout_rate = best_params['dropout_rate'],
        momentum = best_params['momentum'],
        optimizer = best_params['optimizer'],
        lr_decay = best_params['lr_decay'],
        batch_size=best_params.get('batch_size',32),
        weight_init=best_params.get('weight_init','xavier'),
        init_range=best_params.get('init_range',(-0.05, 0.05)),
        early_stopping=True,
        patience=best_params.get('patience',100)
    )

    # Train on full training set
    final_model.train(
        X_train_full, y_train_full,
        epochs=200,
        validation_data=(X_val_final, y_val_final),
        verbose=True,
        csv_log_path=final_full_csv
    )

    # Evaluate performance on the held-out validation set
    loss_full, mse_full, mee_full, r2_full = evaluate_regression(final_model, X_train_final, y_train_final)
    print(f"\nTraining → Loss: {loss_full:.4f}, MSE: {mse_full:.4f}, MEE: {mee_full:.4f}, R²: {r2_full:.4f}")

    loss_full, mse_full, mee_full, r2_full = evaluate_regression(final_model, X_val_final, y_val_final)
    print(f"Held-out Validation → Loss: {loss_full:.4f}, MSE: {mse_full:.4f}, MEE: {mee_full:.4f}, R²: {r2_full:.4f}\n")

    # 9) Predict on test set and save
    preds = final_model.predict(X_test)
    out_file = os.path.join(base_dir, "cup_blind_predictions.csv")
    with open(out_file, 'w') as f:
        f.write("ID,Target1,Target2,Target3\n")
        for idx, p in zip(ids_test, preds):
            f.write(f"{idx},{p[0]},{p[1]},{p[2]}\n")
    print(f"Saved test predictions to {out_file}")

    # Plot learning curves with training loss e validation metrics (MSE e MEE)
    plot_cup_regression(
        final_model.train_mse,
        final_model.train_mee,
        final_model.val_mse,
        final_model.val_mee,
        title_mse='CUP MSE curve',
        title_mee='CUP MEE curve'
    )
