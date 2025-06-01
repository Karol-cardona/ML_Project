import os
import numpy as np
import random

from monk_loader import load_monk_data, one_hot_encode
from MLP import MatrixMLP
from plotting import plot_learning_curve
from evaluation import evaluate_model, evaluate_regression
from search_utils import random_search, create_focused_param_grid, grid_search


def train_and_evaluate_model(X_train, y_train, X_val, y_val, params, epochs=50):
    """
   Build, train, and evaluate a MatrixMLP on a training/validation split.
    """

    model = MatrixMLP(
        n_inputs=X_train.shape[1],
        n_hidden=params['n_hidden'],
        n_outputs=y_train.shape[1],
        activation=params['activation'],
        output_activation='sigmoid',
        learning_rate=params['learning_rate'],
        reg_lambda=0.0,
        l1_lambda=0.0,
        dropout_rate=params['dropout_rate'],
        momentum=params.get('momentum',0.9),
        optimizer=params['optimizer'],
        lr_decay=params['lr_decay'],
        batch_size=params['batch_size'],
        weight_init=params.get('weight_init','range'),
        init_range=params.get('init_range',(-0.7,0.7)),
        early_stopping=False,
        patience=params.get('patience',100)
    )

    # Train the model quietly
    model.train(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=False
    )

    # Compute loss and accuracy on the validation set
    y_val_pred = model.predict(X_val)
    val_loss = model.compute_loss(y_val_pred, y_val)
    val_acc = model.compute_metrics(y_val_pred, y_val)['accuracy']
    return val_loss, val_acc, model

def kfold_split(X, y, k=5, shuffle=True, random_state=None):
    """
    Generate indices for k-fold cross-validation.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Determine fold sizes (evenly distributed)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    # Build the list of splits
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop

    return folds

def training(dataset, initial_param_space):
    monk = dataset

    # 1) Directory of the folder monk
    base_dir = os.path.join("results", f"monk{monk}")
    os.makedirs(base_dir, exist_ok=True)

    # File global best hyperparams for MONK
    BEST_HYPERPARAMS_FILE = os.path.join(base_dir, "grid_search_best_params.txt")

    # 2) Load MONK training and test data with one-hot encoding
    train_file = f'./Monk/monks-{monk}.train'
    test_file  = f'./Monk/monks-{monk}.test'
    X_train_full, y_train_full = load_monk_data(train_file, one_hot=True)
    X_test, y_test = load_monk_data(test_file, one_hot=True)

    # Ensure labels are one-hot encoded
    if y_train_full.ndim == 1:
        y_train_full = one_hot_encode(y_train_full)
    if y_test.ndim == 1:
        y_test = one_hot_encode(y_test)

    print(f"Train full:{X_train_full.shape}, {y_train_full.shape}")
    print(f"Test:{X_test.shape}, {y_test.shape}")

    # 3) Prepare k-fold cross-validation splits
    key = 5
    folds = kfold_split(X_train_full, y_train_full, k=key, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n=== Fold {fold + 1}/{key} ===")
        X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

        # 4a) Perform random search to identify promising configurations
        print("Starting random search...")
        random_results = random_search(
            X_train, y_train, X_val, y_val,
            initial_param_space,
            n_trials=200,
            epochs=50,
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
            n_top=20,
            max_combinations=200
        )

        # 4c) Conduct grid search on the narrowed parameter grid
        print("\nStarting grid search with focused grid...")
        best_params, best_val_loss, best_val_acc = grid_search(
            X_train, y_train, X_val, y_val,
            focused_param_grid,
            epochs=50,
            train_eval_fn=train_and_evaluate_model,
            result_dir=base_dir,
        )

        # 5) Final training pass on the current fold with the best parameters
        final_csv = os.path.join(base_dir, f"monk{monk}_fold{fold+1}_training_log.csv")
        print("\nTraining finale con i migliori iperparametri:")
        final_model = MatrixMLP(
            n_inputs = X_train.shape[1],
            n_hidden = best_params['n_hidden'],
            n_outputs = y_train.shape[1],
            activation = best_params['activation'],
            output_activation = 'sigmoid',
            learning_rate = best_params['learning_rate'],
            reg_lambda = 0.0,
            l1_lambda = 0.0,
            dropout_rate = best_params['dropout_rate'],
            momentum = best_params['momentum'],
            optimizer = best_params['optimizer'],
            lr_decay = best_params['lr_decay'],
            batch_size = best_params.get('batch_size',32),
            weight_init = best_params.get('weight_init','range'),
            init_range = best_params.get('init_range',(-0.5,0.5)),
            early_stopping = False,
            patience = best_params.get('patience', 100)
        )

        final_model.train(
            X_train, y_train,
            epochs=200,
            validation_data=(X_val, y_val),
            verbose=True,
            csv_log_path=final_csv
        )

        # Evaluate and record performance on the validation split
        val_loss, val_acc = evaluate_model(final_model, X_val, y_val)
        fold_results.append((val_loss, val_acc))
        print(f"\nFold {fold+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 6) Compute and display average cross-validation performance
    avg_loss = np.mean([res[0] for res in fold_results])
    avg_acc = np.mean([res[1] for res in fold_results])
    print(f"\n=== Risultati Finali K-Fold MONK-{monk} ===")
    print(f"Validation Loss Medio: {avg_loss:.4f}")
    print(f"Validation Accuracy Medio: {avg_acc:.4f}")

    # 7) Final training on the entire training set and fine-tuning on test set
    final_full_csv = os.path.join(base_dir, "monk_final_training_log.csv")
    print("\nAddestramento finale su tutto il training set...")
    final_model = MatrixMLP(
        n_inputs = X_train_full.shape[1],
        n_hidden = best_params['n_hidden'],
        n_outputs = y_train_full.shape[1],
        activation = best_params['activation'],
        output_activation = 'sigmoid',
        learning_rate = best_params['learning_rate'],
        reg_lambda = 0.0,
        l1_lambda = 0.0,
        dropout_rate = best_params['dropout_rate'],
        momentum = best_params['momentum'],
        optimizer = best_params['optimizer'],
        lr_decay = best_params['lr_decay'],
        batch_size = best_params.get('batch_size',32),
        weight_init = best_params.get('weight_init','range'),
        init_range = best_params.get('init_range',(-0.7,0.7)),
        early_stopping = False,
        patience = best_params.get('patience', 100)
    )

    # Train on full training set
    final_model.train(
        X_train_full, y_train_full,
        epochs=200,
        validation_data=(X_test, y_test),
        verbose=True,
        csv_log_path=final_full_csv
    )

    # Evaluate on test set and plot learning curves
    print("\nValutazione sul test set:")
    print("******** TRAIN ********")
    evaluate_model(final_model, X_train, y_train)
    evaluate_regression(final_model, X_train, y_train)
    print("******** TEST ********")
    evaluate_model(final_model, X_test, y_test)
    evaluate_regression(final_model, X_test, y_test)
    plot_learning_curve(final_model.train_losses,
                        final_model.val_losses,
                        final_model.train_accuracies,
                        final_model.val_accuracies,
                        title_loss=f'MONK-{monk} Loss Curve',
                        title_acc=f'MONK-{monk} Accuracy Curve')
