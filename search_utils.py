import os
import random
import itertools
import numpy as np

from csv_utils import (
    ensure_csv_exists,
    read_csv_cache,
    append_to_csv,
    params_to_tuple,
    update_global_best
)

def random_search(X_train, y_train, X_val, y_val, param_space, n_trials=50, epochs=30, train_eval_fn=None, result_dir="result"):
    """
    Perform a randomized hyperparameter search.
    """
    # Path to CSV log for random search results
    csv_path = os.path.join(result_dir, "monk_random_search.csv")
    ensure_csv_exists(csv_path, is_random_search=True)

    # Load any cached results to skip duplicate evaluations
    cache = read_csv_cache(csv_path)
    results = []
    for trial in range(n_trials):
        # Randomly pick a value for each hyperparameter
        params = {k: random.choice(v) for k, v in param_space.items()}
        key = params_to_tuple(params)
        if key in cache:
            # Cached result available: skip training
            loss, acc = cache[key]
            print(f"Trial {trial+1}/{n_trials}: Skipping (cached) → loss={loss:.4f}, acc={acc:.4f}")
        else:
            # Evaluate new hyperparameter combination
            print(f"Trial {trial+1}/{n_trials}: Evaluating...")
            loss, acc, _ = train_eval_fn(X_train, y_train, X_val, y_val, params, epochs)
            append_to_csv(csv_path, params, loss, acc, trial+1)
            cache[key] = (loss, acc)

        # Collect the parameter set and its performance
        results.append((params, cache[key][0], cache[key][1]))

    # Sort results by validation loss (ascending)
    results.sort(key=lambda x: x[1])
    return results

def create_focused_param_grid(top_results, n_top=5, max_combinations=20):
    """
   Build a reduced grid of hyperparameters focused on the top performing results.
    """

    # Extract just the parameter dicts of the top performing trials
    top = [r[0] for r in top_results[:n_top]]
    grid = {}

    # For each hyperparameter, collect unique values among the top trials
    for k in top[0]:
        vals = []
        for p in top:
            v = tuple(p[k]) if k=='n_hidden' else p[k]
            if v not in vals: vals.append(v)
        # Store back as lists for 'n_hidden', else raw values
        grid[k] = [list(v) for v in vals] if k=='n_hidden' else vals

    # Compute total number of combinations in the current grid
    total = np.prod([len(v) for v in grid.values()])
    if total > max_combinations:
        print(f"Too many combinations ({total}) → reducing")

        # For each parameter, keep only the single best value based on average loss
        for k, vals in grid.items():
            if len(vals) <= 1:
                continue

            # Compute mean loss safely, ignore values without occurrences
            def mean_loss(v):
                losses = [r[1] for r in top_results
                          if (tuple(r[0][k]) if k == 'n_hidden' else r[0][k]) == v]
                return np.mean(losses) if len(losses) > 0 else np.inf

            # Find the value with lowest mean loss among top_results
            best = min(vals, key=mean_loss)
            grid[k] = [best]
    return grid

def k_fold_score(params, X, y, k=5, epochs=30, train_eval_fn=None):
    """
     Evaluate a single hyperparameter set using k-fold cross-validation.
    """
    n = X.shape[0]
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    fold_size = n // k

    losses, accs = [], []

    # Iterate over each fold
    for i in range(k):
        start, end = i*fold_size, (i+1)*fold_size if i<k-1 else n
        val_idx   = idxs[start:end]
        train_idx = np.concatenate([idxs[:start], idxs[end:]])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx],   y[val_idx]

        # Train and evaluate on current fold
        val_loss, val_acc, _ = train_eval_fn(
            X_tr, y_tr, X_va, y_va,
            params, epochs=epochs
        )
        losses.append(val_loss)
        accs.append(val_acc)

    return np.mean(losses), np.mean(accs)

def grid_search(X_train, y_train, X_val, y_val, param_grid, epochs=50, train_eval_fn=None, result_dir="result"):
    """
    Exhaustive grid search over a parameter grid, with caching and global best update.
    """

    csv_path = os.path.join(result_dir, "monk_grid_search.csv")
    ensure_csv_exists(csv_path)
    cache = read_csv_cache(csv_path)

    # Generate all combinations of hyperparameters
    combos = [dict(zip(param_grid.keys(), vals)) for vals in itertools.product(*param_grid.values())]
    best_params, best_loss, best_acc = None, np.inf, 0.0
    for i, params in enumerate(combos, 1):
        key = params_to_tuple(params)
        if key in cache:
            loss, acc = cache[key]
            print(f"[{i}/{len(combos)}] skip (cached): loss={loss:.4f}, acc={acc:.4f}")
        else:
            print(f"[{i}/{len(combos)}] eval → params={params}")
            loss, acc, _ = train_eval_fn(X_train, y_train, X_val, y_val, params, epochs)
            append_to_csv(csv_path, params, loss, acc)
            cache[key] = (loss, acc)

        # Update best if current loss is lower
        if cache[key][0] < best_loss:
            best_loss, best_acc, best_params = cache[key][0], cache[key][1], params

    # Save or update the global best parameters file
    global_best_file = os.path.join(result_dir, "grid_search_best_params.txt")
    best_params, best_loss, best_acc = update_global_best(global_best_file, best_params, best_loss, best_acc)
    print(f"Best → loss={best_loss:.4f}, acc={best_acc:.4f}")
    return best_params, best_loss, best_acc

def grid_search_cv(X_train, y_train, param_grid, k=5, epochs=50, train_eval_fn=None, result_dir="result"):
    """
    Perform grid search with k-fold cross-validation for each parameter combination.
    """
    # Create list of all parameter combinations
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Grid search CV: {len(combos)} combinazioni")

    # CSV caching similar to plain grid_search
    csv_path = os.path.join(result_dir, "monk_grid_search_cv.csv")
    ensure_csv_exists(csv_path)
    cache = read_csv_cache(csv_path)

    best_params, best_loss, best_acc = None, np.inf, 0.0

    for idx, params in enumerate(combos, 1):
        key = params_to_tuple(params)
        if key in cache:
            mean_loss, mean_acc = cache[key]
            print(f"[{idx}/{len(combos)}] skip (cached): loss={mean_loss:.4f}, acc={mean_acc:.4f}")
        else:
            print(f"[{idx}/{len(combos)}] evaluating with k-fold CV...")
            mean_loss, mean_acc = k_fold_score(params, X_train, y_train, k=k, epochs=epochs, train_eval_fn=train_eval_fn)
            append_to_csv(csv_path, params, mean_loss, mean_acc)
            cache[key] = (mean_loss, mean_acc)

        # Update best if current mean loss is lower
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_acc = mean_acc
            best_params = params
            print(f"New best: loss={best_loss:.4f}, acc={best_acc:.4f}")

    # Persist final global best grid-search-cv parameters
    global_best_file = os.path.join(result_dir, "grid_search_cv_best_params.txt")
    best_params, best_loss, best_acc = update_global_best(global_best_file, best_params, best_loss, best_acc)
    print(f"Best overall: loss={best_loss:.4f}, acc={best_acc:.4f}")
    return best_params, best_loss, best_acc