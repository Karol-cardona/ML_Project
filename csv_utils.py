import ast
import os
import csv

def save_global_best(global_best_file, params, val_loss, val_acc):
    """Save the best hyperparameters to a file"""
    os.makedirs(os.path.dirname(global_best_file), exist_ok=True)
    with open(global_best_file, 'w') as f:
        f.write("Best Hyperparameters:\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"Validation Loss: {val_loss}\n")
        f.write(f"Validation Accuracy: {val_acc}\n")

def load_global_best(global_best_file):
    """Load the best hyperparameters from a file"""
    if not os.path.exists(global_best_file):
        return None
    best = {}
    with open(global_best_file) as f:
        for line in f:
            if ':' not in line: continue
            k, v = line.strip().split(':', 1)
            best[k.strip()] = v.strip()
    return best

def update_global_best(global_best_file, new_params, new_val_loss, new_val_acc):
    """Update the global best parameters if the new ones are better"""
    best = load_global_best(global_best_file)
    if best:
        try:
            best_loss = float(best.get("Validation Loss", best.get("val_loss", 1e12)))
        except:
            best_loss = float('inf')

        if new_val_loss < best_loss:
            save_global_best(global_best_file, new_params, new_val_loss, new_val_acc)
            return new_params, new_val_loss, new_val_acc
        else:
            # Keep the old ones
            fixed = {}
            for k, v in best.items():
                if k in ("Validation Loss", "Validation Accuracy", "val_loss", "val_accuracy"):
                    fixed[k] = v
                else:
                    try:
                        fixed[k] = ast.literal_eval(v)
                    except (SyntaxError, ValueError):
                        fixed[k] = v
            return fixed, best_loss, float(best.get("Validation Accuracy", best.get("val_accuracy", 0.0)))
    else:
        save_global_best(global_best_file, new_params, new_val_loss, new_val_acc)
        return new_params, new_val_loss, new_val_acc

def params_to_tuple(params):
    """Convert parameters dictionary to a hashable tuple for caching"""
    n_hidden = tuple(params['n_hidden']) if isinstance(params['n_hidden'], list) else params['n_hidden']
    init_range = tuple(params.get('init_range', ())) if 'init_range' in params else None
    return (
        n_hidden,
        params.get('learning_rate'),
        params.get('reg_lambda'),
        params.get('l1_lambda'),
        params.get('dropout_rate'),
        params.get('momentum'),
        params.get('activation'),
        params.get('optimizer'),
        params.get('lr_decay'),
        params.get('batch_size'),
        params.get('weight_init', 'range'),
        init_range,
        params.get('patience', 15)
    )

def ensure_csv_exists(csv_path, is_random_search=False):
    """Create CSV file with proper headers if it doesn't exist"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = [
                "n_hidden","learning_rate","reg_lambda","l1_lambda","dropout_rate",
                "momentum","activation","optimizer","lr_decay","batch_size",
                "weight_init","init_range","patience","val_loss","val_accuracy"
            ]
            if is_random_search:
                headers.append("trial_number")
            writer.writerow(headers)

def read_csv_cache(csv_path):
    """Read CSV cache of previously evaluated hyperparameters"""
    cache = {}
    if not os.path.exists(csv_path):
        return cache
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = params_to_tuple({
                    'n_hidden': ast.literal_eval(row['n_hidden']) if row['n_hidden'].startswith('[') else int(row['n_hidden']),
                    'learning_rate': float(row['learning_rate']),
                    'reg_lambda': float(row['reg_lambda']),
                    'l1_lambda': float(row['l1_lambda']),
                    'dropout_rate': float(row['dropout_rate']),
                    'momentum': float(row['momentum']),
                    'activation': row['activation'],
                    'optimizer': row['optimizer'],
                    'lr_decay': float(row['lr_decay']),
                    'batch_size': int(row['batch_size']),
                    'weight_init': row.get('weight_init', 'range'),
                    'init_range': ast.literal_eval(row.get('init_range','()')) if row.get('init_range') else None,
                    'patience': int(row.get('patience', 10))
                })
                cache[key] = (float(row['val_loss']), float(row['val_accuracy']))
            except (ValueError, SyntaxError, KeyError):
                # Skip rows with parsing errors
                continue
    return cache

def append_to_csv(csv_path, params, val_loss, val_accuracy, trial_number=None):
    """Append new results to the CSV file"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = [
        params['n_hidden'], params['learning_rate'], params['reg_lambda'],
        params['l1_lambda'], params['dropout_rate'],
        params['momentum'], params['activation'], params['optimizer'],
        params['lr_decay'], params['batch_size'],
        params.get('weight_init', 'range'), params.get('init_range', ()),
        params.get('patience', 15), val_loss, val_accuracy
    ]
    if trial_number is not None:
        row.append(trial_number)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)