import csv
import numpy as np

def load_cup_data(file_path, has_targets=True):
    """
    Load ML-CUP24 data from a CSV file, skipping any lines that start with '#'.
    """
    ids, X_list, y_list = [], [], []

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Skip blank lines and comments
            if not row or row[0].startswith('#'):
                continue

            sample_id = row[0]
            values = list(map(float, row[1:]))

            if has_targets:
                # Last 3 columns are the targets
                inputs = values[:-3]
                targets = values[-3:]
                y_list.append(targets)
            else:
                inputs = values

            ids.append(sample_id)
            X_list.append(inputs)

    ids = np.array(ids)
    X = np.array(X_list, dtype=float)

    if has_targets:
        y = np.array(y_list, dtype=float)
        return ids, X, y
    else:
        return ids, X