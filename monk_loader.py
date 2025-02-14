import numpy as np

def load_monk_data(file_path, one_hot=True):
    """
    Load and normalize MONK dataset from the specified file.
    Assumes each line is formatted as:
    <label> <feat1> <feat2> ... <featN> <ignored_info>
    """
    data, targets = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                target = int(parts[0])
                features = list(map(float, parts[1:-1]))
                data.append(features)
                targets.append(target)
    data = np.array(data)
    targets = np.array(targets)
    # Normalize features to the range [0, 1]
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    range_vals = data_max - data_min
    range_vals[range_vals == 0] = 1
    data = (data - data_min) / range_vals
    if one_hot:
        targets = one_hot_encode(targets)
    return data, targets

def one_hot_encode(labels):
    """
    Convert class labels to one-hot encoded vectors.
    If labels are in {1,2,...}, they are first shifted to 0-indexed.
    """
    labels = np.array(labels)
    if np.min(labels) == 1:
        labels -= 1
    num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels.astype(int)]
