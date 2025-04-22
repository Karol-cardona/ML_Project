import numpy as np

def load_monk_data(file_path, one_hot=True):
    """
    Load and normalize MONK dataset from the specified file.
    """
    data, targets = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                target = int(parts[0])
                features = list(map(float, parts[1:7]))
                data.append(features)
                targets.append(target)

    data = np.array(data)
    targets = np.array(targets)

    # One-hot encode the categorical features
    encoded_data = []
    cardinalities = [3, 3, 2, 3, 4, 2]  # Number of values for each feature
    for instance in data:
        enc = []
        for i, val in enumerate(instance):
            vec = np.zeros(cardinalities[i])
            vec[int(val)-1] = 1  # Values start from 1
            enc.extend(vec)
        encoded_data.append(enc)

    data = np.array(encoded_data)

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

def augment_monk_data(X, y, n_copies=3):
    """
    Augment MONK dataset by creating copies with shuffled samples.
    """
    augmented_X, augmented_y = [], []
    for _ in range(n_copies):
        # Create copies with randomly shuffled indices
        idx = np.random.permutation(X.shape[0])
        shuffled_X = X[idx]
        augmented_X.append(shuffled_X)
        augmented_y.append(y)

    return np.vstack(augmented_X), np.vstack(augmented_y)