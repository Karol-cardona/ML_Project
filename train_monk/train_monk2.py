import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_logic import training

monk2_param_space = {
    'n_hidden': [[3],[4]],
    'learning_rate': list(np.arange(0.01, 0.2, 0.15)),
    'dropout_rate': list(np.arange(0.0, 0.3, 0.05)),
    'momentum': list(np.arange(0.7, 0.9, 0.05)),
    'activation': ['sigmoid'],
    'optimizer': ['sgd'],
    'reg_lambda': [0.0],
    'l1_lambda': [0.0],
    'lr_decay': [0.0],
    'batch_size': [2, 4, 8],
    'weight_init': ['gaussian', 'he', 'glorot'],
    'init_range': [(-0.1, 0.1), (-0.5, 0.5)],
}

training(2, monk2_param_space)