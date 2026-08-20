# Neural Network Framework from Scratch — MONK & ML-CUP24

A multi-layer perceptron (MLP) framework built **entirely from scratch in NumPy** — no TensorFlow, no PyTorch, no scikit-learn. Forward pass, backpropagation, optimizers, regularization and model selection are all implemented by hand. The same framework handles **classification** (MONK's problems) and **regression** (ML-CUP24).

> University of Pisa — MSc in Computer Science (AI), Machine Learning course (Prof. Alessio Micheli), 2024/25. Project Type A.
> Team project by **Karol Cardona** and **Udit Gagnani**.

**Tech:** Python · NumPy · Matplotlib · pandas · SciPy · Jupyter

---

## Overview

The core is a single, configurable class — `MatrixMLP` — that implements a fully-connected feed-forward network trained with mini-batch gradient descent and backpropagation. Everything is vectorized with NumPy matrix operations. The class automatically switches between a *classification* and a *regression* setup based on the chosen output activation, and exposes a large hyperparameter space so that different architectures and training strategies can be explored and selected systematically.

The framework was first validated on the **MONK's problems** (a standard classification benchmark) and then applied to the **ML-CUP24** regression competition, where the best model was retrained on the full dataset and used to produce predictions on a blind test set.

## Features

**Architecture**
- Arbitrary number of fully-connected hidden layers (configurable via a list of layer sizes).
- Automatic task detection: classification (softmax / sigmoid output) vs regression (linear output).

**Activation functions:** Sigmoid, Tanh, ReLU, Leaky ReLU, ELU (hidden); Linear, Softmax, Sigmoid (output).

**Weight initialization:** Gaussian, Xavier, He, Glorot, Orthogonal, and uniform Range.

**Optimizers:** SGD with classical or **Nesterov momentum**, **Adam**, **AdamW** (decoupled weight decay), **Rprop**, and **QuickProp**.

**Regularization & training stability:** L1 and L2 penalties, inverted **dropout**, **early stopping** with patience (best weights restored), learning-rate decay, global-norm **gradient clipping** and weight clipping.

**Losses:** Binary cross-entropy, categorical cross-entropy, and MSE (each combined with the L1/L2 terms).

**Metrics:** Accuracy for classification; MSE, **Mean Euclidean Error (MEE)** and R² for regression.

**Model selection:** Hold-out and **5-fold cross-validation**; **random search** followed by a **focused grid search** around the best configurations. Results are cached to CSV to skip duplicate evaluations, and the global best is persisted to disk.

## Repository structure

```
.
├── MLP.py               # Core: the MatrixMLP class (forward, backprop, optimizers, training loop)
├── monk_loader.py       # Loads and one-hot encodes the MONK datasets
├── CUP_loader.py         # Loads the ML-CUP24 train/test CSV files
├── evaluation.py        # MEE, and evaluation for classification and regression
├── search_utils.py      # Random search, grid search, k-fold cross-validation
├── csv_utils.py         # CSV caching of trials and best-hyperparameter tracking
├── plotting.py          # Learning curves (loss / accuracy / MSE / MEE)
├── train_CUP.py         # End-to-end pipeline for the ML-CUP24 regression task
├── monk_analysis.ipynb  # Experiments and results on the MONK problems
├── cup_analysis.ipynb   # Experiments and results on ML-CUP24
└── ML_Project_Report.pptx
```

## Datasets

- **MONK's problems** — three binary classification tasks. The 6 categorical features are one-hot encoded into **17 input units**.
- **ML-CUP24** — a regression task with 3 continuous target outputs, evaluated with the **Mean Euclidean Error (MEE)**. Data is provided as `ML-CUP24-TR.csv` (training) and `ML-CUP24-TS.csv` (blind test).

> The dataset files are not included in this repository. Place the MONK files where the notebooks expect them, and the CUP files under `./Cup/` before running `train_CUP.py`.

## Results

### MONK's problems

| Problem | Config (neurons, η, momentum, activation, decay, batch, init) | MSE (Train / Test) | Accuracy (Train / Test) |
|---------|----------------------------------------------------------------|--------------------|-------------------------|
| MONK-1  | 4, 0.13, 0.9, Sigmoid, 0, 2, Gaussian                          | 0.0033 / 0.0046    | 100% / 100%             |
| MONK-2  | 4, 0.16, 0.9, Sigmoid, 0, 2, Glorot                            | 0.0000 / 0.0000    | 100% / 100%             |
| MONK-3  | 4, 0.16, 0.85, Sigmoid, 0.75, 2, Gaussian (regularized)        | 0.0276 / 0.0126    | 95.92% / 98.84%         |

### ML-CUP24

Best model selected through 5-fold CV + random/grid search, then retrained on the full training set:

| Config (neurons, η, momentum, activation, decay, batch, init, patience) | MSE (Train / Test) |
|--------------------------------------------------------------------------|--------------------|
| [50, 50, 50], 0.0003, 0.8, Leaky ReLU, 0.001, 64, Glorot, 100            | 0.2248 / 0.1869    |

The final model was used to generate predictions on the blind test set (`cup_blind_predictions.csv`).

## Getting started

### Requirements

```bash
pip install numpy matplotlib pandas scipy jupyter
```

### Run the CUP pipeline

Place the CUP data under `./Cup/`, then:

```bash
python train_CUP.py
```

This runs the full pipeline: 5-fold cross-validation, random search, focused grid search, final training on the full dataset, evaluation (MSE / MEE / R²), and blind-test predictions — with learning curves plotted at the end.

### Run the MONK experiments

Open the notebooks and run the cells:

```bash
jupyter notebook monk_analysis.ipynb
```

### Minimal usage example

```python
from MLP import MatrixMLP

model = MatrixMLP(
    n_inputs=17, n_hidden=[4], n_outputs=1,
    activation='sigmoid', output_activation='sigmoid',
    learning_rate=0.13, momentum=0.9, optimizer='sgd',
    batch_size=2, weight_init='gaussian',
    early_stopping=True, patience=50,
)
model.train(X_train, y_train, epochs=200, validation_data=(X_val, y_val))
preds = model.predict(X_test)
```

## Notes

- Random seeds are fixed (NumPy seed in the model, seed 42 in the data splits) for reproducibility.
- Because the framework is written from scratch, it is meant as a learning and experimentation tool rather than a production library — the emphasis is on understanding and controlling every step of training and optimization.
