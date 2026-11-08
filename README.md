# MyTorch — Neural Network from Scratch

A modular artificial neural network implementation built from scratch in **Python** using only **NumPy**, without any high-level machine learning frameworks (no PyTorch, TensorFlow, etc.).

The project is designed to classify chessboard configurations represented in **FEN** (*Forsyth-Edwards Notation*) into three discrete states:
- `Nothing`: Standard or non-threatening position.
- `Check`: King is under check.
- `Checkmate`: King is in checkmate.

---

## Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Neural Network Generation (`my_torch_generator`)](#1-neural-network-generation-my_torch_generator)
  - [2. Training and Prediction (`my_torch_analyzer`)](#2-training-and-prediction-my_torch_analyzer)
- [File Formats](#file-formats)
  - [Configuration Files (`.conf`)](#configuration-files-conf)
  - [Model Files (`.nn`)](#model-files-nn)
  - [FEN Chessboard Datasets](#fen-chessboard-datasets)
- [Additional Tooling](#additional-tooling)
- [Testing](#testing)

---

## Features

- **Modular Neural Network Architecture (`src/neural_network.py`)**:
  - Fully connected dense layers (`Layer_Dense`) with **He** and **Xavier** weight initializations.
  - Activation functions: **ReLU** and **Softmax**.
  - Regularization: **Dropout** layer.
  - Loss function: **Categorical Cross-Entropy**.
  - Optimizer: **SGD** with momentum and learning rate decay.
  - Mini-batch gradient descent with dataset shuffling.
- **Vectorized FEN Parser (`src/fen_parser.py`)**:
  - One-hot binary encoding converting a chessboard into a 768-dimensional feature vector (64 squares × 12 piece types).
  - Symmetry operations and board flipping for dataset augmentation.
- **Human-Readable Model Serialization (`.nn`)**:
  - Saved and loaded via formatted **JSON** structure, with backward compatibility for legacy binary pickle archives.

---

## Project Architecture

```text
.
├── configs/                  # Sample network configurations (basic, medium, full, test)
├── dataset/                  # Base chessboard datasets
├── dataset_augmented/        # Augmented datasets (color flipped / board flipped)
├── scripts/
│   ├── augment_dataset.py    # Dataset augmentation script
│   ├── evaluate.py           # Evaluation script generating confusion matrices
│   ├── grid_search.py        # Hyperparameter search and validation
│   └── prepare_dataset.py    # Train/validation split utility
├── src/
│   ├── fen_parser.py         # FEN string parsing and board-to-vector conversion
│   └── neural_network.py     # Core neural network implementation
├── tests/                    # Pytest test suite
├── my_torch_analyzer         # CLI entry point for training and prediction
├── my_torch_generator        # CLI entry point to generate untrained networks
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- `pip`

### Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Neural Network Generation (`my_torch_generator`)

Generates one or more uninitialized neural networks based on specified configuration files.

```bash
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2 ...]
```

**Example:**
```bash
# Generates 2 networks based on configs/medium.conf (medium_1.nn and medium_2.nn)
./my_torch_generator configs/medium.conf 2
```

---

### 2. Training and Prediction (`my_torch_analyzer`)

#### Training Mode (`--train`)
Trains the network on annotated chessboard positions. By default, weights are updated in-place inside `LOADFILE`. Specify `--save` to write to a distinct target path.

```bash
# Train network and save output into a new file
./my_torch_analyzer --train --save model_trained.nn model_init.nn dataset/train.txt

# Train network and overwrite model_init.nn in-place
./my_torch_analyzer --train model_init.nn dataset/train.txt
```

#### Prediction Mode (`--predict`)
Predicts the classification for each chessboard in the input file and outputs the results to standard output.

```bash
./my_torch_analyzer --predict model_trained.nn dataset/test.txt
```

---

## File Formats

### Configuration Files (`.conf`)

Two formats are accepted:

1. **Compact format (layer sizes only)**:
   ```text
   768,128,64,3
   ```
2. **JSON format (with hyperparameters)**:
   ```json
   {
     "layers": [768, 256, 128, 64, 3],
     "learning_rate": 0.02,
     "momentum": 0.9,
     "decay": 0.0001,
     "batch_size": 256,
     "epochs": 100,
     "dropout": 0.1,
     "print_every": 5
   }
   ```

### Model Files (`.nn`)

Networks are saved in formatted, human-readable **JSON**:

```json
{
  "config": {
    "learning_rate": 0.02,
    "momentum": 0.9,
    "decay": 0.0001,
    "batch_size": 256,
    "epochs": 100,
    "dropout": 0.1,
    "print_every": 5
  },
  "layers": [
    {
      "type": "Layer_Dense",
      "weights": [[...], [...]],
      "biases": [[...]]
    },
    {
      "type": "Activation_ReLU"
    },
    {
      "type": "Layer_Dropout",
      "rate": 0.1
    }
  ]
}
```

### FEN Chessboard Datasets

Each line contains a FEN string representing the position, followed by the target label for training:

```text
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate
rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check
```

---

## Additional Tooling

- **Model Evaluation (`scripts/evaluate.py`)**:
  Computes accuracy, per-class recall, and displays or exports a confusion matrix:
  ```bash
  python scripts/evaluate.py my_torch_network.nn dataset/test.txt --png confusion.png
  ```

- **Hyperparameter Grid Search (`scripts/grid_search.py`)**:
  Iterates over combinations of learning rates, architectures, batch sizes, and dropout values to benchmark optimal configurations.

- **Data Augmentation (`scripts/augment_dataset.py`)**:
  Expands dataset size by flipping boards and mirroring piece colors.

---

## Testing

Run the automated test suite with `pytest`:

```bash
python -m pytest
```
