#!/usr/bin/env python3
"""Grid Search pour optimiser les hyperparamètres de my_torch."""

import csv
import itertools
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fen_parser import load_dataset, N_FEATURES, LABELS
from neural_network import NeuralNetwork


PARAM_GRID = {
    "layers": [
        [768, 128, 3],
        [768, 256, 128, 3],
        [768, 512, 128, 3],
    ],
    "learning_rate": [0.01, 0.02, 0.05],
    "batch_size": [128, 256],
    "dropout": [0.0, 0.15],
    "momentum": [0.9],
}

SEARCH_EPOCHS = 30


def evaluate_accuracy(model, X, y):
    """Calcule l'accuracy et la loss sur un ensemble de données."""
    preds = model.predict(X)
    targets = np.argmax(y, axis=1) if y.ndim == 2 else y
    accuracy = float(np.mean(preds == targets))

    # Calcul de la loss globale
    raw_output = model.forward(X, training=False)
    loss = float(model.loss.forward(raw_output, y))
    return accuracy, loss


def main():
    train_path = ROOT / "data" / "train.txt"
    val_path = ROOT / "data" / "validation.txt"
    out_dir = ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    results_csv = out_dir / "grid_search_results.csv"

    if not train_path.exists() or not val_path.exists():
        print(f"Erreur : datasets introuvables dans {ROOT / 'data'}. Lance d'abord prepare_dataset.py.")
        sys.exit(84)

    print("Chargement des données en mémoire...")
    X_train, y_train, _ = load_dataset(str(train_path), is_train=True)
    X_val, y_val, _ = load_dataset(str(val_path), is_train=True)
    print(f"Dataset chargé : {len(X_train)} train, {len(X_val)} validation.\n")

    keys = list(PARAM_GRID.keys())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    print(f"Nombre total de configurations à tester : {len(combinations)}")
    print(f"Époques par run : {SEARCH_EPOCHS}\n")

    results = []
    best_accuracy = -1.0
    best_config = None
    best_model_path = ROOT / "my_torch_network_best.nn"

    header = ["id", "layers", "learning_rate", "batch_size", "dropout", "momentum", "val_acc", "val_loss", "train_time_sec"]

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print(f"{'ID':<3} | {'Layers':<20} | {'LR':<5} | {'Batch':<5} | {'Drop':<4} | {'Val Acc':<8} | {'Val Loss':<8} | {'Temps':<6}")
    print("-" * 75)

    for idx, params in enumerate(combinations, start=1):
        config = {
            "learning_rate": params["learning_rate"],
            "momentum": params["momentum"],
            "decay": 1e-4,
            "batch_size": params["batch_size"],
            "epochs": SEARCH_EPOCHS,
            "dropout": params["dropout"],
            "print_every": None,
        }

        layers: list[int] = params["layers"]
        model = NeuralNetwork(layers, config=config)
        start_time = time.time()
        model.train(X_train, y_train, epochs=SEARCH_EPOCHS, batch_size=params["batch_size"], print_every=0)
        elapsed = time.time() - start_time

        val_acc, val_loss = evaluate_accuracy(model, X_val, y_val)

        row = [
            idx,
            "-".join(map(str, layers)),
            params["learning_rate"],
            params["batch_size"],
            params["dropout"],
            params["momentum"],
            round(val_acc, 4),
            round(val_loss, 4),
            round(elapsed, 1),
        ]

        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        print(f"{idx:<3} | {row[1]:<20} | {row[2]:<5} | {row[3]:<5} | {row[4]:<4} | {row[6]:<8.4f} | {row[7]:<8.4f} | {row[8]:<5.1f}s")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = params
            model.save(str(best_model_path))

    print("\n" + "=" * 75)
    print(f"Meilleure configuration trouvée (Accuracy: {best_accuracy:.4f}) :")
    print(best_config)
    print(f"Modèle sauvegardé sous : {best_model_path.name}")
    print(f"Résultats détaillés exportés dans : {results_csv}")


if __name__ == "__main__":
    main()