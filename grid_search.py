#!/usr/bin/env python3
"""Grid Search multi-GPU/CPU pour optimiser les hyperparamètres de my_torch."""

import argparse
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
from neural_network import NeuralNetwork, CUPY_AVAILABLE, to_cpu

PARAM_GRID = {
    "layers": [
        [768, 512, 256, 128, 64, 3],
    ],
    "learning_rate": [0.02, 0.05],
    "batch_size": [128, 256],
    "dropout": [0.0, 0.15],
    "momentum": [0.9],
}

DEFAULT_EPOCHS = 30


def evaluate_accuracy(model, X, y, batch_size=256):
    """Calcule l'accuracy et la perte par mini-lots pour préserver la mémoire."""
    preds = model.predict(X)
    targets = np.argmax(y, axis=1) if y.ndim == 2 else y
    accuracy = float(np.mean(preds == targets))

    total_loss = 0.0
    n_samples = len(X)
    for start in range(0, n_samples, batch_size):
        batch_x = X[start:start + batch_size]
        batch_y = y[start:start + batch_size]
        if model.xp is not np:
            batch_x = model.xp.asarray(batch_x)
            batch_y = model.xp.asarray(batch_y)
        raw_output = model.forward(batch_x, training=False)
        loss = model.loss.forward(raw_output, batch_y)
        total_loss += float(to_cpu(loss)) * len(batch_x)

    return accuracy, total_loss / n_samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_train = ROOT / "data" / "train_augmented.txt"
    if not default_train.exists():
        default_train = ROOT / "data" / "train.txt"

    parser.add_argument("--train", type=Path, default=default_train, help="Chemin du dataset d'entraînement")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "validation.txt", help="Chemin du dataset de validation")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Nombre d'époques par configuration")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Périphérique de calcul (auto/cuda/cpu)")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "grid_search_results.csv", help="Chemin du fichier CSV de sortie")
    parser.add_argument("--save-best", type=Path, default=ROOT / "my_torch_network_best.nn", help="Chemin de sauvegarde du meilleur modèle")
    return parser.parse_args()


def select_device(device_arg: str) -> str:
    if device_arg in ("auto", "cuda"):
        if not CUPY_AVAILABLE:
            if device_arg == "cuda":
                sys.stderr.write("Erreur : CuPy n'est pas installé dans le venv pour l'accélération CUDA.\n")
                sys.stderr.write("Exécutez : pip install cupy-cuda12x[ctk]\n")
                sys.exit(84)
            print("[Info] CuPy non disponible. Utilisation du CPU (NumPy).")
            return "cpu"
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="cupy")
            import cupy as cp
        try:
            probe = cp.array([1.0], dtype=cp.float32)
            _ = probe + 1.0
            _ = cp.dot(probe, probe)
            device_id = cp.cuda.Device().id
            gpu_name = cp.cuda.runtime.getDeviceProperties(device_id)["name"].decode()
            _, total_mem = cp.cuda.Device().mem_info
            vram_gib = total_mem / (1024 ** 3)
            print(f"[GPU Détecté] {gpu_name} ({vram_gib:.2f} GiB VRAM) — Accélération CUDA CuPy activée.")
            return "cuda"
        except Exception as err:
            if device_arg == "cuda":
                sys.stderr.write(f"Erreur d'initialisation CUDA : {err}\n")
                sys.stderr.write("Installez le toolkit CuPy complet avec : pip install cupy-cuda12x[ctk]\n")
                sys.exit(84)
            print(f"[Info] CUDA indisponible ({err}). Utilisation du CPU (NumPy).")
            return "cpu"
    return "cpu"


def main():
    args = parse_args()
    device_type = select_device(args.device)

    if not args.train.exists() or not args.val.exists():
        sys.stderr.write(f"Erreur : Datasets introuvables ({args.train} ou {args.val}).\n")
        sys.exit(84)

    print(f"Chargement des datasets :\n  - Train : {args.train.name}\n  - Val   : {args.val.name}")
    X_train, y_train, _ = load_dataset(str(args.train), is_train=True)
    X_val, y_val, _ = load_dataset(str(args.val), is_train=True)
    print(f"Données chargées : {len(X_train)} train, {len(X_val)} validation.\n")

    keys = list(PARAM_GRID.keys())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    print(f"Nombre total de configurations : {len(combinations)}")
    print(f"Époques par run : {args.epochs}")
    print(f"Moteur d'exécution : NeuralNetwork sur {device_type.upper()}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = ["id", "layers", "learning_rate", "batch_size", "dropout", "momentum", "epochs", "val_acc", "val_loss", "train_time_sec"]

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

    best_acc = -1.0
    best_config = None

    print(f"{'ID':<3} | {'Layers':<20} | {'LR':<5} | {'Batch':<5} | {'Drop':<4} | {'Val Acc':<8} | {'Val Loss':<8} | {'Temps':<6}")
    print("-" * 75)

    for idx, params in enumerate(combinations, start=1):
        layers = params["layers"]
        lr = params["learning_rate"]
        bs = params["batch_size"]
        dropout = params["dropout"]
        momentum = params["momentum"]
        decay = 1e-4

        config = {
            "learning_rate": lr,
            "momentum": momentum,
            "decay": decay,
            "batch_size": bs,
            "epochs": args.epochs,
            "dropout": dropout,
            "print_every": None,
            "device": device_type,
        }

        model = NeuralNetwork(layers, config=config, device=device_type)
        start_time = time.time()
        model.train(X_train, y_train, epochs=args.epochs, batch_size=bs, print_every=0)
        elapsed = time.time() - start_time

        val_acc, val_loss = evaluate_accuracy(model, X_val, y_val, batch_size=bs)

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = {**params, "epochs": args.epochs, "decay": decay}
            model.save(str(args.save_best))

        row = [
            idx,
            "-".join(map(str, layers)),
            lr,
            bs,
            dropout,
            momentum,
            args.epochs,
            round(val_acc, 4),
            round(val_loss, 4),
            round(elapsed, 1),
        ]

        with open(args.output, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)

        print(f"{idx:<3} | {row[1]:<20} | {row[2]:<5} | {row[3]:<5} | {row[4]:<4} | {row[7]:<8.4f} | {row[8]:<8.4f} | {row[9]:<5.1f}s")

    print("\n" + "=" * 75)
    print(f"Meilleure configuration (Validation Accuracy: {best_acc:.4f}) :")
    print(best_config)
    print(f"Modèle exporté sous : {args.save_best.name}")
    print(f"Résultats détaillés enregistrés dans : {args.output}")


if __name__ == "__main__":
    main()
