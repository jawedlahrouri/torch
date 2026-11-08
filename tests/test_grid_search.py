import json
import pytest
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_network import NeuralNetwork
from grid_search import PARAM_GRID


def test_param_grid_structure():
    assert "layers" in PARAM_GRID
    assert "learning_rate" in PARAM_GRID
    assert "batch_size" in PARAM_GRID
    assert "dropout" in PARAM_GRID
    assert "momentum" in PARAM_GRID

    assert PARAM_GRID["layers"] == [[768, 512, 256, 128, 64, 3]]
    assert set(PARAM_GRID["learning_rate"]) == {0.02, 0.05}
    assert set(PARAM_GRID["batch_size"]) == {128, 256}
    assert set(PARAM_GRID["dropout"]) == {0.0, 0.15}
    assert PARAM_GRID["momentum"] == [0.9]


def test_exported_nn_compatibility(tmp_path):
    layers = [768, 512, 256, 128, 64, 3]
    json_structure = []
    for i in range(len(layers) - 1):
        w = np.random.randn(layers[i], layers[i + 1]) * 0.01
        b = np.zeros((1, layers[i + 1]))
        json_structure.append({
            "type": "Layer_Dense",
            "weights": w.tolist(),
            "biases": b.tolist(),
        })
        if i < len(layers) - 2:
            json_structure.append({"type": "Activation_ReLU"})
            json_structure.append({"type": "Layer_Dropout", "rate": 0.15})

    payload = {
        "config": {
            "learning_rate": 0.02,
            "momentum": 0.9,
            "decay": 1e-4,
            "batch_size": 128,
            "epochs": 30,
            "dropout": 0.15,
        },
        "layers": json_structure,
    }

    model_path = tmp_path / "test_exported.nn"
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded = NeuralNetwork.load(str(model_path))
    assert len(loaded.trainable_layers) == 5
    assert loaded.trainable_layers[0].weights.shape == (768, 512)
    assert loaded.trainable_layers[-1].weights.shape == (64, 3)

    dummy_x = np.random.randn(2, 768)
    preds = loaded.predict(dummy_x)
    assert len(preds) == 2
    assert all(p in (0, 1, 2) for p in preds)


def test_device_fallback_cpu():
    from grid_search import evaluate_accuracy, select_device
    device = select_device("cpu")
    assert device == "cpu"

    model = NeuralNetwork([10, 4, 3], device="cpu")
    assert model.device == "cpu"
    X = np.random.randn(20, 10)
    y = np.random.randint(0, 3, size=20)
    acc, loss = evaluate_accuracy(model, X, y, batch_size=5)
    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0

