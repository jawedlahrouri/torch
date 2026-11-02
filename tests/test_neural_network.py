import os
import sys
from pathlib import Path
import numpy as np

# Ajout du chemin du module pour importation directe
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "neural_network.py"))

from neural_network import (
    Layer_Dense,
    Activation_ReLU,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Optimizer_SGD,
    NeuralNetwork,
    spiral_data
)


def test_inner_classes_accessible_on_neural_network():
    assert hasattr(NeuralNetwork, "Layer_Dense")
    assert hasattr(NeuralNetwork, "Activation_ReLU")
    assert hasattr(NeuralNetwork, "Activation_Softmax")
    assert hasattr(NeuralNetwork, "Loss")
    assert hasattr(NeuralNetwork, "Loss_CategoricalCrossentropy")
    assert hasattr(NeuralNetwork, "Activation_Softmax_Loss_CategoricalCrossentropy")
    assert hasattr(NeuralNetwork, "Optimizer_SGD")


def test_modular_chess_architecture_instantiation():
    arch = [768, 256, 128, 64, 3]
    net = NeuralNetwork(layer_sizes=arch)

    assert len(net.layers) == 7
    assert len(net.trainable_layers) == 4
    assert net.layers[0].weights.shape == (768, 256)
    assert net.layers[2].weights.shape == (256, 128)
    assert net.layers[4].weights.shape == (128, 64)
    assert net.layers[6].weights.shape == (64, 3)


def test_manual_layer_addition():
    net = NeuralNetwork()
    net.add(NeuralNetwork.Layer_Dense(768, 128))
    net.add(NeuralNetwork.Activation_ReLU())
    net.add(NeuralNetwork.Layer_Dense(128, 3))

    assert len(net.layers) == 3
    assert len(net.trainable_layers) == 2


def test_forward_pass_chess_dimensions():
    batch_size = 16
    X = np.random.randn(batch_size, 768)
    net = NeuralNetwork(layer_sizes=[768, 256, 128, 64, 3])

    output = net.forward(X)
    assert output.shape == (batch_size, 3)


def test_train_step_and_convergence():
    X, y = spiral_data(50, 3)
    net = NeuralNetwork(
        layer_sizes=[2, 32, 3],
        loss=Activation_Softmax_Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_SGD(learning_rate=1.0, momentum=0.9)
    )

    initial_loss, _ = net.train_step(X, y)
    for _ in range(50):
        final_loss, accuracy = net.train_step(X, y)

    assert final_loss < initial_loss
    assert 0.0 <= accuracy <= 1.0


def test_predict():
    X, _ = spiral_data(10, 3)
    net = NeuralNetwork(
        layer_sizes=[2, 16, 3],
        loss=Activation_Softmax_Loss_CategoricalCrossentropy()
    )
    predictions = net.predict(X)

    assert predictions.shape == (30,)
    assert set(np.unique(predictions)).issubset({0, 1, 2})


def test_save_and_load_binary(tmp_path):
    filepath = str(tmp_path / "model.nn")
    net = NeuralNetwork([2, 16, 8, 3])
    X = np.random.randn(5, 2)
    original_output = net.forward(X)

    net.save(filepath)
    assert os.path.exists(filepath)

    loaded_net = NeuralNetwork.load(filepath)
    assert len(loaded_net.layers) == len(net.layers)
    loaded_output = loaded_net.forward(X)
    assert np.allclose(original_output, loaded_output)


def test_save_and_load_instance(tmp_path):
    filepath = str(tmp_path / "model_instance.nn")
    net = NeuralNetwork([2, 32, 3])
    net.save(filepath)

    empty_net = NeuralNetwork()
    empty_net.load(filepath)
    assert len(empty_net.layers) == len(net.layers)


def test_save_and_load_json(tmp_path):
    filepath = str(tmp_path / "model.json")
    net = NeuralNetwork([2, 8, 3])
    X = np.random.randn(4, 2)
    original_output = net.forward(X)

    net.save(filepath)
    assert os.path.exists(filepath)

    loaded_net = NeuralNetwork.load(filepath)
    loaded_output = loaded_net.forward(X)
    assert np.allclose(original_output, loaded_output)
