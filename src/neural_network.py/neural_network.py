import json
import os
import pickle
import sys
import numpy as np

number_case = 64
number_pawn = 32

def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y

class _DualLoad:
    def __get__(self, instance, owner):
        if instance is None:
            return owner._load_from_file
        return instance._load_into_self

class NeuralNetwork:
    class Layer_Dense:
        def __init__(self, n_inputs, n_neurons, init_method="he"):
            if init_method == "xavier":
                self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1.0 / n_inputs)
            else:
                self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
            self.biases = np.zeros((1, n_neurons))

        def forward(self, inputs):
            self.inputs = inputs
            self.output = np.dot(inputs, self.weights) + self.biases

        def backward(self, dvalues):
            self.dweights = np.dot(self.inputs.T, dvalues)
            self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
            self.dinputs = np.dot(dvalues, self.weights.T)

    class Activation_ReLU:
        def forward(self, inputs):
            self.inputs = inputs
            self.output = np.maximum(0, inputs)

        def backward(self, dvalues):
            self.dinputs = dvalues.copy()
            self.dinputs[self.inputs <= 0] = 0

    class Layer_Dropout:
        def __init__(self, rate):
            self.rate = 1 - rate

        def forward(self, inputs, training=True):
            self.inputs = inputs
            if not training:
                self.output = inputs.copy()
                return
            self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
            self.output = inputs * self.binary_mask

        def backward(self, dvalues):
            self.dinputs = dvalues * self.binary_mask

    class Activation_Softmax:
        def forward(self, inputs):
            self.inputs = inputs
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        def backward(self, dvalues):
            sum_sd = np.sum(self.output * dvalues, axis=1, keepdims=True)
            self.dinputs = self.output * (dvalues - sum_sd)

    class Loss:
        def calculate(self, output, y):
            return np.mean(self.forward(output, y))

    class Loss_CategoricalCrossentropy(Loss):
        def forward(self, y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            target = np.argmax(y_true, axis=1) if y_true.ndim == 2 else y_true
            return -np.log(y_pred_clipped[range(samples), target])

        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            if y_true.ndim == 1:
                y_true = np.eye(dvalues.shape[1])[y_true]
            self.dinputs = (-y_true / dvalues) / samples

    class Activation_Softmax_Loss_CategoricalCrossentropy:
        def __init__(self):
            self.activation = NeuralNetwork.Activation_Softmax()
            self.loss = NeuralNetwork.Loss_CategoricalCrossentropy()

        def forward(self, inputs, y_true):
            self.activation.forward(inputs)
            self.output = self.activation.output
            return self.loss.calculate(self.output, y_true)

        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            target = np.argmax(y_true, axis=1) if y_true.ndim == 2 else y_true
            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), target] -= 1
            self.dinputs /= samples

    class Optimizer_SGD:
        def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0
            self.momentum = momentum

        def pre_update_params(self):
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

        def update_params(self, layer):
            if self.momentum:
                if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)
                weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
                layer.weight_momentums = weight_updates
                bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
                layer.bias_momentums = bias_updates
            else:
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases
            layer.weights += weight_updates
            layer.biases += bias_updates

        def post_update_params(self):
            self.iterations += 1

    def __init__(self, layer_sizes=None, loss=None, loss_activation=None, optimizer=None):
        self.layers = []
        if layer_sizes:
            if isinstance(layer_sizes[0], int):
                for i in range(len(layer_sizes) - 1):
                    self.layers.append(NeuralNetwork.Layer_Dense(layer_sizes[i], layer_sizes[i + 1]))
                    if i < len(layer_sizes) - 2:
                        self.layers.append(NeuralNetwork.Activation_ReLU())
            else:
                self.layers = list(layer_sizes)

        self.loss = loss or loss_activation or NeuralNetwork.Activation_Softmax_Loss_CategoricalCrossentropy()
        self.loss_activation = self.loss
        self.optimizer = optimizer or NeuralNetwork.Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)

    def add(self, layer):
        self.layers.append(layer)

    @property
    def trainable_layers(self):
        return [layer for layer in self.layers if isinstance(layer, NeuralNetwork.Layer_Dense)]

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def backward(self, dvalues):
        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs
        return dvalues

    def train_step(self, X, y):
        loss = self.loss.forward(self.forward(X), y)
        y_true = np.argmax(y, axis=1) if y.ndim == 2 else y
        accuracy = np.mean(np.argmax(self.loss.output, axis=1) == y_true)

        self.loss.backward(self.loss.output, y)
        self.backward(self.loss.dinputs)

        self.optimizer.pre_update_params()
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

        return loss, accuracy

    def train(self, X, y, epochs=1001, batch_size=64, print_every=200):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[indices], y[indices]

            batch_losses, batch_accuracies = [], []
            for start in range(0, len(X), batch_size):
                batch_X = X_shuffled[start:start + batch_size]
                batch_y = y_shuffled[start:start + batch_size]
                loss, accuracy = self.train_step(batch_X, batch_y)
                batch_losses.append(loss)
                batch_accuracies.append(accuracy)

            if print_every and epoch % print_every == 0:
                avg_loss = np.mean(batch_losses)
                avg_acc = np.mean(batch_accuracies)
                print(f"epoch: {epoch:4d} | loss: {avg_loss:.4f} | acc: {avg_acc:.4f} | lr: {self.optimizer.current_learning_rate:.4f}")

    def predict(self, X):
        output = self.forward(X)
        self.loss.activation.forward(output)
        return np.argmax(self.loss.activation.output, axis=1)

    def save(self, filepath):
        structure = []
        for layer in self.layers:
            if isinstance(layer, NeuralNetwork.Layer_Dense):
                structure.append({
                    "type": "Layer_Dense",
                    "weights": layer.weights,
                    "biases": layer.biases,
                })
            else:
                structure.append({"type": layer.__class__.__name__})

        if str(filepath).endswith(".json"):
            json_structure = []
            for item in structure:
                if item["type"] == "Layer_Dense":
                    json_structure.append({
                        "type": "Layer_Dense",
                        "weights": item["weights"].tolist(),
                        "biases": item["biases"].tolist(),
                    })
                else:
                    json_structure.append(item)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_structure, f, indent=2)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(structure, f)

    @classmethod
    def _load_from_file(cls, filepath):
        instance = cls()
        return instance._load_into_self(filepath)

    def _load_into_self(self, filepath):
        structure = None
        try:
            with open(filepath, "rb") as f:
                structure = pickle.load(f)
        except Exception:
            with open(filepath, "r", encoding="utf-8") as f:
                structure = json.load(f)

        self.layers = []
        for item in structure:
            layer_type = item["type"]
            if layer_type == "Layer_Dense":
                weights = np.array(item["weights"])
                biases = np.array(item["biases"])
                dense = NeuralNetwork.Layer_Dense(weights.shape[0], weights.shape[1])
                dense.weights = weights
                dense.biases = biases
                self.layers.append(dense)
            elif hasattr(NeuralNetwork, layer_type):
                self.layers.append(getattr(NeuralNetwork, layer_type)())
        return self

    load = _DualLoad()


for _cls in (NeuralNetwork.Layer_Dense, NeuralNetwork.Activation_ReLU, NeuralNetwork.Layer_Dropout,
            NeuralNetwork.Activation_Softmax, NeuralNetwork.Loss, NeuralNetwork.Loss_CategoricalCrossentropy,
            NeuralNetwork.Activation_Softmax_Loss_CategoricalCrossentropy, NeuralNetwork.Optimizer_SGD):
    globals()[_cls.__name__] = _cls


if __name__ == "__main__":
    np.random.seed(0)
    X, y = spiral_data(100, 3)

    nn = NeuralNetwork([2, 64, 3])
    nn.train(X, y, epochs=1001, print_every=200)
    nn.save("spiral_model.nn")
    loaded_nn = NeuralNetwork.load("spiral_model.nn")
    assert np.allclose(nn.forward(X), loaded_nn.forward(X))
    print(f"\nModèle exporté et rechargé avec succès ({len(loaded_nn.layers)} couches).")
    if os.path.exists("spiral_model.nn"):
        os.remove("spiral_model.nn")
    chess_architectures = [
        [768, 256, 128, 64, 3],
        [768, 128, 64, 3],
        [768, 64, 3]
    ]

    X_chess = np.random.randn(128, 768)
    y_chess = np.random.randint(0, 3, size=128)

    print("\nTest de comparaison d'architectures échiquier :")
    for arch in chess_architectures:
        model = NeuralNetwork(arch, optimizer=NeuralNetwork.Optimizer_SGD(learning_rate=0.01, decay=1e-3, momentum=0.9))
        loss, acc = model.train_step(X_chess, y_chess)
        print(f"Architecture {arch} -> Loss: {loss:.4f} | Acc: {acc:.4f}")
