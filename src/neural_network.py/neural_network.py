import sys
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

number_case = 64
number_pawn = 32

nnfs.init()


class NeuralNetwork:
    class Layer_Dense:
        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
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

    class Activation_Softmax:
        def forward(self, inputs):
            self.inputs = inputs
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probabilities

        def backward(self, dvalues):
            self.dinputs = np.empty_like(dvalues)
            for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
                single_output = single_output.reshape(-1, 1)
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    class Loss:
        def calculate(self, output, y):
            sample_losses = self.forward(output, y)
            data_loss = np.mean(sample_losses)
            return data_loss

    class Loss_CategoricalCrossentropy(Loss):
        def forward(self, y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[range(samples), y_true]
            elif len(y_true.shape) == 2:
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods

        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            labels = len(dvalues[0])
            if len(y_true.shape) == 1:
                y_true = np.eye(labels)[y_true]
            self.dinputs = -y_true / dvalues
            self.dinputs = self.dinputs / samples

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
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)
            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples

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

    def __init__(self, layer_sizes=None, loss_activation=None, loss=None, optimizer=None):
        self.layers = []
        if layer_sizes:
            if isinstance(layer_sizes[0], int):
                for i in range(len(layer_sizes) - 1):
                    self.layers.append(NeuralNetwork.Layer_Dense(layer_sizes[i], layer_sizes[i + 1]))
                    if i < len(layer_sizes) - 2:
                        self.layers.append(NeuralNetwork.Activation_ReLU())
            else:
                self.layers = list(layer_sizes)

        selected_loss = loss if loss is not None else loss_activation
        self.loss_activation = selected_loss if selected_loss is not None else NeuralNetwork.Activation_Softmax_Loss_CategoricalCrossentropy()
        self.loss = self.loss_activation
        self.optimizer = optimizer if optimizer is not None else NeuralNetwork.Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)

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
        output = self.forward(X)
        loss = self.loss_activation.forward(output, y)

        predictions = np.argmax(self.loss_activation.output, axis=1)
        y_true = np.argmax(y, axis=1) if len(y.shape) == 2 else y
        accuracy = np.mean(predictions == y_true)

        self.loss_activation.backward(self.loss_activation.output, y)
        self.backward(self.loss_activation.dinputs)

        self.optimizer.pre_update_params()
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

        return loss, accuracy

    def train(self, X, y, epochs=1001, print_every=200):
        for epoch in range(epochs):
            loss, accuracy = self.train_step(X, y)
            if print_every and epoch % print_every == 0:
                print(f"epoch: {epoch:4d} | loss: {loss:.4f} | acc: {accuracy:.4f} | lr: {self.optimizer.current_learning_rate:.4f}")

    def predict(self, X):
        output = self.forward(X)
        self.loss_activation.activation.forward(output)
        return np.argmax(self.loss_activation.activation.output, axis=1)


Layer_Dense = NeuralNetwork.Layer_Dense
Activation_ReLU = NeuralNetwork.Activation_ReLU
Activation_Softmax = NeuralNetwork.Activation_Softmax
Loss = NeuralNetwork.Loss
Loss_CategoricalCrossentropy = NeuralNetwork.Loss_CategoricalCrossentropy
Activation_Softmax_Loss_CategoricalCrossentropy = NeuralNetwork.Activation_Softmax_Loss_CategoricalCrossentropy
Optimizer_SGD = NeuralNetwork.Optimizer_SGD


if __name__ == "__main__":
    X, y = spiral_data(samples=100, classes=3)

    nn = NeuralNetwork([2, 64, 3])
    nn.train(X, y, epochs=1001, print_every=200)
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
