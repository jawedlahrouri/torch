import sys
import numpy as np
import matplotlib

number_case = 64
number_pawn=32

inputs = [ [1, 2, 3, 2.5],
           [2, 5, -1, 2],
           [-1.5, 2.7, 3.3, -0.8]] #valeur d'entrée de nos neurones les chiffres son superflux plus tard ces poids seront mit à jour automatiquement

weights = [[0.2, 0.8, -0.5, 1], #poids de nos neurones les chiffres son superflux
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5], #poids de nos neurones les chiffres son superflux
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)