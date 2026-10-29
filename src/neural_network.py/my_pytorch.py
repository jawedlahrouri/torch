import sys
import numpy as np
import matplotlib


inputs = [1.2, 5.1, 2.1] #valeur d'entrée de nos neurones les chiffres son superflux plus tard ces poids seront mit à jour automatiquement
weights = [3.1, 2.1, 8.7] #poids de nos neurones les chiffres son superflux
bias = 3 #biais = nombre de neurones

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)