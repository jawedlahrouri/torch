import numpy as np

softmax_outputs = np.array([[0.0, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

print(np.mean(-np.log(softmax_outputs[[0, 1, 2], [class_targets]])))
