import numpy as np


def remove_lowest_weights(weights):
    x, y, z = find_min_absolute_position(weights)
    clear_weight(x, y, z, weights)


def find_min_absolute_position(arr):
    cur_min = np.inf
    cur_i = (-1, -1, -1)
    for x, neuron in enumerate(arr):
        for y, connections in enumerate(neuron):
            for z, weight in enumerate(connections):
                abs_val = abs(weight)
                if cur_min > abs_val > 0:
                    cur_i = (x, y, z)
                    cur_min = abs_val
    return cur_i


def clear_weight(x, y, z, weights):
    if x >= 0 and y >= 0 and z >= 0:
        weights[x][y][z] = 0


def load_pima():
    import pandas as pd
    DATASET_PATH = 'diabetes.csv'
    dataset = pd.read_csv(DATASET_PATH)

    return dataset.values[:, :-1], dataset.values[:, -1]
