from NN.NN_helpers import write_pdf

res = """({'solver': 'adam', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 4], 'alpha': 1e-05}, 0.7428571428571429)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [16, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.7350649350649351)
({'solver': 'adam', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 1e-05}, 0.7298701298701299)
({'solver': 'adam', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8], 'alpha': 1e-05}, 0.7272727272727273)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8], 'alpha': 0.0001}, 0.7272727272727273)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.7272727272727273)
({'solver': 'adam', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 4], 'alpha': 1e-05}, 0.7246753246753247)
({'solver': 'adam', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 1e-05}, 0.7246753246753247)
({'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 4], 'alpha': 0.0001}, 0.7220779220779221)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8], 'alpha': 1e-05}, 0.7220779220779221)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8], 'alpha': 1e-05, 'learning_rate': 'adaptive'}, 0.7220779220779221)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 1e-05}, 0.7168831168831169)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 1e-05, 'learning_rate': 'adaptive'}, 0.7168831168831169)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 4], 'alpha': 1e-05}, 0.7064935064935065)
({'solver': 'adam', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 8], 'alpha': 1e-05}, 0.7064935064935065)
({'solver': 'adam', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 1e-05}, 0.7038961038961039)
({'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8], 'alpha': 0.0001}, 0.7038961038961039)
({'solver': 'adam', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 1e-05}, 0.7038961038961039)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 4], 'alpha': 0.0001}, 0.7012987012987014)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 4], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.7012987012987014)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 1e-05, 'learning_rate': 'adaptive'}, 0.6987012987012988)
({'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 4], 'alpha': 0.0001}, 0.6961038961038961)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 0.0001}, 0.6961038961038961)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 4], 'alpha': 1e-05}, 0.6961038961038961)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 4], 'alpha': 1e-05, 'learning_rate': 'adaptive'}, 0.6961038961038961)
({'solver': 'sgd', 'learning_rate_init': 0.0001, 'activation': 'relu', 'hidden_layer_sizes': [32, 16, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6935064935064936)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 1e-05}, 0.6935064935064935)
({'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 0.0001}, 0.6935064935064935)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6935064935064934)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [8, 8], 'alpha': 1e-05}, 0.690909090909091)
({'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 0.0001}, 0.6909090909090908)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [16, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6883116883116884)
({'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 0.0001}, 0.6857142857142857)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'relu', 'hidden_layer_sizes': [32, 16, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6805194805194805)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 0.0001}, 0.6727272727272727)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6727272727272727)
({'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 0.0001}, 0.6727272727272726)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'relu', 'hidden_layer_sizes': [32, 16, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6727272727272726)
({'solver': 'sgd', 'learning_rate_init': 0.0001, 'activation': 'relu', 'hidden_layer_sizes': [16, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6701298701298701)
({'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8], 'alpha': 0.0001}, 0.6649350649350649)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 0.0001}, 0.6623376623376623)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6623376623376623)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 0.0001}, 0.6571428571428571)
({'solver': 'sgd', 'learning_rate_init': 0.01, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6571428571428571)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 4], 'alpha': 0.0001}, 0.6493506493506493)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 1e-05}, 0.6493506493506493)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 4], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6493506493506493)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 4], 'alpha': 1e-05, 'learning_rate': 'adaptive'}, 0.6493506493506493)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8], 'alpha': 0.0001}, 0.6493506493506493)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 1e-05}, 0.6493506493506493)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8], 'alpha': 0.0001, 'learning_rate': 'adaptive'}, 0.6493506493506493)
({'solver': 'sgd', 'learning_rate_init': 0.001, 'activation': 'logistic', 'hidden_layer_sizes': [8, 8, 8], 'alpha': 1e-05, 'learning_rate': 'adaptive'}, 0.6493506493506493)
"""
from ast import literal_eval as make_tuple
from collections import defaultdict

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import os

def format_label(di):
    label = f"{di['solver']}_{di['activation']}_{di['hidden_layer_sizes']}_{di['learning_rate_init']}_{di['alpha']}"
    if 'learning_rate' in di:
        label += f"_{di['learning_rate']}"
    return label

def draw_net_histogram(nets, name):
    fig = plt.figure(figsize=(8, 8))
    # plt.rcParams["figure.figsize"] = (unit , unit* 1.5)
    bar_width = 0.6
    opacity = 0.8
    colors = ['#F44336', '#CDDC39', '#FFC107', '#03A9F4', '#9C27B0', '#8BC34A']
    for i, [key, value] in enumerate(nets):
        plt.bar(i, value, bar_width,
                alpha=opacity,
                # color=colors[i % len(colors)],
                label=format_label(key))
    ax = fig.gca()
    print(len(nets))
    ax.set_ylim(0.625, 0.75)
    ax.set_xlim(-1, 52)
    plt.xticks([])
    plt.legend(ncol=2, loc='upper left', bbox_to_anchor=(-0.1,0))

    plt.show()
    write_pdf(fig, name + "_histo")


lines = [make_tuple(l) for l in res.split("\n") if len(l) > 0]
lines.sort(key=lambda x: x[1], reverse=True)
draw_net_histogram(lines, "pima_perf")

