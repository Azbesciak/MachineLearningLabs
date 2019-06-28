## Gist originally developed by @craffel and improved by @ljhuang2017
## ...and further improved/extended by MK
import copy
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import math
import matplotlib.pyplot as plt
import numpy as np

import os


def findLastIndex(str, x):
    index = -1
    for i in range(0, len(str)):
        if str[i] == x:
            index = i
    return index


def write_pdf(fig, name):
    backslash_index = findLastIndex(name, '/')
    if backslash_index > 0:
        dir = name[:backslash_index]
        if not os.path.exists(dir):
            os.makedirs(dir)
    fig.savefig(name + ".pdf")


def draw_neural_net(ax, left, right, bottom, top, layer_sizes, actfun_hid, actfun_out, coefs_, intercepts_, n_iter_,
                    loss_):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Input-Arrows
    layer_top_0 = v_spacing * (layer_sizes[0] - 1) / 2. + (top + bottom) / 2.
    for m in range(layer_sizes[0]):
        plt.arrow(left - 0.18, layer_top_0 - m * v_spacing, 0.12, 0, lw=1, head_width=0.01, head_length=0.02)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 8.,
                                color='w', ec='green' if n == 0 else 'red' if n == len(layer_sizes) - 1 else 'k',
                                zorder=4)
            if n == 0:
                plt.text(left - 0.125, layer_top - m * v_spacing, r'$X_{' + str(m + 1) + '}$', fontsize=15)
            elif (n_layers == 3) & (n == 1):
                plt.text(n * h_spacing + left + 0.00, layer_top - m * v_spacing + (v_spacing / 8. + 0.01 * v_spacing),
                         r'$H_{' + str(m + 1) + '}$', fontsize=15)
            elif n == n_layers - 1:
                plt.text(n * h_spacing + left + 0.10, layer_top - m * v_spacing, r'$y_{' + str(m + 1) + '}$',
                         fontsize=15)
            ax.add_artist(circle)
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers - 1:
            x_bias = (n + 0.5) * h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing / 8., color='w', ec='k', zorder=4)
            plt.text(x_bias - (v_spacing / 8. + 0.10 * v_spacing + 0.01), y_bias, r'$1$', fontsize=15)
            ax.add_artist(circle)
            # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
                xm = (n * h_spacing + left)
                xo = ((n + 1) * h_spacing + left)
                ym = (layer_top_a - m * v_spacing)
                yo = (layer_top_b - o * v_spacing)
                rot_mo_rad = np.arctan((yo - ym) / (xo - xm))
                rot_mo_deg = rot_mo_rad * 180. / np.pi
                xm1 = xm + (v_spacing / 8. + 0.05) * np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing / 8. + 0.12) * np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing / 8. + 0.05) * np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing / 8. + 0.12) * np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing / 8. + 0.04) * np.sin(rot_mo_rad)
                plt.text(xm1, ym1, \
                         str(round(coefs_[n][m, o], 4)), \
                         rotation=rot_mo_deg, \
                         fontsize=10)
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers - 1:
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        x_bias = (n + 0.5) * h_spacing + left
        y_bias = top + 0.005
        for o in range(layer_size_b):
            line = plt.Line2D([x_bias, (n + 1) * h_spacing + left],
                              [y_bias, layer_top_b - o * v_spacing], c='k')
            ax.add_artist(line)
            xo = ((n + 1) * h_spacing + left)
            yo = (layer_top_b - o * v_spacing)
            rot_bo_rad = np.arctan((yo - y_bias) / (xo - x_bias))
            rot_bo_deg = rot_bo_rad * 180. / np.pi
            xo2 = xo - (v_spacing / 8. + 0.01) * np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing / 8. + 0.01) * np.sin(rot_bo_rad)
            xo1 = xo2 - 0.05 * np.cos(rot_bo_rad)
            yo1 = yo2 - 0.05 * np.sin(rot_bo_rad)
            plt.text(xo1, yo1, \
                     str(round(intercepts_[n][o], 4)), \
                     rotation=rot_bo_deg, \
                     fontsize=10)

            # Output-Arrows
    layer_top_0 = v_spacing * (layer_sizes[-1] - 1) / 2. + (top + bottom) / 2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right + 0.015, layer_top_0 - m * v_spacing, 0.16 * h_spacing, 0, lw=1, head_width=0.01,
                  head_length=0.02)

    plt.text(0.5, bottom - 0.005 * v_spacing,
             'Hidden,Output: ' + str(actfun_hid) + ',' + str(actfun_out) + '\nSteps:' + str(n_iter_) + '\nLoss:' + str(
                 loss_),
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12,
             color='blue')

    plt.show()
    # would be nice to write ranges of values for input nodes (min,max), but we don't get this information in this function...
    # also would be nice to move Hidden,Output info about act functions and write it slightly below bottom-most nodes in each layer


# MK, added this extremely convenient function with only one parameter (the network)
def draw_nn(mlp, name):  # gets all params from 'mlp' and passes to draw_neural_net()
    # layer_sizes = [2] + list(my_hidden_layer_sizes) + [1]  #<-- MK: seems the author was wrong by using fixed [2] here, while he constructed 3 inputs? and perhaps only used 2, unknowingly?
    print("layer weight matrix shapes:", [coef.shape for coef in mlp.coefs_])
    layer_sizes = [coef.shape[0] for coef in mlp.coefs_] + [
        mlp.coefs_[-1].shape[1]]  # for last, use in-number and also append out-number
    print("detected layer sizes:", layer_sizes)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9, layer_sizes, mlp.activation, mlp.out_activation_, mlp.coefs_, mlp.intercepts_,
                    mlp.n_iter_, mlp.loss_)
    write_pdf(fig, name + "_schema")


def draw_response(mlp, X, y, name):
    h = .02  # step size in the mesh
    X = np.array(X)  # only needed if someone provided a list of lists as X
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = mlp.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    # Put the prediction into a color plot
    ax.contourf(xx, yy, Z, cmap='coolwarm', alpha=.4)
    # show dataset on top of contours
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=80)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()
    write_pdf(fig, name + "_resp_2d")


def draw_response3D(mlp, X, y, name):
    h = .02  # step size in the mesh
    X = np.array(X)  # only needed if someone provided a list of lists as X
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = mlp.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Wejście 1')
    ax.set_ylabel('Wejście 2')
    ax.set_zlabel('Wyjśce')

    surf = ax.plot_surface(xx, yy, Z, cmap='coolwarm', linewidth=0, antialiased=True)
    ax.scatter(X[:, 0], X[:, 1], y, c=y, s=200, cmap='coolwarm')
    # TODO overlay Axes3D.scatter X and color by y, as in draw_response()? https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    write_pdf(fig, name + "_resp_3d")


def draw_square_error(licz, accu, loss, square_error, name):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(licz, accu, label='accuracy')
    plt.plot(licz, loss, label='loss')
    plt.plot(licz, square_error, label='square error')
    ax = fig.gca()
    ax.set_xlabel('Numer iteracji')
    ax.set_ylabel('Wartość')
    plt.gca().legend(('accuracy', 'loss', 'square error'))
    write_pdf(fig, name + "_se")


def draw_train_and_test_error(iter, se_train, se_test, name):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(iter, se_train, label='Train')
    plt.plot(iter, se_test, label='Test')
    ax = fig.gca()
    ax.set_xlabel('Numer iteracji')
    ax.set_ylabel('Trafność')
    plt.gca().legend(('Train', 'Test'))
    write_pdf(fig, name + "_se")


def draw_n_accuracy(iter, data, name, x_label='Numer iteracji', y_label='Trafność'):
    fig = plt.figure(figsize=(8, 8))
    keys = []
    for [k, value] in data.items():
        keys.append(k)
        plt.plot(iter, value, label=k)
    ax = fig.gca()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if len(keys) > 1:
        plt.gca().legend(keys)
    write_pdf(fig, name + "_accu")


def draw_weights_histogram(licz, weights, name):
    fig = plt.figure(figsize=(8, 8))
    # plt.rcParams["figure.figsize"] = (unit , unit* 1.5)
    index = np.arange(len(licz))
    bar_width = 0.5
    opacity = 0.8
    w = defaultdict(lambda: [])
    for d, data in enumerate(weights):
        for lay_i, lay in enumerate(data):
            for source_i, source in enumerate(lay):
                for dest_i, dest in enumerate(source):
                    w[f"{lay_i}:{source_i}->{dest_i}"].append(dest)
    colors = ['#F44336', '#CDDC39', '#FFC107', '#03A9F4', '#9C27B0', '#8BC34A']
    charts = len(w.items())
    rows = math.ceil(math.sqrt(charts))
    columns = math.ceil(charts / rows)
    for i, [key, value] in enumerate(w.items()):
        plt.subplot(rows, columns, i + 1)
        plt.bar(index + 0.5, value, bar_width,
                alpha=opacity,
                color=colors[i % len(colors)],
                label=key)
        ax = fig.gca()
        # ax.set_xlabel('Numer iteracji')
        # ax.set_ylabel('Wartość')
        plt.xticks(index + bar_width, licz)
        plt.legend()

    plt.show()
    write_pdf(fig, name + "_histo")


def draw_hist(x_vals, data, name):
    fig = plt.figure(figsize=(5, 5))
    index = np.arange(len(list(data.items())[0][1]))
    bar_width = 0.3
    opacity = 0.8
    for i, [key, value] in enumerate(data.items()):
        value.reverse()
        plt.bar(index + bar_width * (i), value, bar_width,
                alpha=opacity,
                label=key)
    plt.xlabel('Wartość K1')
    plt.ylabel("Procent przypadków")
    plt.xticks(index + bar_width, x_vals)
    plt.legend()
    plt.show()
    write_pdf(fig, name + "_hist_k1")


def normalize_dataset(dataset):
    data = np.array(dataset)
    for i in range(data.shape[1]):
        elems = data[:, i]
        min_val = elems.min()
        max_val = elems.max()
        data[:, i] = (elems - min_val) / (max_val - min_val)
    return data


def drop_each(data, each_num):
    return [d for i, d in enumerate(data) if i % each_num != 0]


if __name__ == '__main__':
    results = {'good': [70, 68, 65, 63, 62], 'bad': [7, 9, 12, 14, 15]}
    # results = {'good': [0, 20, 29, 43, 49, 58], 'bad': [0, 2, 6, 9, 14, 19], 'unknown': [77, 55, 42, 25, 14, 0]}
    normalized = {}
    totals = [0, 0, 0, 0, 0, 0]
    for k, values in results.items():
        for i, v in enumerate(values):
            totals[i] += v
    for k, values in results.items():
        values = [v/t * 100 for v,t in zip(values, totals)]
        normalized[k] = values
    k1 = [0.25,0.3,0.35,0.4,0.45]
    # k1 = [1, 0.9, 0.8, 0.7, 0.6, 0.50001]
    # k1.reverse()

    draw_hist(k1, normalized, "pima/7/unknown_own")
