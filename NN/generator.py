def parse_args():
    from argparse import ArgumentParser
    from argparse import ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description="Framestick feed-forward f0 generator",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--schema", type=str, help="architecture", required=True)
    args = parser.parse_args()

    return args


def generate_connections(start_index: int, l1_neurons: int, l2_neurons: int):
    connections = []
    l1_end_neuron = start_index + l1_neurons
    for l1 in range(start_index, l1_end_neuron):
        for l2 in range(l1_end_neuron, l1_end_neuron + l2_neurons):
            connections.append((l1, l2))
    return connections


def generate_f0(neurons, connections):
    out = "//0\np:\n"
    out += "n:d=Nu\n" * neurons
    for c in connections:
        out += f"c:{c[0]}, {c[1]}, 1\n"
    return out


if __name__ == '__main__':
    args = parse_args()
    layers = [int(v) for v in args.schema.split('-')]
    layers.reverse()
    neurons = sum(layers)
    layer_start_index = 0
    connections = []
    for i in range(len(layers) - 1):
        l1_neurons = layers[i]
        connections.extend(generate_connections(layer_start_index, l1_neurons, layers[i + 1]))
        layer_start_index += l1_neurons
    net = generate_f0(neurons, connections)
    print(net)
