from src.data import load, build_graphs
from src.transitions import egonet_transitions

def main(verbose=False):
    dataset = './data/eucore.edges'
    resolution = 5

    N, data = load(dataset)
    graphs = build_graphs(N, data, resolution=resolution)

    result = egonet_transitions(N, graphs)
    exit()

    pass

    if verbose:
        print('index\tnodes\tedges')
        for idx, g in enumerate(graphs):
            print(idx, g)

if __name__ == '__main__':
    main(verbose=True)
