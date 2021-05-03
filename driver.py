from src.data import load, build_graphs
from src.transitions import egonet_transitions

def main(verbose=False):
    dataset = './data/eucore.edges'
    resolution = 5

    n, data = load(dataset)
    graphs = build_graphs(n, data, resolution=resolution)

    if verbose:
        print('index\tnodes\tedges')
        for idx, g in enumerate(graphs):
            print(idx, g)

    embeddings = egonet_transitions(n, graphs, cpus=32)

    with open('embeddings.txt', 'w') as outfile:
        for (ego, embedding) in embeddings:
            outfile.write(f'{ego}\t{embedding}\n')

if __name__ == '__main__':
    main()
