from src.data import load, build_graphs
from src.transitions import egonet_transitions

def main(verbose=False):
    #dataset = 'eucore.edges'
    #extension = 'edges'
    #resolution = 5

    #dataset = 'enron'
    #extension = 'csv'
    #resolution = 5

    dataset = 'enron_truncated'
    extension = 'csv'
    resolution = 5

    filename = f'./data/{dataset}.{extension}'

    n, data = load(filename)
    graphs = build_graphs(n, data, resolution=resolution, equal_temperament=True)

    if verbose:
        print('index\tnodes\tedges')
        for idx, g in enumerate(graphs):
            print(idx, g)

    embeddings = egonet_transitions(n, graphs, cpus=32)

    with open(f'./embeddings/{dataset}_embeddings.txt', 'w') as outfile:
        for (ego, embedding) in embeddings:
            outfile.write(f'{ego}\t{embedding}\n')

if __name__ == '__main__':
    main(verbose=True)
