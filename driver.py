from src.data import load, build_graphs

def main(verbose=False):
    filename = './data/eucore.edges'
    res = 5

    data = load(filename)
    graphs = build_graphs(data, resolution=res)

    if verbose:
        print('index\tnodes\tedges')
        for idx in range(res):
            print(f'{idx}\t{graphs[idx].order()}\t{graphs[idx].size()}')

if __name__ == '__main__':
    main(verbose=True)
