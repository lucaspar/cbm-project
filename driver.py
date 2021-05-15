from igraph import summary

from src.data_read import load, build_graphs
from src.data_synthetic import generate_temporal_graph
from src.transitions import egonet_transitions

def synthetic_data(n=500, p=0.25, anomalies=0.05, anomaly_type='clique', num_graphs=10):
    graphs = generate_temporal_graph(n, p, anomalies=anomalies, anomaly_type=anomaly_type, num_graphs=num_graphs)
    return graphs, f'synthetic_{anomaly_type}'

def real_data(name='enron_truncated_smallestest', extension='csv', resolution=5):
    filename = f'./data/{name}.{extension}'

    n, data = load(filename)
    graphs = build_graphs(n, data, resolution=resolution, equal_temperament=True)

    return graphs, name

def main(filename='', datatype='', verbose=False):
    # parameters for synthetic data
    n = 500
    p = 0.005
    anomalies = 0.05
    anomaly_type = 'clique'
    num_graphs = 5

    # parameters for real data
    name, extension = filename.split('.') if filename != '' else ('', '')
    resolution = 5

    graphs, dataset = synthetic_data(n=n, p=p, anomalies=anomalies, anomaly_type=anomaly_type, num_graphs=num_graphs) if datatype == 'synthetic' \
                      else real_data(name=name, extension=extension, resolution=resolution)

    ###
    #temp = []
    #for graph in graphs:
    #    temp.append(graph.to_graph_tool())
    #graphs = temp
    ###

    if verbose:
        print('index\tnodes\tedges')
        for idx, g in enumerate(graphs):
            print(idx, end='', flush=True)
            summary(g)

    embeddings = egonet_transitions(n, graphs)

    with open(f'./embeddings/{dataset}_embeddings.txt', 'w') as outfile:
        for (ego, embedding) in embeddings:
            outfile.write(f'{ego}\t{embedding}\n')

if __name__ == '__main__':
    main(filename='', \
         datatype='synthetic', \
         verbose=True)
