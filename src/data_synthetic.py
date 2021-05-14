import igraph as ig
from random import random, seed

# n:
#   number of nodes in the shared nodeset
#   this will include both genuine and anomalous nodes
# p:
#   probability of randomly connecting any two nodes
#   this does not apply to anomalous nodes
# anomalies:
#   percentage of the total nodes that should be anomalous
#   this should be a floating-point number between 0.0 and 1.0
# anomaly_type: 
#   determines the behavior characterizing the anomalous nodes
#   possibilities: ['clique', 'cycle', ...]
# num_graphs:
#   number of graphs in the temporal sequence
def generate_temporal_graph(n, p, anomalies=0.05, anomaly_type='clique', num_graphs=10):
    assert 0 <= p and p <= 1

    graphs = []
    nodes = list(range(n))
    regular = nodes[:n - int(n*anomalies)]
    anomalous = nodes[n - int(n*anomalies):]

    for t in range(num_graphs):
        edgelist = set()
        for idx, u in enumerate(regular):
            for v in regular[idx:]:
                if random() < p:
                    edgelist.add((u, v))
        graph = ig.Graph(n)
        graph.add_edges(edgelist)
        graphs.append(graph)

    if anomaly_type == 'clique':
        graphs = generate_temporal_cliques(graphs, anomalous)
    elif anomaly_type == 'clique':
        graphs = generate_temporal_cycles(graphs, anomalous)
    else:
        print(anomaly_type)
        raise NotImplementedError

    for graph in graphs:
        graph.simplify()

    return graphs

def generate_temporal_cliques(graphs, anomalous):
    flag = False
    for graph in graphs:
        if flag:
            edgelist = {(u, v) for u in anomalous for v in anomalous if u != v}
        else:
            edgelist = {}
        graph.add_edges(edgelist)
        flag = not flag
    return graphs

def generate_temporal_cycles():
    pass
    #g = 
    return
