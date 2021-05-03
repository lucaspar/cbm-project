import networkx as nx

def load(filename):
    edges = []
    with open(filename, 'r') as file:
        extension = filename.split('.')[-1]
        for line in file:
            u, v, t = line.strip().split(',')
            edges.append((u, v, int(t)))
    return edges

def build_graphs(edge_stream, resolution=10):
    if resolution <= 1:
        G = nx.Graph()
        G.add_edges_from(edge_stream, time=0)
        return [G]

    nodes = {u for u, v, t in edge_stream} | {v for u, v, t in edge_stream}
    times = [t for u, v, t in edge_stream]
    t_min, t_max = (min(times), max(times))

    delta = (t_max - t_min)/resolution

    timesteps = [delta*k for k in range(1, resolution)]

    first = [[(u, v) for u, v, t in edge_stream if t <= timesteps[0]]]
    middle = [[(u, v) for u, v, t in edge_stream if timesteps[idx] <= t < timesteps[idx+1]] \
              for idx in range(len(timesteps)-1)]
    last = [[(u, v) for u, v, t in edge_stream if t > timesteps[-1]]]

    edge_sequence = first + middle + last
    graphs = [nx.Graph() for edge_stream in edge_sequence]

    for idx, G in enumerate(graphs):
        G.add_nodes_from(nodes)
        G.add_edges_from(edge_sequence[idx], time=idx)

    return graphs
