from os.path import join
import graph_tool.all as gt

# renames the vertices so they are integers in the range [0, ... N-1]
def reindex(edgelist):
    vertices = list({u for u, v, t in edgelist} | {v for u, v, t in edgelist})
    f = {v: i for (v, i) in zip(vertices, range(len(vertices)))}
    edgelist = [(f[u], f[v], t) for u, v, t in edgelist]
    return len(vertices), edgelist

# returns a list of triplets (u, v, t)
#   where the edge (u, v) occurs at time t
def load(filename):
    edgelist = []
    with open(filename, 'r') as infile:
        extension = filename.split('.')[-1]
        for line in infile:
            u, v, t = line.strip().split(',')
            edgelist.append((u, v, int(t)))
    return reindex(edgelist)

# returns a list of graph_tool graphs
def build_graphs(N, edgelist, resolution=10):
    if resolution <= 1:
        G = gt.Graph(directed=False)
        G.add_edge_list([(u, v) for (u, v, t) in  edgelist])
        return [G]

    #nodes = {u for u, v, t in edgelist} | {v for u, v, t in edgelist}
    times = [t for u, v, t in edgelist]
    t_min, t_max = (min(times), max(times))

    delta = (t_max - t_min)/resolution

    timesteps = [delta*k for k in range(1, resolution)]

    first = [[(u, v) for u, v, t in edgelist if t <= timesteps[0]]]
    middle = [[(u, v) for u, v, t in edgelist if timesteps[idx] <= t < timesteps[idx+1]] \
              for idx in range(len(timesteps)-1)]
    last = [[(u, v) for u, v, t in edgelist if t > timesteps[-1]]]

    edge_sequence = first + middle + last
    graphs = [gt.Graph(directed=False) for edgelist in edge_sequence]

    for idx, G in enumerate(graphs):
        G.add_vertex(n=N)
        G.add_edge_list(edge_sequence[idx])

    return graphs

# returns a list of graph_tool graphs
def load_enumerated_graphs(path='enumerated_graphs', filename='graphs.adj'):
    graphs = []
    with open(join(path, 'graphs.adj')) as infile:
        for line in infile:
            line = line.strip().split(' ')
            if line[0] == 'Graph':
                n = int(line[-1].strip('.'))
                G = gt.Graph(directed=False)
                G.add_vertex(n)
                for _ in range(n):
                    line = next(infile).strip().strip(';').split(' ')
                    u = int(line[0])
                    for v in line[2:]:
                        if v != '':
                            v = int(v)
                            G.add_edge(u, v)
                graphs.append(G)
    return graphs
