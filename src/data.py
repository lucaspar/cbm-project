from os.path import join
import datetime
import graph_tool.all as gt

# renames the vertices so they are integers in the range [0, ... n-1]
def reindex(edgelist):
    vertices = list({u for u, v, t in edgelist} | {v for u, v, t in edgelist})
    f = {v: i for (v, i) in zip(vertices, range(len(vertices)))}
    edgelist = [(f[u], f[v], t) for u, v, t in edgelist]
    return len(vertices), edgelist

# reindex the timestamps to be in the range [0, ... T-1]
def retime(edgelist):
    edgelist = sorted(edgelist, key=lambda x : x[2])
    times = [t for (u, v, t) in edgelist]
    f = {t: idx for (t, idx) in zip(times, range(len(times)))}
    return [(u, v, f[t]) for (u, v, t) in edgelist]

# returns a list of triplets (u, v, t)
#   where the edge (u, v) occurs at time t
def load(filename):
    edgelist = []
    with open(filename, 'r') as infile:
        extension = filename.split('.')[-1]
        if 'emails_cleaned_Dan' in filename and 'csv' in extension:
            count = 0
            name_dict = {}
            for line in infile:
                t, u, v = line.strip().split(',')
                year, month, day = map(int, t.split('-'))
                t = datetime.date(year, month, day).toordinal()
                if u not in name_dict.keys():
                    name_dict[u] = count
                    count += 1
                if v not in name_dict.keys():
                    name_dict[v] = count
                    count += 1
                edgelist.append((name_dict[u], name_dict[v], t))
                n = len(name_dict)
        elif 'edges' in extension:
            for line in infile:
                u, v, t = line.strip().split(',')
                edgelist.append((u, v, int(t)))
            n, edgelist = reindex(edgelist)
        else:
            raise IOError
    return n, edgelist

# returns a list of graph_tool graphs
def build_graphs(n, edgelist, resolution=10, equal_temperament=False):
    if resolution <= 1:
        G = gt.Graph(directed=False)
        G.add_edge_list([(u, v) for (u, v, t) in edgelist])
        return [G]

    if equal_temperament:
        edgelist = retime(edgelist)

    #nodes = {u for u, v, t in edgelist} | {v for u, v, t in edgelist}
    times = [t for u, v, t in edgelist]
    t_min, t_max = (min(times), max(times))

    delta = (t_max - t_min)/resolution

    timesteps = [t_min + delta*k for k in range(1, resolution)]

    first = [[(u, v) for u, v, t in edgelist if t <= timesteps[0]]]
    middle = [[(u, v) for u, v, t in edgelist if timesteps[idx] <= t < timesteps[idx+1]] \
            for idx in range(len(timesteps)-1)]
    last = [[(u, v) for u, v, t in edgelist if t > timesteps[-1]]]

    edge_sequence = first + middle + last
    graphs = [gt.Graph(directed=False) for edgelist in edge_sequence]

    for idx, G in enumerate(graphs):
        G.add_vertex(n=n)
        G.add_edge_list(edge_sequence[idx])

    return graphs

# returns a list of graph_tool graphs
#   if size <= 1, load all the graphs from the 'graphs.adj' file
#   if size >= 2, load only the graphs with that many vertices
def load_enumerated_graphs(path='enumerated_graphs', size=0):
    graphs = []
    with open(join(path, f'graph{size if size > 1 else "s"}.adj')) as infile:
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

# pairs up the enumerated graphs of the same size
def load_graph_pairs(sizes=[2, 3, 4, 5]):
    enums = {size: load_enumerated_graphs(size=size) for size in sizes}
    transitions = [(size, g, h) for size in enums for g in enums[size] for h in enums[size] if g.num_edges() > 0 or h.num_edges() > 0]
    return transitions
