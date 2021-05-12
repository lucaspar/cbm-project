import numpy as np
import graph_tool.all as gt
from graph_tool.topology import isomorphism, subgraph_isomorphism

from tqdm import tqdm
from pqdm.processes import pqdm

from itertools import chain, combinations

from src.data_read import load_graph_pairs

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def process_iso(task):
    idx, g, h, pregraph, postgraph = task
    if isomorphism(g, pregraph) and isomorphism(h, postgraph):
        return idx, 1
    else:
        return idx, 0

def process_transitions(task):
    ego, G, H, enumerated, cpus = task
    counts = [0 for _ in enumerated]

    G_vp = G.new_vertex_property('bool')
    H_vp = H.new_vertex_property('bool')

    G_nbhd = {w for w in G.vertex(ego).out_neighbors()} | {ego}
    H_nbhd = {w for w in H.vertex(ego).out_neighbors()} | {ego}

    # construct a sharerd vertex set for the egonets
    for v in G.vertices():
        if v in G_nbhd or v in H_nbhd:
            G_vp[v] = True
        else:
            G_vp[v] = False
    for v in H.vertices():
        if v in H_nbhd or v in G_nbhd:
            H_vp[v] = True
        else:
            H_vp[v] = False

    G_egonet = gt.GraphView(G, vfilt=G_vp, directed=False)
    H_egonet = gt.GraphView(H, vfilt=H_vp, directed=False)

    G_ego_vert = powerset(G_egonet.vertices())
    H_ego_vert = powerset(H_egonet.vertices())

    G_subgraphs = []
    H_subgraphs = []

    for vertices in powerset(G_egonet.vertices()):
        if len(vertices) != 0:
            vp = G_egonet.new_vertex_property('bool')
            G_subgraphs.append(gt.GraphView(G_egonet, vfilt=vp, directed=False))
    for vertices in powerset(H_egonet.vertices()):
        if len(vertices) != 0:
            vp = H_egonet.new_vertex_property('bool')
            H_subgraphs.append(gt.GraphView(H_egonet, vfilt=vp, directed=False))

    transitions = [(g, h) for g in G_subgraphs for h in H_subgraphs \
                   if g.num_vertices() == h.num_vertices() \
                   and np.array_equal(g.get_vertices(), h.get_vertices())]

    #tasks = [(idx, g, h, pregraph, postgraph) for (g, h) in transitions for (idx, (size, pregraph, postgraph)) in enumerate(enumerated)]
    #results = pqdm(tasks, process_iso, n_jobs=cpus, desc='ISOMORPHISMS', colour='red', leave=False)

    #for idx, count in results:
    #    counts[idx] += count

    for g, h in tqdm(transitions, desc='ISOMORPHISMS', total=len(transitions), colour='red', leave=False):
        for idx, (size, pregraph, postgraph) in enumerate(enumerated):
            if isomorphism(g, pregraph) and isomorphism(h, postgraph):
                counts[idx] += 1

    return counts

    # count the transitions
    for idx, (size, pregraph, postgraph) in tqdm(enumerate(enumerated), desc='ISOMORPHISMS', total=len(enumerated), colour='red', leave=False):
        assert pregraph.num_vertices() == size and postgraph.num_vertices() == size

        G_iso = subgraph_isomorphism(pregraph, G_egonet, induced=True)
        H_iso = subgraph_isomorphism(postgraph, H_egonet, induced=True)

        if len(G_iso) > 0 and len(H_iso) > 0:
            for phi in G_iso:
                for psi in H_iso:
                    if np.array_equal(phi.get_array(), psi.get_array()):
                        counts[idx] += 1

    return counts

        #for z in G_egonet.vertices():
        #    print(z)
        #print(G_iso[0].get_array())
        #for z in G_iso[0]:
        #    print(z in G_egonet.vertices())
        #print(g_iso, h_iso)

def process_ego(task):
    ego, graphs, enumerated, cpus = task

    tasks = [(ego, graphs[t], graphs[t+1], enumerated, cpus) for t in range(len(graphs) - 1)]
    counts = pqdm(tasks, process_transitions, n_jobs=cpus, desc='TRANSITIONS', colour='yellow', leave=False)
    counts = np.asarray(counts)
    return (ego, np.asarray(counts).mean(axis=0).tolist())

    #for t in tqdm(range(len(graphs) - 1), desc='ITERATING OVER TIMES', total=len(graphs)-1, colour='yellow', leave=False):
    #for t in range(len(graphs) - 1):
    #return

# TODO
# n: the number of vertices in the shared vertex set
#   graph_tool ensures that a graph with n vertices has vertex names {0, 1, ... n-1}
# graphs: a temporal sequence of graphs on a shared vertex set
# cpus: max number of CPU cores available for parallel processing
# empty: decides whether the empty (i.e., no edges) subgraph is considered
def egonet_transitions(n, graphs, cpus=8, empty=True):
    enumerated = load_graph_pairs(sizes=list([2, 3]))
    counts = [0 for _ in enumerated]

    tasks = [(ego, graphs, enumerated, cpus) for ego in range(n)]

    embeddings = pqdm(tasks, process_ego, n_jobs=cpus//2, desc='EGOS', colour='green')

    return embeddings

    #tqdm.set_lock(RLock())

    #pool = Pool(cpus//4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    #pool.map(process_ego, tasks)
    #return

    ##for ego in tqdm(range(n), desc='ITERATING OVER NODES', total=n, colour='green'):
    #for ego in range(n):

    #    pass
    #return
