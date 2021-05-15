import numpy as np
import igraph as ig
import graph_tool.all as gt
from graph_tool.topology import isomorphism, subgraph_isomorphism

from tqdm import tqdm
from pqdm.processes import pqdm
from multiprocessing import Pool

from itertools import chain, combinations

from src.data_read import load_graph_pairs

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def process_iso(task):
    idx, g, h, pregraph, postgraph = task
    #print(g.vertices(), h.vertices(), pregraph.vertices(), postgraph.vertices())
    if isomorphism(g, pregraph) and isomorphism(h, postgraph):
        return idx, 1
    else:
        return idx, 0

def process_transitions(task):
    ego, G, H, enumerated = task

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

    tasks = [(idx, g, h, pregraph, postgraph) for (g, h) in transitions for (idx, (size, pregraph, postgraph)) in enumerate(enumerated)]
    #results = pqdm(tasks, process_iso, n_jobs=cpus, desc='ISOMORPHISMS', colour='red', leave=False)

    results = []
    #with Pool(2) as pool:
    #for result in tqdm(map(process_iso, tasks), total=len(tasks), colour='red', leave=False):
    #    results.append(result)

    #for idx, count in results:
    #    counts[idx] += count

    counts = [0 for _ in enumerated]
    for g, h in tqdm(transitions, desc='ISOMORPHISMS', total=len(transitions), colour='red', leave=False):
        for idx, (size, pregraph, postgraph) in enumerate(enumerated):
            if isomorphism(g, pregraph) and isomorphism(h, postgraph):
                counts[idx] += 1

    return counts

    # count the transitions
    #for idx, (size, pregraph, postgraph) in tqdm(enumerate(enumerated), desc='ISOMORPHISMS', total=len(enumerated), colour='red', leave=False):
    #    assert pregraph.num_vertices() == size and postgraph.num_vertices() == size

    #    G_iso = subgraph_isomorphism(pregraph, G_egonet, induced=True)
    #    H_iso = subgraph_isomorphism(postgraph, H_egonet, induced=True)

    #    if len(G_iso) > 0 and len(H_iso) > 0:
    #        for phi in G_iso:
    #            for psi in H_iso:
    #                if np.array_equal(phi.get_array(), psi.get_array()):
    #                    counts[idx] += 1

    #return counts

        #for z in G_egonet.vertices():
        #    print(z)
        #print(G_iso[0].get_array())
        #for z in G_iso[0]:
        #    print(z in G_egonet.vertices())
        #print(g_iso, h_iso)

def process_ego(task):
    ego, graphs, pairs = task

    #tasks = [(ego, graphs[t], graphs[t+1], pairs) for t in range(len(graphs) - 1)]

    counts = []
    for G, H in [(graphs[t], graphs[t+1]) for t in range(len(graphs) - 1)]:
        G_nbhd = set(G.vs[ego].neighbors()) | {G.vs[ego]}
        H_nbhd = set(H.vs[ego].neighbors()) | {H.vs[ego]}

        # construct a sharerd vertex set for the egonets
        shared_nbhd = G_nbhd | H_nbhd

        G_egonet = G.subgraph(shared_nbhd)
        H_egonet = H.subgraph(shared_nbhd)

        enumerated_verts = powerset(shared_nbhd)

        G_subgraphs = []
        H_subgraphs = []

        #for vertices in powerset(G_egonet.vertices()):
        #    if len(vertices) != 0:
        #        vp = G_egonet.new_vertex_property('bool')
        #        G_subgraphs.append(gt.GraphView(G_egonet, vfilt=vp, directed=False))
        #for vertices in powerset(H_egonet.vertices()):
        #    if len(vertices) != 0:
        #        vp = H_egonet.new_vertex_property('bool')
        #        H_subgraphs.append(gt.GraphView(H_egonet, vfilt=vp, directed=False))

        #transitions = [(g, h) for g in G_subgraphs for h in H_subgraphs \
        #            if g.num_vertices() == h.num_vertices() \
        #            and np.array_equal(g.get_vertices(), h.get_vertices())]

        counts_t = [0 for _ in pairs]
        for idx, (size, pregraph, postgraph) in tqdm(enumerate(pairs), desc='ISOMORPHISMS', total=len(pairs), colour='red', leave=False, disable=False):
            for phi in G_egonet.get_subisomorphisms_vf2(pregraph):
                for psi in H_egonet.get_subisomorphisms_vf2(postgraph):
                    if phi == psi:
                        counts_t[idx] += 1

        counts.append(counts_t)

    return (ego, np.asarray(counts).mean(axis=0).tolist())

    counts = []
    #with Pool(2) as pool:
    for count in tqdm(map(process_transitions, tasks), total=len(tasks), colour='yellow', leave=False):
        counts.append(count)
    #counts = pqdm(tasks, process_transitions, n_jobs=cpus, desc='TRANSITIONS', colour='yellow', leave=False)
    counts = np.asarray(counts)
    #print(ego, counts)
    return (ego, np.asarray(counts).mean(axis=0).tolist())

    #for t in tqdm(range(len(graphs) - 1), desc='ITERATING OVER TIMES', total=len(graphs)-1, colour='yellow', leave=False):
    #for t in range(len(graphs) - 1):
    #return

# TODO
# n: the number of vertices in the shared vertex set
#   graph_tool ensures that a graph with n vertices has vertex names {0, 1, ... n-1}
# graphs: a temporal sequence of graphs on a shared vertex set
def egonet_transitions(n, graphs):
    pairs = load_graph_pairs(sizes=[2, 3])
    pairs = [(size, ig.Graph.from_graph_tool(g), ig.Graph.from_graph_tool(h)) for (size, g, h) in pairs]

    tasks = [(ego, graphs, pairs) for ego in range(n)]

    embeddings = []
    with Pool(8) as pool:
        for embedding in tqdm(pool.imap(process_ego, tasks), total=len(tasks), colour='green', disable=False):
            embeddings.append(embedding)
    #with Pool(2) as pool:
    #    for embedding in tqdm(pool.imap(process_ego, tasks), total=len(tasks), colour='green', leave=False):
    #        embeddings.append(embedding)
    #embeddings = pqdm(tasks, process_ego, n_jobs=cpus//2, desc='EGOS', colour='green')

    return embeddings

    #tqdm.set_lock(RLock())

    #pool = Pool(cpus//4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    #pool.map(process_ego, tasks)
    #return

    ##for ego in tqdm(range(n), desc='ITERATING OVER NODES', total=n, colour='green'):
    #for ego in range(n):

    #    pass
    #return
