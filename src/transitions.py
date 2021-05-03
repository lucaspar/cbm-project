from tqdm import tqdm
from networkx.algorithms import isomorphism as iso
import graph_tool.all as gt
from graph_tool.topology import isomorphism, subgraph_isomorphism

from pqdm.processes import pqdm

from multiprocessing import Pool, RLock

from src.data import load_graph_pairs

def process_transitions(task):
    ego, G, H, transitions = task
    counts = [0 for _ in transitions]

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

    # count the transitions
    for idx, (size, pregraph, postgraph) in tqdm(enumerate(transitions), desc='ISOMORPHISMS', total=len(transitions), colour='red', leave=False):
        assert pregraph.num_vertices() == size and postgraph.num_vertices() == size

        G_iso = subgraph_isomorphism(pregraph, G_egonet, induced=True)
        H_iso = subgraph_isomorphism(postgraph, H_egonet, induced=True)

        if len(G_iso) > 0 and len(H_iso) > 0:
            for phi in G_iso:
                for psi in H_iso:
                    print(phi.get_array(), psi.get_array())
                    if phi.get_array() == psi.get_array():#problem here
                        print(f'success on transition {idx}')
                        counts[idx] += 1

    print(sum(counts))
    return counts

def process_ego(task):
    ego, graphs, transitions, cpus = task

    tasks = [(ego, graphs[t], graphs[t+1], transitions) for t in range(len(graphs) - 1)]
    counts = pqdm(tasks, process_transitions, n_jobs=cpus//2, desc='TRANSITIONS', colour='yellow', leave=False)
    return (ego, counts)

    #for t in tqdm(range(len(graphs) - 1), desc='ITERATING OVER TIMES', total=len(graphs)-1, colour='yellow', leave=False):
    #for t in range(len(graphs) - 1):
    #return

# TODO
# n: the number of vertices in the shared vertex set
#   graph_tool ensures that a graph with n vertices has vertex names {0, 1, ... n-1}
# graphs: a temporal sequence of graphs on a shared vertex set
# cpus: max number of CPU cores available for parallel processing
# empty: decides whether the empty (i.e., no edges) subgraph is considered
def egonet_transitions(n, graphs, cpus=4, empty=True):
    transitions = load_graph_pairs(sizes=[2, 3, 4, 5])
    counts = [0 for _ in transitions]

    tasks = [(ego, graphs, transitions, cpus//2) for ego in range(n)]

    result = pqdm(tasks, process_ego, n_jobs=cpus//2, desc='EGOS', colour='green')

    return

    tqdm.set_lock(RLock())

    pool = Pool(cpus//4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    pool.map(process_ego, tasks)
    return

    #for ego in tqdm(range(n), desc='ITERATING OVER NODES', total=n, colour='green'):
    for ego in range(n):

        pass
    return
