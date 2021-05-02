from tqdm import tqdm
from networkx.algorithms import isomorphism as iso
import graph_tool.all as gt
from graph_tool.topology import isomorphism, subgraph_isomorphism

from src.data import load_graph_pairs

# bad function name -- should change
#def load_graph_pairs():
#    enums = {idx: load_enumerated_graphs(size=idx) for idx in range(2, 6)}
#    transitions = [(g, h) for enum in enums.values() for g in enum for h in enum]
#    return transitions

# TODO
# n: the number of vertices in the shared vertex set
#   graph_tool ensures that a graph with n vertices has vertex names {0, 1, ... n-1}
# graphs: a temporal sequence of graphs on a shared vertex set
# empty: decides whether the empty (i.e., no edges) subgraph is considered
def egonet_transitions(n, graphs, empty=True):
    transitions = load_graph_pairs(sizes=[2, 3, 4, 5])
    counts = [0 for _ in transitions]
    for ego in tqdm(range(n), desc='ITERATING OVER NODES', total=n):
        for t in tqdm(range(len(graphs) - 1), desc='ITERATING OVER TIMES', total=len(graphs)-1, leave=False):
            g = graphs[t]
            h = graphs[t+1]

            g_vp = g.new_vertex_property('bool')
            h_vp = h.new_vertex_property('bool')

            g_nbhd = {w for w in g.vertex(ego).out_neighbors()} | {ego}
            h_nbhd = {w for w in h.vertex(ego).out_neighbors()} | {ego}

            for v in g.vertices():
                if v in g_nbhd:
                    g_vp[v] = True
                else:
                    g_vp[v] = False
            for v in h.vertices():
                if v in h_nbhd:
                    h_vp[v] = True
                else:
                    h_vp[v] = False

            g_egonet = gt.GraphView(g, vfilt=g_vp, directed=False)
            h_egonet = gt.GraphView(h, vfilt=h_vp, directed=False)

            for idx, (size, pregraph, postgraph) in tqdm(enumerate(transitions), desc='COMPUTING ISOMORPHISMS', total=len(transitions), leave=False):
                assert pregraph.num_vertices() == size and postgraph.num_vertices() == size

                g_iso = subgraph_isomorphism(pregraph, g_egonet, induced=True)
                h_iso = subgraph_isomorphism(postgraph, g_egonet, induced=True)
                continue

                for z in g_egonet.vertices():
                    print(z)
                print(g_iso[0].get_array())
                for z in g_iso[0]:
                    print(z in g_egonet.vertices())
                #print(g_iso, h_iso)
                exit()

            continue
            exit()

            #print(len(g_nbhd), g_egonet.num_vertices())
            #print(len(h_nbhd), h_egonet.num_vertices())

        pass
    return
