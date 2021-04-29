from networkx.algorithms import isomorphism as iso
import graph_tool.all as gt

from src.data import load_enumerated_graphs

# bad function name -- should change
def transition_counter():
    enum = load_enumerated_graphs()
    transitions = [(g, h) for g in enum for h in enum]
    return transitions

# TODO
def egonet_transitions(N, graphs):
    transitions = transition_counter()
    for ego in range(N):
        for t in range(len(graphs) - 1):
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

            print(len(g_nbhd), g_egonet.num_vertices())
            print(len(h_nbhd), h_egonet.num_vertices())
            exit()

            #G = GraphView(graphs[t], vfilt=)
            #G = graphs[t]
            #H = graphs[t+1]
            #for g, h in transitions:
            #    pass
        pass
    return
