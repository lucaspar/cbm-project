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
    for v in range(N):
        for t in range(len(graphs) - 1):
            neighbors_G = [w for w in graphs[t].vertex(v).out_neighbors()]
            neighbors_H = [w for w in graphs[t+1].vertex(v).out_neighbors()]
            print(len(neighbors_G))

            #G = GraphView(graphs[t], vfilt=)
            #G = graphs[t]
            #H = graphs[t+1]
            #for g, h in transitions:
            #    pass
        pass
    return
