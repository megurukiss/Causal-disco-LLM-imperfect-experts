from cdt.metrics import SHD
import networkx as nx
import numpy as np
from utils.dag_utils import list_of_tuples_to_digraph, get_undirected_edges_pdag

def get_mec_shd(true_G, mec):
    """
    the graphs need to be ordered to be comparable
    """
    target = nx.to_numpy_array(true_G)
    pred = np.stack([nx.to_numpy_array(list_of_tuples_to_digraph(dag)) for dag in mec], -1).sum(-1)
    pred = np.clip(pred, 0, 1)
    shd = SHD(target, pred, double_for_anticausal=False)
    return shd, pred

def edge_acc(directed_set,cpdag, true_G):
    arcs = cpdag.arcs
    undirected_edges=get_undirected_edges_pdag(cpdag)

    true_edges = true_G.edges
    correct = 0
    count = 0
    for edge in true_G.edges:
        if edge in arcs or edge in directed_set:
            correct += 1
        count += 1
    return correct/count