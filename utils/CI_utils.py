import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from causallearn.search.ConstraintBased.PC import pc
from causaldag import PDAG
import numpy as np
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ScoreBased.GES import ges
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io


def save_data_to_file(data, filename):
    """
    Save data to a file using pickle
    
    Parameters:
    data: Any Python object that can be pickled
    filename (str): Name of the file to save the data to
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {filename}")

def load_data_from_file(filename):
    """
    Load pickled data from a file
    
    Parameters:
    filename (str): Name of the file to load the data from
    
    Returns:
    The unpickled data
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def visualize_graph(G, title="Network Graph", layout_func=nx.spring_layout, 
                   node_color='lightblue', node_size=500, 
                   with_labels=True, arrows=True):
    """
    Visualize a NetworkX graph with customizable options.
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The graph to visualize
    title : str
        Title of the plot
    layout_func : callable
        NetworkX layout function to use (e.g., nx.spring_layout, nx.circular_layout)
    node_color : str or list
        Color(s) of the nodes
    node_size : int or list
        Size(s) of the nodes
    with_labels : bool
        Whether to show node labels
    arrows : bool
        Whether to show arrow directions (for directed graphs)
    """
    plt.figure(figsize=(6, 4))
    pos = layout_func(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=arrows)
    
    # Draw labels if requested
    if with_labels:
        nx.draw_networkx_labels(G, pos)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def PC(df):
    # get df column names to list
    columns = list(df.columns)

    # to numpy array
    data = df.to_numpy()
    cg = pc(data,node_names=columns)
    
    # draw the graph
    # cg.draw_pydot_graph()
    pyd = GraphUtils.to_pydot(cg.G)
    pyd.write_png('pc_graph.png')
    return cg

def GES(df):
    Record = ges(df.to_numpy(),node_names=df.columns)

    pyd = GraphUtils.to_pydot(Record['G'])
    pyd.write_png('ges_graph.png')

    # tmp_png = pyd.create_png(f="png")
    # fp = io.BytesIO(tmp_png)
    # img = mpimg.imread(fp, format='png')
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()

    return Record

def adj2pdag(adj, columns):
    # columns = list(df.columns), ordered list of column names
    # convert casual dag to DAG PDAG
    nodes = set(columns)
    directed_edges = set()
    undirected_edges = set()
    for i in range(len(columns)):
        for j in range(i+1,len(columns)):
            if adj[i,j]==-1 and adj[j,i]==1:
                # directed edge i -> j
                directed_edges.add((columns[i],columns[j]))
            elif adj[i,j]==-1 and adj[j,i]==-1:
                # undirected edge i - j
                undirected_edges.add((columns[i],columns[j]))
            elif adj[i,j]==1 and adj[j,i]==1:
                # undirected edge i <-> j
                undirected_edges.add((columns[i],columns[j]))
    
    # init PDAG object
    pdag = PDAG(nodes=nodes, arcs=directed_edges, edges=undirected_edges, known_arcs=directed_edges)
    return pdag

# compare PC, GES graphs, if the edges are the same, then the edge is considered as true, otherwise false
# cg.G.graph # 
# Record['G'].graph # j,i =1 and i,j = -1, i->j; i,j=j,i=1, i - j
def combine_pdags(pdag1, pdag2):
    """
    Combine two PDAGs by taking the union of their directed and undirected edges.
    """
    
    # create result PDAG
    n = len(pdag1)
    result = np.zeros((n, n))

    edges1 = (pdag1 != 0)
    edges2 = (pdag2 != 0)

    for i in range(n):
        for j in range(i+1):
            if i == j:
                continue
            # if edge in both PDAGs, keep it
            if edges1[i,j] and edges1[j,i] and edges2[i,j] and edges2[j,i]:
                # check if the same direction, if not, then undirected
                if pdag1[i,j]==pdag2[i,j] and pdag1[j,i]==pdag2[j,i]:
                    result[i,j]=pdag1[i,j]
                    result[j,i]=pdag1[j,i]
                else:
                    result[i,j]=-1
                    result[j,i]=-1
            # if edge in only one PDAG, keep it as undirected
            elif (edges1[i,j] and edges1[j,i]) or (edges2[i,j] and edges2[j,i]):
                result[i,j] = result[j,i] = -1
                
    return result

def data2pdag(df):
    cg=PC(df)
    ges=GES(df)

    graph=combine_pdags(cg.G.graph, ges['G'].graph)
    cpdag=adj2pdag(graph, df.columns)
    return cpdag
