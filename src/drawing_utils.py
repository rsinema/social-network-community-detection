import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from typing import Hashable, Tuple, Set
from .dendrogram_handler_v2 import DendrogramHandler
from scipy.cluster.hierarchy import dendrogram # type: ignore
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import Literal

HeightMetric = Literal["distance", "max_cluster"]

#######################
## Drawing Utilities ##
#######################

def draw_edge_by_type(G: nx.Graph, 
                      pos: dict[Hashable, Tuple[float, float]], 
                      edge: Tuple[Hashable, Hashable], 
                      partition: Tuple[Set[Hashable], ...]
                      ) -> None:
    """
        Draw edges between nodes in different partitions using dashed lines.
        Draw edges between nodes within the same partition using solid lines.
    """
    edge_style = 'dashed'
    for part in partition:
        if edge[0] in part and edge[1] in part:
            edge_style = 'solid'
            break
    nx.draw_networkx_edges(G, pos, edgelist=[edge], style = edge_style)

def count_edges_cut(G: nx.Graph,
                    partition: Tuple[Set[Hashable], ...]
                    ) -> int:
    """ 
        Count the number of edges cut if the nodes in graph G are split into 
        the groups in the partition
    """
    cut_size:int = 0
    for i in range(len(partition) - 1):
        for j in range(i+1, len(partition)):
            for u in partition[i]:
                for v in G.neighbors(u):
                    if v in partition[j]:
                        cut_size += 1
    return cut_size

def show_graph(G: nx.Graph,
                    pos: dict[Hashable, Tuple[float, float]] | None = None,
                    title: str = "",
                    show_node_labels: bool = True
                    ) -> None:
    """ 
        Show the networkx graph 
    """
    
    if pos is None: 
        #pos = nx.spring_layout(G, seed = 0)
        #pos = nx.nx_pydot.pydot_layout(G, prog = "neato")
        pos = nx.nx_pydot.graphviz_layout(G, prog = "neato")
    nx.draw(
        G,
        pos,
        node_color='lightblue',
        alpha=0.8,
        with_labels=show_node_labels,
    )
    plt.title(title)
    plt.axis('off')

def show_partitions(G: nx.Graph,
                    partition: Tuple[Set[Hashable], ...], 
                    pos: dict[Hashable, Tuple[float, float]] | None = None,
                    title: str = "",
                    show_node_labels: bool = True
                    ) -> None:
    """ 
        Show the networkx graph with colors and edges indicating properties
        of the partition

        Edges:
        • Dashed lines indicate edges between nodes in different partitions
        • Solid lines indicate edges between nodes in the same partition

        Nodes:
        • All nodes in the same partition get mapped to the same color
        • When there are more partitions than ther are in the color pallette, repeat colors
    """
    #color_list = ['c','m','y','g','r']
    color_list: list[str] = ['y', 'lightblue', 'violet', 'salmon', 
                         'aquamarine', 'magenta', 'lightgray', 'linen']
    plt.clf()
    ax: Axes = plt.gca()
    if pos is None: 
        #pos = nx.spring_layout(G, seed = 0)
        pos = nx.nx_pydot.graphviz_layout(G, prog = "neato")
        #pos = nx.nx_pydot.pydot_layout(G, prog = "neato")
    for i in range(len(partition)):
        nx.draw_networkx_nodes(partition[i],pos,node_color=color_list[i%len(color_list)], alpha = 0.8)
    for edge in G.edges:
        draw_edge_by_type(G, pos, edge, partition)
    if show_node_labels:
        nx.draw_networkx_labels(G, pos)
    if len(G.edges) == 0:
        mod = 0
    else:
        mod = nx.algorithms.community.quality.modularity(G,partition)
    if title[-1] == ":" or title[-1] == "\n":
        title = title + " groups=" + str(len(partition))
    else:
        title = title + ", groups=" + str(len(partition))
    title = title + ", edges cut=" + str(count_edges_cut(G, partition))
    title = title + ", mod = " + str(np.round(mod,2))

    ax.set_title(title)
    ax.set_axis_off()

def show_partitions_with_scaled_nodesize(G: nx.Graph,
                    partition: Tuple[Set[Hashable], ...], 
                    pos: dict[Hashable, Tuple[float, float]] | None = None,
                    title: str = ""
                    ) -> None:
    """ 
        Show the networkx graph with colors and edges indicating properties
        of the partition. The node size is determined by node degree

        Edges:
        • Dashed lines indicate edges between nodes in different partitions
        • Solid lines indicate edges between nodes in the same partition

        Nodes:
        • All nodes in the same partition get mapped to the same color
        • When there are more partitions than ther are in the color pallette, repeat colors
    """
    #color_list = ['c','m','y','g','r']
    color_list: list[str] = ['y', 'lightblue', 'violet', 'salmon', 
                         'aquamarine', 'magenta', 'lightgray', 'linen']
    plt.figure(figsize=(8.0,8))
    ax: Axes = plt.gca()
    if pos is None: 
        #pos = nx.spring_layout(G, seed = 0)
        pos = nx.nx_pydot.pydot_layout(G, prog = "neato")
    for i in range(len(partition)):
        nx.draw_networkx_nodes(partition[i],
                               pos,
                               node_color=color_list[i%len(color_list)], 
                               alpha = 0.8,
                               node_size=[50 + 150*nx.degree(G, node) for node in partition[i]])
    for edge in G.edges:
        draw_edge_by_type(G, pos, edge, partition)
    nx.draw_networkx_labels(G,pos)
    if len(G.edges) == 0:
        mod = 0
    else:
        mod = nx.algorithms.community.quality.modularity(G,partition)
    if title[-1] == ":" or title[-1] == "\n":
        title = title + " groups=" + str(len(partition))
    else:
        title = title + ", groups=" + str(len(partition))
    title = title + ", edges cut=" + str(count_edges_cut(G, partition))
    title = title + ", mod = " + str(np.round(mod,2))

    ax.set_title(title)
    ax.set_axis_off()

def show_dendrogram(G: nx.Graph,
                    title: str = "Dendrogram",
                    height_metric: HeightMetric = "distance") -> None:
    plt.figure()
    myHandler: DendrogramHandler = DendrogramHandler(G, height_metric=height_metric)
    Z = myHandler.link_matrix       # Python style guides suggest direct access of public class variables
    ZLabels = myHandler.link_matrix_labels
    dendrogram(Z, labels=ZLabels)
    plt.title(title)
    plt.xlabel("Node")
    if height_metric == "max_cluster":
        plt.ylabel("Max Cluster Size")
    else:
        plt.ylabel("Distance")
    del myHandler

def show_kCores(G: nx.Graph,
                title: str = "K-core of Network"
                ) -> None:
    """ Visualize by k-cores. 
    Thanks to [Corralien's response on stackoverflow]
    (https://stackoverflow.com/questions/70297329/visualization-of-k-cores-using-networkx).
    """
    # build a dictionary of k-level with the list of nodes
    kcores = defaultdict(list)
    for n, k in nx.core_number(G).items():
        kcores[k].append(n)

    # compute position of each node with shell layout
    nlist = []
    for k in sorted(kcores.keys(),reverse=True):
        nlist.append(kcores[k])
    pos = nx.layout.shell_layout(G, nlist = nlist)
    colors = ['y', 'lightblue', 'violet', 'salmon', 'aquamarine', 'magenta', 'lightgray', 'linen','green','lightblue','olive', 'cyan', 'b']
    legend_elements = []

    # draw nodes, edges and labels
    for i, kcore in enumerate(sorted(list(kcores.keys()),reverse = True)):
        nodes = kcores[kcore]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors[i%len(colors)], alpha=0.5)
        label = f"kcore = {kcore}"
        color = colors[i%len(colors)]
        legend_elements.append(Line2D([0], [0], marker='o', color=color, label=label,markerfacecolor=color, markersize=15))
    nx.draw_networkx_edges(G, pos, width=0.3, edge_color='lightgray')
    #nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.legend(handles = legend_elements, loc = 'best')
    plt.axis('off')

def show_kCores_by_partition(G:nx.Graph, 
                             colors: list[str], 
                             title: str = "K-core of Network"
                             ) -> None:
    """ Visualize by k-cores. 
    Thanks to [Corralien's response on stackoverflow]
    (https://stackoverflow.com/questions/70297329/visualization-of-k-cores-using-networkx).
    """
    # build a dictionary of k-level with the list of nodes
    kcores = defaultdict(list)
    for n, k in nx.core_number(G).items():
        kcores[k].append(n)

    # Shapes
    shapes = ["o", "v", "s", "*", "+", "d"]

    # compute position of each node with shell layout
    nlist = []
    for k in sorted(kcores.keys(),reverse=True):
        nlist.append(kcores[k])
    pos = nx.layout.shell_layout(G, nlist = nlist)
    legend_elements = []

    # draw nodes, edges and labels
    for kcore in sorted(kcores.keys(),reverse=True):
        nodes = kcores[kcore]
        shape = shapes[kcore%len(shapes)]
        
        #nx.draw_networkx_nodes(self.G, pos, nodelist=nodes, node_color=colors[nodes[0]], node_shape=shape, alpha = 0.5, node_size=90)
        for node in nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=colors[node], node_shape=shape, alpha = 0.5, node_size=90)
        label = f"kcore = {kcore}"
        legend_elements.append(Line2D([0], [0], marker=shape, color='k', markerfacecolor = 'w', label=label, markersize=10))
    
    nx.draw_networkx_edges(G, pos, width=0.1)
    #nx.draw_networkx_labels(self.G, pos)
    plt.title(title)
    plt.legend(handles = legend_elements, loc = 'best')

def show_2D_scatterplot(vector1: NDArray[np.float32],
                        vector2: NDArray[np.float32],
                        colors: list[str],
                        xlabel: str = "values of first eigenvector",
                        ylabel: str = "values of second eigenvector",
                        title: str = "Clusters of L or M"
                        ) -> None:
    
    plt.scatter(vector1, vector2,s=100,alpha = 0.8, color = colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)  

def show_node_probability(G:nx.Graph,
                          probabilities: list[float],
                          title: str ="karate network",
                          show_scale: bool = False, 
                          show_degree_as_size: bool = False, 
                          show_labels: bool = True
                          ) -> None:
    pos: dict[Hashable, tuple[float, float]] = nx.nx_pydot.graphviz_layout(G, prog = "neato")
    plt.figure()
    plt.axis('off')
    plt.title(title)
    
    # C style type declaration
    my_node_size: list[int]
    
    if show_degree_as_size:
        my_node_size = [50 + 150*k for k in dict(G.degree).values()]
    else: 
        my_node_size = [200 for _ in G.nodes()]
    nx.draw_networkx(G, 
                 pos = pos,
                 node_color=probabilities,
                 node_size=my_node_size,
                 cmap='cool',
                 font_size=9,
                 font_color='white',
                 with_labels=show_labels)
    if show_scale:
        sm = plt.cm.ScalarMappable(cmap = 'cool',norm=plt.Normalize(vmin = 0, vmax=max(probabilities)))
        _ = plt.colorbar(sm, ax=plt.gca())

def show_graph_by_pagerank(G:nx.Graph,
                          title: str ="Graph nodes by page rank",
                          pos: dict[Hashable,tuple[float,float]] | None = None,
                          show_scale: bool = False,  
                          show_labels: bool = True
                          ) -> None:
    if pos is None:
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog="neato")
        except Exception as e:
            print(f"Graphviz layout failed, falling back to spring_layout: {e}")
    pos = nx.spring_layout(G, seed=42)
    plt.figure()
    plt.axis('off')
    plt.title(title)

    pageranks = nx.pagerank(G)
    node_values: list[float] = [float(v) for v in pageranks.values()]
    
    # C style type declaration
    my_node_size: list[int]
    my_node_size = [20 for _ in G.nodes()]
    nx.draw_networkx(G, 
                 pos = pos,
                 node_color=node_values,
                 node_size=my_node_size,
                 cmap='cool',
                 font_size=9,
                 font_color='white',
                 with_labels=show_labels,
                 alpha = 0.5)
    if show_scale:
        sm = plt.cm.ScalarMappable(cmap = 'cool',norm=plt.Normalize(vmin = min(node_values), vmax=max(node_values)))
        _ = plt.colorbar(sm, ax=plt.gca())