import networkx as nx
import numpy as np
from numpy.typing import NDArray
from typing import Hashable, Tuple, Set, List
import random
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, diags
from copy import deepcopy
from typing import Callable as function
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize



###############################################
## Newman Modularity Hill Climbing Utilities ##
###############################################
def split_into_random_shores(G: nx.Graph
                             ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """ 
        The Newman algorithm for random and greedy hill-climbing 
        starts with nodes assigned randomly two two shores.
    """
    shore_size: int = np.ceil(len(G.nodes()))/2
    shore1: set[Hashable] = set(G.nodes)
    shore2: set[Hashable] = set()
    
    while len(shore2) < shore_size:
        node: Hashable = random.choice(list(shore1))
        shore2.add(node)
        shore1.remove(node)
    return (shore1, shore2)

def swap_shores(partition: Tuple[Set[Hashable], Set[Hashable]], 
                node: Hashable
                ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """ 
        Swapping shores means moving a node from one
        partition to another.
    """
    shore1: Set[Hashable] = deepcopy(partition[0])
    shore2: Set[Hashable] = deepcopy(partition[1])
    if node in partition[0]:
        shore1.remove(node)
        shore2.add(node)
    else:
        shore2.remove(node)
        shore1.add(node)
    return (shore1, shore2)

def find_best_node_to_swap(G: nx.Graph,
                           partition: Tuple[Set[Hashable], Set[Hashable]],
                           already_swapped: Set[Hashable]
                           ) -> Hashable | None:
    best_mod: float = -np.inf
    # Node that produces the highest modularity increase if it swaps shores
    best_node_to_swap: Hashable | None = None  
    # Track nodes that have already been swapped
    for node in set(G.nodes()) - already_swapped:
        possible_partition = swap_shores(partition, node)
        mod = nx.algorithms.community.quality.modularity(G,possible_partition)
        if mod > best_mod:
            best_mod = mod
            best_node_to_swap = node
    return best_node_to_swap

def Newman_hill_climbing(G: nx.Graph
                         ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """ 
        Implement Newman's hill climbing algorithm for estimating
        the partition that maximizes modularity.

        Returns:
            The best partition found
    """
    # Initialize
    partition: Tuple[Set[Hashable], Set[Hashable]] = split_into_random_shores(G)
    already_swapped: set[Hashable] = set()
    best_partition: Tuple[Set[Hashable], Set[Hashable]] = deepcopy(partition)
    best_modularity: float = nx.community.modularity(G, partition)
    
    best_node_to_swap: Hashable| None = find_best_node_to_swap(G, partition, already_swapped)
    while best_node_to_swap is not None:
        partition = swap_shores(partition, best_node_to_swap)
        already_swapped.add(best_node_to_swap)
        
        if nx.community.modularity(G, partition) >= best_modularity:
            best_modularity = nx.community.modularity(G, partition)
            best_partition = deepcopy(partition)
        else:
            return best_partition  # Stop when modularity starts going down

        best_node_to_swap = find_best_node_to_swap(G, partition, already_swapped)

    return best_partition

#######################################
## Spectral Modularity Cut Utilities ##
#######################################
def get_leading_eigenvector(G: nx.Graph
                            ) -> Tuple[float, NDArray[np.float32]]:
    
    B: NDArray[np.float32] = nx.modularity_matrix(G, nodelist=sorted(G.nodes()))
    eigenvalues, eigenvectors = np.linalg.eig(B)
    largest_eigenvalue_index = np.argmax(eigenvalues)
    largest_eigenvalue = eigenvalues[largest_eigenvalue_index]
    leading_eigenvector = eigenvectors[:, largest_eigenvalue_index]
    return largest_eigenvalue, leading_eigenvector

def get_shores_from_eigenvector(G: nx.Graph,
                                x: NDArray[np.float32]) -> Tuple[Set[Hashable], Set[Hashable]]:
    shore1: Set[Hashable] = set()
    shore2: Set[Hashable] = set()
    nodes = sorted(list(G.nodes()))
    for i in range(len(nodes)):
        if x[i] >= 0: 
            shore1.add(nodes[i])
        else: 
            shore2.add(nodes[i])
    return (shore1, shore2)

def modularity_spectral_split(G: nx.Graph) -> Tuple[Set[Hashable], Set[Hashable]]:
    _, v = get_leading_eigenvector(G)
    return get_shores_from_eigenvector(G,v)

###########################################
## Kernighan-Lin Hill-Climbing Graph Cut ##
###########################################
def initialize_partition(G: nx.Graph,
                         seed: int | None = None
                         ) -> Tuple[set[Hashable], set[Hashable]]:
    """
        Input: networkx undirected graph
        Output: two sets with the graph nodes split in half
    """
    # Check types
    if type(G) is not nx.Graph:
        raise TypeError("Requires undirected graph")
    
    # Initialize partitions
    nodes: list[Hashable] = list(G.nodes())
    if seed is not None:
        random.seed(seed)
    random.shuffle(nodes)
    mid: int = len(nodes) // 2
    A: set[Hashable] = set(nodes[:mid])
    B: set[Hashable] = set(nodes[mid:])

    return (A,B)

def gain(G: nx.Graph,
         u: Hashable, 
         group_A: set[Hashable], 
         group_B: set[Hashable]
         ) -> int:
    """ 
        count the net gain in the number of edges cut if we swap node u
        from group A to group B. See D_a from https://en.wikipedia.org/wiki/Kernighan-Lin_algorithm
    """

    internal = sum(1 for v in G.neighbors(u) if v in group_A)
    external = sum(1 for v in G.neighbors(u) if v in group_B)
    return external - internal

def gain_from_swap(G: nx.Graph,
                   u: Hashable,
                   v: Hashable,
                   group_A: set[Hashable],
                   group_B: set[Hashable]
                   ) -> int:
    """ 
        Compute the net gain from swapping a and b using the equation
        T_{old} - T_{new} = D(a,A) + D(b,B) - 2 delta({a,b} in E)
    """
    gain_u: int = gain(G, u, group_A, group_B)
    gain_v: int = gain(G, v, group_B, group_A)
    delta: int = int(v in G.neighbors(u))

    return gain_u + gain_v - 2*delta

def kernighan_lin_bisection(G: nx.Graph, 
                            max_iter: int=100,
                            seed: int | None = None
                            ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Input: undirected graph
    """
    # Check types
    if type(G) is not nx.Graph:
        raise TypeError("Requires undirected graph")
    
    # Initialize
    group_A, group_B = initialize_partition(G, seed)
    
    # Compute scores for all swaps
    for _ in range(max_iter):
        gains: List[Tuple[int, Hashable, Hashable]] = []
        for u in group_A:
            for v in group_B:
                swap_score: int = gain_from_swap(G, u, v, group_A, group_B)
                gains.append((swap_score, u, v))
        
        gains.sort(reverse=True)
        
        best_gain: int = 0
        best_pair: tuple[Hashable, Hashable] | None = None
        current_gain: int = 0
        for gain_value, u, v in gains:
            current_gain += gain_value
            if current_gain > best_gain:
                best_gain = current_gain
                best_pair = (u, v)
        
        if best_pair is not None:
            u, v = best_pair
            group_A.remove(u)
            group_B.add(u)
            group_B.remove(v)
            group_A.add(v)
        else:
            break
    
    return group_A, group_B

################################
## Minimum Balanced Graph Cut ##
################################

def get_fiedler_eigenvector_sparse(L: csr_matrix) -> NDArray[np.float32]:
    """
        Computes the numerically stable Fiedler eigenvector for a given sparse 
        Laplacian matrix. Generated by chatGPT in response to some numerical 
        stability problems that arise when some of the vertices in the graph have 
        degree one.
    """
    eigenvectors: NDArray[np.float32]
    _, eigenvectors = eigsh(L, k=2, which="SM")  # Compute two smallest eigenvalues: "SM" means "smallest magnitude"
    return eigenvectors[:, 1]  # Return the second smallest eigenvector

def get_fiedler_eigenvector(Laplacian: NDArray[np.float32]
                            ) -> NDArray[np.float32]:
    
    eigenvalues, eigenvectors = np.linalg.eig(Laplacian)
    # choose second smallest as fiedler eigenvalue
    sorted_indices = np.argsort(eigenvalues)
    # return eigenvector of second smallest index
    return(eigenvectors[:,sorted_indices[1]])

def laplacian_graph_cut(G: nx.Graph) -> Tuple[Set[Hashable], Set[Hashable]]:
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()
    v = get_fiedler_eigenvector(L)
    return get_shores_from_eigenvector(G,v)

def laplacian_graph_cut_sparse(G: nx.Graph,
                               get_shores: function[[nx.Graph, NDArray[np.float32]], Tuple[Set[Hashable], Set[Hashable]]] = get_shores_from_eigenvector
                               ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Computes graph cut using the standard Laplacian matrix with sparse computations.
        Generated by chatGPT in response to help finding numerically stable computations
    """
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).astype(float)  # Sparse matrix
    v = get_fiedler_eigenvector_sparse(L)
    return get_shores(G,v)

def normalized_laplacian_graph_cut(G: nx.Graph) -> Tuple[Set[Hashable], Set[Hashable]]:
    N = nx.normalized_laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()
    v = get_fiedler_eigenvector(N)
    return get_shores_from_eigenvector(G,v)

def normalized_laplacian_graph_cut_sparse(G: nx.Graph,
                                          get_shores: function[[nx.Graph, NDArray[np.float32]], Tuple[Set[Hashable], Set[Hashable]]] = get_shores_from_eigenvector
                                          ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Computes graph cut using the normalized Laplacian matrix with sparse computations.
        Generated by chatGPT in response to help finding numerically stable computations
    """
    N = nx.normalized_laplacian_matrix(G, nodelist=sorted(G.nodes())).astype(float)  # Sparse matrix
    v = get_fiedler_eigenvector_sparse(N)
    return get_shores(G,v)

def randomwalk_laplacian_graph_cut(G: nx.Graph) -> Tuple[Set[Hashable], Set[Hashable]]:
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()
    D = compute_degree_matrix(G)
    v = get_fiedler_eigenvector(L@np.linalg.inv(D))
    return get_shores_from_eigenvector(G,v)

def randomwalk_laplacian_graph_cut_sparse(G: nx.Graph,
                                          get_shores: function[[nx.Graph, NDArray[np.float32]], Tuple[Set[Hashable], Set[Hashable]]] = get_shores_from_eigenvector
                                          ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Computes graph cut using the stable random walk Laplacian (fully sparse).
        Generated by chatGPT in response to help finding numerically stable computations
    """

    # Compute Standard Laplacian (Sparse)
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).astype(float)

    # Compute Degree Matrix (Sparse D⁻¹)
    degrees = np.array([G.degree(node) for node in sorted(G.nodes())], dtype=float)
    degrees[degrees == 0] = 1  # Avoid division by zero
    D_inv = diags(1.0 / degrees)  # Sparse D^(-1)

    # Compute Random Walk Laplacian: L_rw = D⁻¹ L
    L_rw = L @ D_inv   # Sparse matrix multiplication

    # Compute Fiedler Eigenvector
    v = get_fiedler_eigenvector_sparse(L_rw)

    return get_shores(G,v)

def compute_degree_matrix(G: nx.Graph) -> NDArray[np.float32]:
    """Computes the degree matrix D of a graph G."""
    degrees = np.array([G.degree(node) for node in sorted(G.nodes())], dtype=float)  # Extract node degrees
    D = np.diag(degrees)  # Create a diagonal matrix with degrees
    return D

def get_shores_from_eigenvector_median(G: nx.Graph,
                                       x: NDArray[np.float32]
                                       ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Partitions nodes into two sets based on the median of the Fiedler eigenvector.
        Generated by chatGPT in response to prompts about how to improve partitioning
    """
    nodes = sorted(G.nodes())  # Ensure consistent ordering
    median_value = np.median(x)  # Compute the median of the eigenvector values

    shore1 = {nodes[i] for i in range(len(nodes)) if x[i] >= median_value}
    shore2 = set(G.nodes()) - shore1

    return shore1, shore2
    
##########################
## Spectral Clustering  ##
##########################

def get_k_principal_eigenvectors_sparse(A: csr_matrix, k: int) -> NDArray[np.float32]:
    """
        Computes the eigenvectors corresponding to the k eigenvalues with
        largest modulus (largest absolute value) of a sparse symmetric matrix.

        Returns eigenvectors ordered from largest to smallest eigenvalue
        modulus so column 0 corresponds to the dominant eigenvalue by modulus.
    """
    eigenvalues, eigenvectors = eigsh(A, k=k, which="LM")
    order = np.lexsort((-np.real(eigenvalues), -np.abs(eigenvalues)))
    return np.asarray(eigenvectors[:, order], dtype=np.float32)

def get_k_fiedler_eigenvectors_sparse(L: csr_matrix, k: int) -> NDArray[np.float32]:
    """
        Computes the numerically stable Fiedler eigenvectors for a given sparse 
        Laplacian matrix. Generated by chatGPT in response to some numerical 
        stability problems that arise when some of the vertices in the graph have 
        degree one. Unlike the function above which just returns the fiedler eigenvector
        this function can return more than the two smallest eigenvectors.

        Assumes that the graph is connected so that there is only one eigenvalue
        with value 0

        Returns the k smallest nontrivial (lambda=0) eigenvectors. The first eigenvector
        is not returned since it is zero.
    """
    # Ensure that k is not bigger than the number of possible non-zero eigenvalues
    if k >= L.shape[0] - 1:
        raise ValueError("k must be smaller than the dimension of the Laplacian.")
    
    _, eigenvectors = eigsh(L, k=k+1, which="SM")  # Compute two smallest eigenvalues: "SM" means "smallest magnitude"
    return np.asarray(eigenvectors[:, 1:k+1], dtype=np.float32)  # Return the second smallest eigenvector

def get_partition_from_single_eigenvector(
    eigenvector: NDArray[np.float32],
    nodes: list[Hashable] | None = None,
    method: str = "median",
    num_clusters: int = 2,
) -> list[set[Hashable]]:
    """Create node clusters from a single eigenvector.

    Methods:
    - "sign": split by value >= 0 vs < 0
    - "median": split by value >= median vs < median
    - "kmeans": 1D k-means on eigenvector values
    """
    values: NDArray[np.float32] = np.asarray(eigenvector, dtype=np.float32).reshape(-1)
    node_list: list[Hashable] = list(range(len(values))) if nodes is None else nodes

    if len(node_list) != len(values):
        raise ValueError("Length of nodes must equal length of eigenvector")

    if method == "sign":
        return [
            {node_list[i] for i, v in enumerate(values) if v >= 0},
            {node_list[i] for i, v in enumerate(values) if v < 0},
        ]

    if method == "median":
        threshold = float(np.median(values))
        return [
            {node_list[i] for i, v in enumerate(values) if v >= threshold},
            {node_list[i] for i, v in enumerate(values) if v < threshold},
        ]

    if method == "kmeans":
        if num_clusters < 2:
            raise ValueError("num_clusters must be at least 2 for method='kmeans'")
        X = values.reshape(-1, 1)
        kmeans = KMeans(
            init="k-means++",
            n_clusters=num_clusters,
            n_init=50,
            max_iter=500,
            algorithm="lloyd",
            random_state=1234,
        )
        labels = kmeans.fit_predict(X)
        return [{node_list[i] for i, lbl in enumerate(labels) if lbl == c} for c in range(num_clusters)]

    raise ValueError("method must be one of: 'sign', 'median', 'kmeans'")

def get_clusters(embedding: NDArray[np.float32], 
                 num_clusters: int = 4
                 ) -> KMeans:
    # Normalize rows so clustering uses direction in spectral embedding space.
    X = normalize(embedding, norm="l2", axis=1)
    kmeans = KMeans(
        init="k-means++",
        n_clusters=num_clusters,
        n_init=50,
        max_iter=500,
        algorithm="lloyd",
        random_state=1234
        )
    kmeans.fit(X)
    return kmeans

def get_colors_from_clusters(embedding: NDArray[np.float32], 
                             num_clusters: int = 4
                             ) -> list[str]:
    kmeans = get_clusters(embedding, num_clusters=num_clusters)
    labels = kmeans.labels_
    color_template = ['y', 'c', 'm', 'k', 'red', 'green', 'lightblue']
    color: list[str] = [color_template[x % len(color_template)] for x in list(labels)]
    return color
