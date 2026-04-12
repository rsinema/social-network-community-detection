## Dendrogram
## 
## Mike Goodrich
## Brigham Young University
## Feb 2025, December 2024
## 
## Basic code started with sharing the get_all_partitions with chatGPT
## and asking for code to generate the link matrix required for scipy's
## dendrogram function. ChatGPT's code didn't work, but by using the
## understanding_dendrograms Jupyter notebook and iterating with my
## own code and chatGPT's code, this code was produced. 


from __future__ import annotations
import networkx as nx  # type: ignore
import numpy as np
from typing import Hashable, FrozenSet, List, Set, Tuple, Dict, Literal

Group = Set[Hashable]
Partition = Tuple[Group, ...]
PartitionWithHeight = Tuple[Partition, float]
HeightMetric = Literal["distance", "max_cluster"]


class DendrogramHandler:
    def __init__(self, G: nx.Graph, height_metric: HeightMetric = "distance"):
        """
        Build a linkage matrix and labels from G using Girvan-Newman splits.
        
        Parameters:
        -----------
        G : nx.Graph
            The graph to partition
        height_metric : HeightMetric
            "distance" - uses reverse order of edge removal (default)
            "max_cluster" - uses size of largest cluster being merged
        """
        self.height_metric = height_metric
        all_partitions_with_heights: List[PartitionWithHeight] = self.get_all_partitions_with_heights(G)
        self.link_matrix, self.link_matrix_labels = self.partitions_to_linkage(all_partitions_with_heights)

    def get_all_partitions_with_heights(self, G: nx.Graph, normalized: bool = True) -> List[PartitionWithHeight]:
        """
        Perform a Girvan-Newman style divisive clustering, with heights based on
        the reverse order of edge removal.

        Returns a list of (partition, height) pairs ordered from coarse -> fine,
        where partition is a tuple of sets (communities) and height represents
        the reverse order of edge removal (first edge removed has highest height).
        The first element is the coarse partition (all nodes) with height 0.0.
        """
        graph = G.copy()
        n_nodes: int = graph.number_of_nodes()

        partitions_with_order: List[Tuple[Partition, int]] = []
        # initial coarse partition (everything together)
        partitions_with_order.append((tuple([set(graph.nodes())]), 0))

        # if there are no edges, we're already fully split into singletons
        if graph.number_of_edges() == 0:
            singletons: Partition = tuple({v} for v in graph.nodes())
            partitions_with_order.append((singletons, 0))
            # Convert to heights format
            return [(partition, 0.0) for partition, _ in partitions_with_order]

        # Continue removing max-betweenness edges until no edges remain
        step = 1
        while graph.number_of_edges() > 0:
            # compute edge betweenness centrality
            betw: Dict[Tuple[Hashable, Hashable], float] = nx.edge_betweenness_centrality(graph, normalized=normalized)

            if not betw:
                break

            # maximum betweenness at this step
            max_bw: float = max(betw.values())

            # remove only one edge with maximum betweenness
            edge_to_remove = next(edge for edge, score in betw.items() if score == max_bw)
            graph.remove_edge(*edge_to_remove)

            # connected components become the new partition
            comps: Partition = tuple(set(c) for c in nx.connected_components(graph))
            partitions_with_order.append((comps, step))
            step += 1

            # stop if we've reached singletons
            if len(comps) == n_nodes:
                break

        # Convert order to height (reverse order: first edge removed = highest height)
        max_step = len(partitions_with_order) - 1
        partitions_with_heights: List[PartitionWithHeight] = [
            (partition, float(max_step - order)) for partition, order in partitions_with_order
        ]
        
        return partitions_with_heights

    def partitions_to_linkage(self, all_partitions_with_heights: List[PartitionWithHeight]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert a list of (partition, height) pairs (coarse -> fine) into a SciPy-style linkage matrix.
        The code inverts the divisive sequence (works fine -> coarse) to construct merges.

        Three alternative distance choices are shown; the BETW_HOST (betweenness) option
        is the active one (uncommented). The other two are left as commented alternatives.

        Returns:
            Z: np.ndarray shape (n-1, 4) linkage matrix
            labels: list[str] labels for leaf nodes (stringified original node ids)
        """
        # Extract the final partition (leaves) to form labels
        leaves_partition, _ = all_partitions_with_heights[-1]
        n_leaves: int = len(leaves_partition)
        labels: List[str] = [str(list(leaf)[0]) for leaf in leaves_partition]

        # clusters maps dendrogram-cluster-index -> frozenset of original leaf ids
        # Map each leaf cluster index to its actual node ID (not just the index)
        clusters: Dict[int, FrozenSet[Hashable]] = {}
        for i, leaf in enumerate(leaves_partition):
            node_id = list(leaf)[0]
            clusters[i] = frozenset([node_id])
        
        linkage_rows: List[List[float]] = []

        # ...existing code...
        # reverse partitions so we iterate from fine -> coarse
        partitions_with_heights_rev: List[PartitionWithHeight] = list(reversed(all_partitions_with_heights))

        # We'll need a local helper to find all merges between fine and coarse partitions
        def _find_all_merges(coarse: Partition, fine: Partition) -> List[Tuple[FrozenSet[Hashable], FrozenSet[Hashable]]]:
            """
            Given a coarse partition and the finer partition immediately before it (in the divisive order),
            find all pairs of groups in 'fine' that combined to make groups in 'coarse'.
            Multiple merges can happen at once if multiple edges with same betweenness are removed.
            
            For groups requiring multi-way merges (>2 parts), create binary merges sequentially.
            """
            merges = []
            for group in coarse:
                parts = [c for c in fine if c.issubset(group)]
                if len(parts) == 1:
                    # No merge needed - group stayed the same
                    continue
                elif len(parts) == 2:
                    # Simple binary merge
                    merges.append((frozenset(parts[0]), frozenset(parts[1])))
                elif len(parts) > 2:
                    # Multi-way merge - convert to sequence of binary merges
                    # Merge parts[0] and parts[1], then merge result with parts[2], etc.
                    for i in range(len(parts) - 1):
                        if i == 0:
                            merges.append((frozenset(parts[0]), frozenset(parts[1])))
                        else:
                            # Merge the accumulated set with the next part
                            accumulated = frozenset().union(*parts[:i+1])
                            merges.append((accumulated, frozenset(parts[i+1])))
            return merges

        def _get_ids(c1: FrozenSet[Hashable], c2: FrozenSet[Hashable], clusters_map: Dict[int, FrozenSet[Hashable]]) -> Tuple[int, int]:
            cid1: int | None = None
            cid2: int | None = None
            for cid, cl in clusters_map.items():
                if cl == c1:
                    cid1 = cid
                if cl == c2:
                    cid2 = cid
            if cid1 is None or cid2 is None:
                raise ValueError("Cluster id for a merging cluster not found")
            return cid1, cid2

        # counter for new cluster indices (SciPy-style: leaves are 0..n-1, new clusters n, n+1, ...)
        next_cluster_idx: int = n_leaves

        # For each step in the reversed list, determine which fine groups merged to form coarser groups
        for i in range(len(partitions_with_heights_rev) - 1):
            fine_partition, fine_height = partitions_with_heights_rev[i]
            coarse_partition, coarse_height = partitions_with_heights_rev[i + 1]

            # find all pairs of groups in the fine partition that merged
            merges = _find_all_merges(coarse_partition, fine_partition)
            
            # Skip if no merges occurred (partitions are the same)
            if not merges:
                continue
            
            # Process each merge
            for c1, c2 in merges:
                cid1, cid2 = _get_ids(c1, c2, clusters)

                # choose the distance for this merge based on height_metric
                if self.height_metric == "max_cluster":
                    # Use size of largest cluster being merged
                    distance = float(max(len(c1), len(c2)))
                else:  # "distance" (default)
                    # Use reverse order of edge removal
                    distance = float(coarse_height)

                # count = size of the newly formed cluster (number of leaves)
                count = float(len(c1 | c2))

                # add linkage row: [idx1, idx2, distance, count]
                linkage_rows.append([float(cid1), float(cid2), distance, count])

                # register new cluster in clusters dict
                clusters[next_cluster_idx] = frozenset(c1 | c2)
                next_cluster_idx += 1

        Z: np.ndarray = np.array(linkage_rows, dtype=np.float64)
        return Z, labels

    # (Optional) keep the original helper methods as public if you need to debug or extend:
    def debug_get_all_partitions(self, G: nx.Graph) -> List[PartitionWithHeight]:
        """Return partitions with heights (same as get_all_partitions_with_heights)."""
        return self.get_all_partitions_with_heights(G)