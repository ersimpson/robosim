import typing as T

import networkx as nx
import numpy as np

from utils import is_point_in_radius


def create_rgg(N: int) -> T.Tuple[nx.Graph, float]:
    """Create a random geometric graph with N nodes.

    NOTE: The radius is calculated such that the graph is almost surely
    connected.

    Args:
        N (int): the number of nodes in the graph.

    Returns:
        A tuple containing the random geometric graph and the radius.
    """
    radius = np.sqrt(np.log(N) / N)
    radius = np.nextafter(radius, radius+1)
    return nx.random_geometric_graph(N, radius), radius


def select_target_node(
    G: nx.Graph,
    criteria: T.Optional[T.Callable[[nx.Graph], T.Mapping[int, float]]] = None,
) -> int:
    """Select a target node in the graph based on a mapping criteria.

    NOTE: The criteria function should return a mapping of node indices to
    scores given the graph as input. If no criteria function is provided, then
    the highest closeness centrality of all nodes is used.

    Args:
        G (nx.Graph): the graph to select the target node from.
        criteria (Callable[[nx.Graph], Mapping[int, float]]): the criteria
            function to select the target node.

    Returns:
        An int of the index of the selected target node.
    """
    if criteria is None:
        criteria = nx.closeness_centrality
    scores = criteria(G)
    max_node = 0
    for node, score in scores.items():
        max_node = node if score > scores[max_node] else max_node
    return max_node


def select_start_node(G: nx.Graph, target: int) -> int:
    """Select a random starting node in the graph.

    NOTE: The target node will not be selected.

    Args:
        G (nx.Graph): the graph to select the starting node from.
        target (int): the target node to avoid.

    Returns:
        An int of the index of the selected starting node.
    """
    N = len(G.nodes)
    while (node := np.random.choice(N)) == target:
        continue
    return node


def get_node_weights(G: nx.Graph, target: int, start: int) -> T.Mapping[int, float]:
    """Compute the weights of all nodes in the graph based on the shortest path
    to the target node.

    Args:
        G (nx.Graph): the graph to compute the node weights from.
        target (int): the target node index.
        start (int): the start node index.

    Returns:
        A mapping of node indices to weights.
    """
    starting_nodes = [node for node in G.nodes if node != start]
    weights = {}
    for node in starting_nodes:
        if not nx.has_path(G, node, target):
            weights[node] = 0
            continue
        shortest_path_len = nx.astar_path_length(G, node, target)
        weights[node] = 1000 if node == target else 1 / shortest_path_len
    return weights


def set_node_weights(G: nx.Graph, weights: T.Mapping[int, float]) -> nx.Graph:
    """Set the node weights in the graph.

    Args:
        G (nx.Graph): the graph to set the node weights in.
        weights (Mapping[int, float]): the mapping of node indices to weights.
    
    Returns:
        The graph with the node weights set.
    """
    for node, weight in weights.items():
        G.nodes[node]["weight"] = weight
    return G


def get_nodes_in_radius(G: nx.Graph, radius: float, x: float, y: float) -> T.List[int]:
    """Get the nodes in the graph within a given radius of a point.
    
    Args:
        G (nx.Graph): the graph to search for nodes in.
        radius (float): the radius to search within.
        x (float): the x-coordinate of the point.
        y (float): the y-coordinate of the point.

    Returns:
        A list of node indices within the radius of the point.
    """
    pos = get_node_positions(G)
    nodes = []
    for node, (cx, cy) in pos.items():
        if not is_point_in_radius(cx, cy, x, y, radius):
            continue
        nodes.append(node)
    return nodes


def get_node_positions(G: nx.Graph) -> T.Mapping[int, T.Tuple[float, float]]:
    """Get the positions of all nodes in the graph.

    Args:
        G (nx.Graph): the graph to get the node positions from.

    Returns:
        A mapping of node indices to (x, y) coordinates.
    """
    return nx.get_node_attributes(G, "pos")
