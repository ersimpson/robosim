"""Module for robot trajectory regressor model."""

import typing as T

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import aggr

from utils import get_angle_between_vectors, get_local_to_world_orientation, is_point_in_radius


NODE_FEATURES_V1 = [
    "x", "y", # Node position
    "rx", "ry", # Robot position
    "tx", "ty", # Target position
    "rdist", # Node to robot distance
    "tdist", # Node to target distance
    "rangle", # Node angle relative to robot world position
    "tangle", # Node angle relative to target position
]

NODE_FEATURES_V2 = [
    "x", "y", # Node position
    "rx", "ry", # Robot position
    "tx", "ty", # Target position
    "rdist", # Node to robot distance
    "tdist", # Node to target distance
    "rangle", # Node angle relative to robot world position,
    "tangle", # Node angle relative to target position
    "tgt_connected", # Node connected to target
    "closeness_centrality", # Closeness centrality
    "betweenness_centrality", # Betweenness centrality
]


class TrajectoryRegressor(nn.Module):
    """Model for predicting the next trajectory position given a subgraph of
    adjacent nodes to an existing position in a robot navigation graph.
    """

    def __init__(self,
        gnn_hidden_dim: int,
        embedding_dim: int,
        rnn_hidden_dim: int,
        dropout: float = 0.1,
    ):
        """Inits TrajectoryRegressor.
        
        Args:
            gnn_hidden_dim (int): the dimension of the hidden layer in the GCN.
            embedding_dim (int): the dimension of the embedded output.
            rnn_hidden_dim (int): the dimension of the hidden layer in the RNN.
            dropout (float): the dropout rate.
        """
        super(TrajectoryRegressor, self).__init__()

        self.embedding_dim = embedding_dim
        self.gnn = gnn.GCN(
            in_channels=-1,
            hidden_channels=gnn_hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
            dropout=dropout,
        )
        self.pool = aggr.MeanAggregation()
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
        )
        self.linear = nn.Linear(
            in_features=rnn_hidden_dim,
            out_features=2, # x and y coordinates
        )

    def forward(self, data: T.List[Data]) -> torch.Tensor:
        """Feedforward layer for TrajectoryRegressor model.

        Args:
            data (list[Data]): a list of input graphs with node features.

        Returns:
            A tensor of the predicted trajectory.
        """
        h = torch.zeros((len(data), self.embedding_dim))
        for i, subgraph in enumerate(data):
            x, edge_index, edge_weight = subgraph.x, subgraph.edge_index, subgraph.edge_weight
            x = x.float()
            z = F.relu(self.gnn(x, edge_index, edge_weight))
            h[i, :] = self.pool(z)
        h, _ = self.rnn(h)
        out = self.linear(h)
        out = F.sigmoid(out)
        return out


def add_node_features(
    model_version: str,
    graph: nx.Graph,
    subgraph: nx.Graph,
    radius: float,
    pos: T.Tuple[float, float],
    tpos: T.Tuple[float, float],
) -> nx.Graph:
    """Add node features to a graph based on the model version.

    Args:
        model_version (str): the version of the model to use.
        graph (nx.Graph): the graph that the subgraph is in.
        subgraph (nx.Graph): the subgraph to add node features to.
        radius (float): the radius of the subgraph.
        pos (tuple[float, float]): the position to center the subgraph around.
        tpos (tuple[float, float]): the target position.
    
    Raises:
        ValueError: if the model version is not supported.
    
    Returns:
        A graph with node features added.
    """
    if model_version == "v1":
        return add_node_features_v1(subgraph, pos, tpos)
    if model_version == "v2":
        return add_node_features_v2(graph, subgraph, radius, pos, tpos)
    raise ValueError(f"Model version {model_version} is not supported.")


def add_node_features_v1(
    graph: nx.Graph,
    pos: T.Tuple[float, float],
    tpos: T.Tuple[float, float],
) -> nx.Graph:
    """Add node features to a graph for model version 1.

    Args:
        graph (nx.Graph): the graph to add node features to.
        pos (tuple[float, float]): the position to center the subgraph around.
        tpos (tuple[float, float]): the target position.

    Returns:
        A graph with node features added.
    """
    rx, ry = pos
    tx, ty = tpos
    for node, data in graph.nodes(data=True):
        # Node position
        x, y = data["pos"]

        # Add position features
        graph.nodes[node]["x"] = x
        graph.nodes[node]["y"] = y
        graph.nodes[node]["rx"] = rx
        graph.nodes[node]["ry"] = ry
        graph.nodes[node]["tx"] = tx
        graph.nodes[node]["ty"] = ty

        # Add distance features
        graph.nodes[node]["rdist"] = np.sqrt((x - rx)**2 + (y - ry)**2)
        graph.nodes[node]["tdist"] = np.sqrt((x - tx)**2 + (y - ty)**2)

        # Add angle features
        nv = np.array([x, y])
        rv = np.array([rx, ry])

        # Get relative angle between robot and node
        rtheta = get_angle_between_vectors(rv, np.array([1, 0]))
        rangle = get_local_to_world_orientation(
            rx, ry,
            rtheta,
            x, y,
        )
        # Get relative angle between node and target where node orientation
        # is assumed to be in direction of world orientation
        ntheta = get_angle_between_vectors(nv, np.array([1, 0]))
        tangle = get_local_to_world_orientation(
            x, y,
            ntheta,
            tx, ty,
        )
        graph.nodes[node]["rangle"] = rangle         
        graph.nodes[node]["tangle"] = tangle

        # Remove the position attribute
        del graph.nodes[node]["pos"]

    return graph


def add_node_features_v2(
    graph: nx.Graph,
    subgraph: nx.Graph,
    radius: float,
    pos: T.Tuple[float, float],
    tpos: T.Tuple[float, float],
) -> nx.Graph:
    """Add node features to a graph for model version 2.

    Args:
        graph (nx.Graph): the graph that the subgraph is in.
        subgraph (nx.Graph): the subgraph to add node features to.
        radius (float): the radius of the subgraph.
        pos (tuple[float, float]): the position to center the subgraph around.
        tpos (tuple[float, float]): the target position.

    Returns:
        A graph with node features added.
    """
    rx, ry = pos
    tx, ty = tpos

    node_pos = {node: data["pos"] for node, data in subgraph.nodes(data=True)}
    nodes_within_target = [node for node, pos in node_pos.items() if is_point_in_radius(pos[0], pos[1], tx, ty, radius)]

    node_thetas = {node: get_angle_between_vectors(np.array(data["pos"]), np.array([1, 0])) for node, data in subgraph.nodes(data=True)}
    node_closeness_centralities = nx.closeness_centrality(graph)
    node_btwn_centralities = nx.betweenness_centrality(graph)

    for node, data in subgraph.nodes(data=True):
        # Node position
        x, y = data["pos"]

        # Add centrality features
        subgraph.nodes[node]["closeness_centrality"] = node_closeness_centralities[node]
        subgraph.nodes[node]["betweenness_centrality"] = node_btwn_centralities[node]

        # Add position features
        subgraph.nodes[node]["x"] = x
        subgraph.nodes[node]["y"] = y
        subgraph.nodes[node]["rx"] = rx
        subgraph.nodes[node]["ry"] = ry

        # Add distance features
        subgraph.nodes[node]["rdist"] = np.sqrt((x - rx)**2 + (y - ry)**2)

        # Add angle features
        rv = np.array([rx, ry])
        # Get relative angle between robot and node
        rtheta = get_angle_between_vectors(rv, np.array([1, 0]))
        rangle = get_local_to_world_orientation(
            rx, ry,
            rtheta,
            x, y,
        )
        subgraph.nodes[node]["rangle"] = rangle

        # Find shortest path length to node within the target
        shortest_path_lengths = [
            (
                target,
                1000 if not nx.has_path(graph, node, target)
                else nx.astar_path_length(graph, node, target),
            ) 
            for target in nodes_within_target
        ]
        shortest_path_node = node if len(nodes_within_target) == 0 else min(shortest_path_lengths, key=lambda item: item[1])[0]

        # Check if a node is connected to the target
        if is_point_in_radius(x, y, tx, ty, radius):
            # Add target features compared to actual target
            subgraph.nodes[node]["tgt_connected"] = 1
            subgraph.nodes[node]["tdist"] = np.sqrt((x - tx)**2 + (y - ty)**2)
            subgraph.nodes[node]["tangle"] = get_local_to_world_orientation(
                x, y,
                node_thetas[node],
                tx, ty,
            )
            subgraph.nodes[node]["tx"] = tx
            subgraph.nodes[node]["ty"] = ty
        else:
            # Add target features based on node closest to another node within the target
            subgraph.nodes[node]["tgt_connected"] = 0
            ux, uy = graph.nodes[shortest_path_node]["pos"]
            subgraph.nodes[node]["tdist"] = np.sqrt((x - ux)**2 + (y - uy)**2)
            subgraph.nodes[node]["tangle"] = get_local_to_world_orientation(
                x, y,
                node_thetas[node],
                ux, uy,
            )
            subgraph.nodes[node]["tx"] = ux
            subgraph.nodes[node]["ty"] = uy

        # Remove the position attribute
        del subgraph.nodes[node]["pos"]

    return subgraph


def extract_subgraph(
    graph: nx.Graph,
    radius: float,
    pos: T.Tuple[float, float],
) -> nx.Graph:
    """Extract a subgraph around a given position.

    Args:
        graph (nx.Graph): the graph to extract the subgraph from.
        radius (float): the radius of the subgraph.
        pos (tuple[float, float]): the position to center the subgraph around.

    Returns:
        A subgraph of the input graph.
    """
    subgraph = nx.Graph()
    px, py = pos
    pv = np.array([px, py])
    for node, data in graph.nodes(data=True):
        x, y = data["pos"]
        nv = np.array([x, y])
        if np.linalg.norm(nv - pv) <= radius:
            subgraph.add_node(node, pos=(x, y))
    for u, v in graph.edges():
        if u in subgraph.nodes and v in subgraph.nodes:
            subgraph.add_edge(u, v)
            if "weight" in graph[u][v]:
                subgraph[u][v]["weight"] = graph[u][v]["weight"]
    return subgraph


def get_features(
    model_version: str,
    subgraph: nx.Graph,
) -> T.List[str]:
    """Get the node features for a given model version.

    Args:
        model_version (str): the version of the model to use.
        subgraph (nx.Graph): the subgraph to extract node features from.

    Returns:
        A list of node attributes.
    """
    if model_version == "v1":
        return NODE_FEATURES_V1
    if model_version == "v2":
        return NODE_FEATURES_V2
    raise ValueError(f"Model version {model_version} is not supported.")


def get_graph_edge_weights(
    graph: nx.Graph,
) -> T.List[float]:
    """Get the edge weights for a graph.

    Args:
        graph (nx.Graph): the graph to extract edge weights from.

    Returns:
        A tuple pair of the list of edge indices and corresponding list of 
        edge weights.
    """
    graph = graph.to_directed()
    edge_weights = []
    for _, _, data in graph.edges(data=True):
        edge_weights.append(data["weight"])
    return edge_weights


def predict_trajectory(
    model: TrajectoryRegressor,
    G: nx.Graph,
    radius: float,
    pos: T.Tuple[int, int],
    tpos: T.Tuple[int, int],
    model_version: str = "v1",
) -> T.Tuple[int, int]:
    """Predict the trajectory of a robot given a graph and a position.

    Args:
        model (TrajectoryRegressor): the model to predict the trajectory.
        G (nx.Graph): the graph to predict the trajectory on.
        radius (float): the radius of the subgraph.
        robot_node (int): the node index of the robot.
        pos (tuple[int, int]): the position to center the subgraph around.
        tpos (tuple[int, int]): the target position.
        
    Returns:
        A tuple of the predicted trajectory.
    """
    model.eval()
    edge_weights = nx.edge_betweenness_centrality(G)
    for (u, v), w in edge_weights.items():
        G[u][v]["weight"] = w
    subgraph = extract_subgraph(G, radius, pos)
    # If the subgraph is empty, return a random position, 
    if len(subgraph.nodes) == 0:
        dx, dy = tuple(np.random.uniform(-0.1, 0.1, 2).tolist())
        rx, ry = (pos[0] + dx, pos[1] + dy)
        rx = min(max(rx, 0), 1)
        ry = min(max(ry, 0), 1)
        return (rx, ry)
    subgraph = add_node_features(model_version, G, subgraph, radius, pos, tpos)
    group_node_attrs = get_features(model_version, subgraph)
    data = from_networkx(subgraph, group_node_attrs=group_node_attrs)
    data.edge_weight = torch.tensor(get_graph_edge_weights(subgraph)).float()
    out = model([data])
    out = out.squeeze(0)
    return (out[0].item(), out[1].item())