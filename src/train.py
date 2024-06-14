import itertools as it
import time
import typing as T

import numpy as np
import networkx as nx
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from model import (
    TrajectoryRegressor,
    get_features,
    extract_subgraph,
    add_node_features,
    get_graph_edge_weights,
)
from graph import create_rgg


def create_trajectory_dataset(
    filename: str,
    graphs: T.List[T.Tuple[nx.Graph, float]],
    model_version: str = "v1",
    padding: bool = True,
) -> T.List[T.Tuple[T.List[Data], torch.Tensor]]:
    """Create a dataset of trajectory samples.

    Args:
        filename (str): the name of the CSV file containing the trajectory samples.
        graphs (List[Tuple[nx.Graph, float]]): the list of graphs to and their
            radii to sample from.
        model_version (str): the version of the model to use for training.
        padding (bool): whether to pad the trajectory samples to the maximum
            trajectory length.
    
    Returns:
        A list of trajectory samples with subgraphs and target positions.
    """
    df = read_trajectories_from_csv(filename)
    max_trajectory_len = df["ts_idx"].max() + 1
    num_trajectories = df["tid"].nunique()
    all_data = []
    for idx in range(num_trajectories):
        # Sample a random graph from the list
        graph, radius = graphs[np.random.choice(len(graphs))]
        edge_weights = nx.edge_betweenness_centrality(graph)
        for (u, v), w in edge_weights.items():
            graph[u][v]["weight"] = w

        # Get the trajectory and target positions
        trajectory_df = df[df["tid"] == idx]
        trajectory = [(x, y) for x, y in trajectory_df[["rx", "ry"]].values]
        tpos = trajectory[-1]
        trajectory.append(tpos)
        
        # Generate the subgraphs from the trajectory with node features
        tgts = []
        data = []
        for pos, npos in it.pairwise(trajectory):
            subgraph = extract_subgraph(graph, radius, pos)
            if len(subgraph.nodes) == 0:
                continue # Skip empty subgraphs
            subgraph = add_node_features(model_version, graph, subgraph, radius, pos, tpos)
            group_node_attrs = get_features(model_version, subgraph)
            d = from_networkx(subgraph, group_node_attrs=group_node_attrs)
            d.edge_weight = torch.tensor(get_graph_edge_weights(subgraph)).float()
            data.append(d)
            tgts.append(npos)
        
        # Handle case where graph overlay does not cover the entire trajectory at all
        # NOTE: This is a naive assumption but with large enough graphs it should be fine
        if len(data) == 0:
            continue

        # Pad the sequence of subgraphs based on the maximum trajectory length
        if padding:
            tdata = data[-1]
            for _ in range(len(data), max_trajectory_len):
                data.append(tdata)

        target = torch.zeros(len(data), 2) + torch.tensor(tpos)
        target[:len(tgts)] = torch.tensor(tgts)
        all_data.append((data, target.float()))
    return all_data


def generate_trajectories(
    num_samples: int,
    num_targets_per_graph: int = 10,
    num_graphs: int = 10,
    min_nodes: int = 10,
    max_nodes: int = 100,
    ts_interval: float = 1.0,
) -> pd.DataFrame:
    """Create a dataset of trajectory samples.
    
    NOTE: The dataset contains the following columns:
        - `tid` (int): identifier for a trajectory
        - `ts_idx` (int): index of the timestep in the trajectory
        - `ts` (float): timestamp of the timestep
        - `rx` (float): x-coordinate of the robot
        - `ry` (float): y-coordinate of the robot
        - `tx` (float): x-coordinate of the target
        - `ty` (float): y-coordinate of the target

    Args:
        num_samples (int): number of trajectories to generate.
        num_targets_per_graph (int): number of target nodes for the trajectories
            to sample per graph.
        num_graphs (int): number of graphs to sample from.
        min_nodes (int): minimum number of nodes in a graph.
        max_nodes (int): maximum number of nodes in a graph.
        ts_interval (float): time interval between each timestep.

    Returns:
        A DataFrame containing the generated trajectory samples.
    """
    # Generate graphs with varying number of nodes
    num_nodes = np.random.randint(min_nodes, max_nodes+1, num_graphs)
    graphs: T.List[nx.Graph] = []
    for n in num_nodes:
        rgg, _ = create_rgg(n)
        graphs.append(rgg)

    # Sample trajectories from the graphs
    trajectories_df = pd.DataFrame()
    trajectories_per_graph = max(num_samples // num_graphs, 1)
    trajectories_per_target = max(trajectories_per_graph // num_targets_per_graph, 1)
    tid = 0
    for graph in graphs:
        target_nodes = np.random.choice(graph.nodes, num_targets_per_graph, replace=True)
        for target_node in target_nodes:
            # Choose random start nodes
            start_nodes = np.random.choice(graph.nodes, trajectories_per_target, replace=True)
            for start_node in start_nodes:
                # If a path does not exist, skip the trajectory 
                if not nx.has_path(graph, start_node, target_node):
                    continue

                # Find the shortest path between the start and target nodes
                path = nx.astar_path(graph, start_node, target_node)
            
                # Generate a trajectory from the path
                trajectory_df = generate_trajectory_from_path(graph, path, tid, ts_interval=ts_interval)

                # Append the trajectory to the dataset and increment trajectory id
                trajectories_df = pd.concat([trajectories_df, trajectory_df], ignore_index=True)
                tid += 1
    return trajectories_df


def generate_trajectory_from_path(
    graph: nx.Graph,
    path: T.List[int],
    tid: int,
    ts_interval: float = 1.0,
) -> pd.DataFrame:
    """Generate a trajectory from a given path in a graph.

    Args:
        graph (nx.Graph): the graph to generate the trajectory on.
        path (List[int]): the list of nodes in the path.
        tid (int): the identifier of the trajectory.
        ts_interval (float): the time interval between each timestep.

    Returns:
        A DataFrame containing the generated trajectory.
    """
    # Get the trajectory coordinates
    path_len = len(path)
    trajectory_coords = [graph.nodes[node]["pos"] for node in path]
    target_coords = [trajectory_coords[-1] for _ in range(path_len)]

    # Create the trajectory DataFrame
    data = {
        "tid": [tid for _ in range(path_len)],
        "ts_idx": list(range(path_len)),
        "ts": [ts_interval * i for i in range(path_len)],
        "rx": [x for x, _ in trajectory_coords],
        "ry": [y for _, y in trajectory_coords],
        "tx": [x for x, _ in target_coords],
        "ty": [y for _, y in target_coords],
    }
    return pd.DataFrame(data)


def read_trajectories_from_csv(
    filename: str,
) -> pd.DataFrame:
    """Read trajectory samples from a CSV file.

    Args:
        filename (str): the name of the CSV file containing the trajectory samples.

    Returns:
        A DataFrame containing the trajectory samples.
    """
    df = pd.read_csv(filename)
    df["tid"] = df["tid"].astype(int)
    df["ts_idx"] = df["ts_idx"].astype(int)
    df["ts"] = df["ts"].astype(float)
    df["rx"] = df["rx"].astype(float)
    df["ry"] = df["ry"].astype(float)
    df["tx"] = df["tx"].astype(float)
    df["ty"] = df["ty"].astype(float)
    return df


def train(
    model: TrajectoryRegressor,
    optimizer: optim.Optimizer,
    dataset: T.List[T.Tuple[T.List[Data], torch.Tensor]],
    criterion: nn.Module,
    target_tolerance_radius: float = 0.01,
) -> TrajectoryRegressor:
    """Run a single training epoch for the trajectory regressor model.

    Args:
        model (TrajectoryRegressor): the model to train.
        optimizer (optim.Optimizer): the optimizer to use for training.
        dataset (List[Tuple[List[Data], torch.Tensor]]): the training dataset.
        criterion (nn.Module): the loss function to use for training.
        target_tolerance_radius (float): the radius of the target tolerance for

    Returns:
        The trained model the average loss and accuracy over the training dataset.
    """
    model.train()
    avg_loss = 0
    successes = 0
    for input, tgt in dataset:
        optimizer.zero_grad()
        out = model(input)
        loss = criterion(out, tgt)
        avg_loss += loss.item()
        successes += get_success(out, tgt, target_tolerance_radius)
        loss.backward()
        optimizer.step()
    avg_loss /= len(dataset)
    accuracy = successes / len(dataset)
    return model, avg_loss, accuracy


def evaluate(
    model: TrajectoryRegressor,
    dataset: T.List[T.Tuple[T.List[Data], torch.Tensor]],
    criterion: nn.Module,
    target_tolerance_radius: float = 0.01,
) -> float:
    """Run a single evaluation epoch for the trajectory regressor model.

    Args:
        model (TrajectoryRegressor): the model to evaluate.
        dataset (List[Tuple[List[Data], torch.Tensor]]): the evaluation dataset.
        criterion (nn.Module): the loss function to use for evaluation.
        target_tolerance_radius (float): the radius of the target tolerance for 
            the test accuracy.

    Returns:
        The average loss and accuracy over the evaluation dataset.
    """
    model.eval()
    avg_loss = 0
    successes = 0
    with torch.no_grad():
        for input, tgt in dataset:
            out = model(input)
            loss = criterion(out, tgt)
            avg_loss += loss.item()
            successes += get_success(out, tgt, target_tolerance_radius)
    avg_loss /= len(dataset)
    accuracy = successes / len(dataset)
    return avg_loss, accuracy


def get_success(
    output: torch.Tensor,
    target: torch.Tensor,
    target_tolerance_radius: float = 0.01,
) -> int:
    """Check if the model successfully reached the final target.

    Args:
        output (torch.Tensor): the output of the model.
        target (torch.Tensor): the target values.

    Returns:
        1 if the model successfully reached the target, 0 otherwise.
    """
    dist = torch.norm(output - target, dim=1)
    within_tol = dist <= target_tolerance_radius
    return 1 if torch.all(within_tol).item() else 0
    

def train_model(
    model: TrajectoryRegressor,
    optimizer: optim.Optimizer,
    train_dataset: T.List[T.Tuple[T.List[Data], torch.Tensor]],
    test_dataset: T.List[T.Tuple[T.List[Data], torch.Tensor]],
    criterion: nn.Module,
    num_epochs: int,
    target_tolerance_radius: float = 0.01,
) -> TrajectoryRegressor:
    """Train the trajectory regressor model.
    
    Args:
        model (TrajectoryRegressor): the model to train.
        optimizer (optim.Optimizer): the optimizer to use for training.
        train_dataset (List[Tuple[List[Data], torch.Tensor]]): the training dataset.
        test_dataset (List[Tuple[List[Data], torch.Tensor]]): the test dataset.
        criterion (nn.Module): the loss function to use for training.
        num_epochs (int): the number of epochs to train the model for.
        target_tolerance_radius (float): the radius of the target tolerance for 
            the test accuracy.

    Returns:
        The best model based on the test accuracy.
    """   
    best_model = model
    best_test_accuracy = 0.0

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        start = time.time()
        model, train_loss, train_accuracy = train(
            model,
            optimizer,
            train_dataset,
            criterion,
            target_tolerance_radius,
        )
        test_loss, test_accuracy = evaluate(
            model,
            test_dataset,
            criterion,
            target_tolerance_radius,
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model = model
        print((
            f"Epoch: {epoch+1}/{num_epochs}, " +
            f"Train Loss: {train_loss:.4f}, " +
            f"Train Accuracy: {train_accuracy:.4f}, " +
            f"Test Loss: {test_loss:.4f}, " +
            f"Best Test Accuracy: {best_test_accuracy:.4f}, " +
            f"Time Elapsed: {time.time() - start:.2f}s"
        ))
    return best_model, train_losses, train_accuracies, test_losses, test_accuracies
