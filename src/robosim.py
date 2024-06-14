import click
import datetime
import matplotlib
import typing as T

import numpy as np
from torch.optim import AdamW
from torch import nn
import torch
from torch_geometric.explain import GNNExplainer, Explainer
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import pandas as pd

from simulator import Simulator, draw_simulator
from graph import create_rgg
from gui import start_gui
from train import generate_trajectories, create_trajectory_dataset, train_model
from model import TrajectoryRegressor, get_features, predict_trajectory



@click.group()
def cli():
    """CLI for robot navigation simulation program."""


@cli.command()
@click.option(
    "-N",
    "--num-nodes",
    default=10,
    type=int,
    help="Number of nodes to inclued in agent graph.",
)
@click.option(
    "--labels/--no-labels",
    default=True,
    help="Display node weights and labels.",
)
@click.option(
    "--orient/--no-orient",
    default=True,
    help="Display robot orientation information.",
)
@click.option(
    "-m",
    "--model-filename",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to trained model file.",
)
@click.option(
    "-v",
    "--model-version",
    default="v1",
    type=str,
    help="Version of model to use.",
)
def run(
    num_nodes: int,
    labels: bool,
    orient: bool,
    model_filename: T.Optional[str],
    model_version: str,
):
    """Command to run robot navigation simulation GUI."""
    matplotlib.use("TkAgg")

    nav_strategy_name = "algo"
    nav_strategy_kwargs = {}
    if model_filename is not None:
        nav_strategy_name = "model"
        nav_strategy_kwargs["model_filename"] = model_filename
        nav_strategy_kwargs["model_version"] = model_version

    G, radius = create_rgg(num_nodes)
    sim = Simulator(
        G=G,
        radius=radius,
        nav_strategy_name=nav_strategy_name,
        nav_strategy_kwargs=nav_strategy_kwargs,
        rtheta_init=0.0,
        tau=1.0,
    )
    start_gui(
        gui_title="ECE227 Final Project - Multi-Agent Cooperation Simulation",
        sim=sim,
        show_labels=labels,
        show_orient=orient,
    )


@cli.command()
@click.option(
    "-N",
    "--num-samples",
    default=1000,
    type=int,
    help="Number of trajectory samples to generate.",
)
@click.option(
    "-g",
    "--num-graphs",
    default=10,
    type=int,
    help="Number of graphs to sample from.",
)
@click.option(
    "--min-nodes",
    default=10,
    type=int,
    help="Minimum number of nodes in a graph.",
)
@click.option(
    "--max-nodes",
    default=100,
    type=int,
    help="Maximum number of nodes in a graph.",
)
@click.option(
    "-o",
    "--output-filename",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help="Name of file to write sample trajectories to.",
)
def gen_trajectories(
    num_samples: int,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    output_filename: T.Optional[str],
):
    """Command to generate trajectory samples."""
    if output_filename is None:
        # Create output filename with todays date in name using datetime
        today = datetime.date.today().strftime("%Y%m%d")
        output_filename = f"traj_samples_N{num_samples}_G{num_graphs}_{min_nodes}-{max_nodes}_{today}.csv"

    df = generate_trajectories(
        num_samples,
        num_graphs=num_graphs,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
    )
    df.to_csv(output_filename, index=False)


@cli.command()
@click.option(
    "-f",
    "--input-filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Name of file to read sample trajectories from.",
)
@click.option(
    "-o",
    "--output-filename",
    default="model.pth",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help="Name of file to write trained model to.",
)
@click.option(
    "-n",
    "--num-epochs",
    default=100,
    type=int,
    help="Number of epochs to train model.",
)
@click.option(
    "-l",
    "--learning-rate",
    default=0.001,
    type=float,
    help="Learning rate for optimizer.",
)
@click.option(
    "-g",
    "--num-graphs",
    default=1000,
    type=int,
    help="Number of graphs to sample from.",
)
@click.option(
    "--min-nodes",
    default=10,
    type=int,
    help="Minimum number of nodes in a graph.",
)
@click.option(
    "--max-nodes",
    default=100,
    type=int,
    help="Maximum number of nodes in a graph.",
)
@click.option(
    "-v",
    "--model-version",
    default="v1",
    type=str,
    help="Version of model to use.",
)
@click.option(
    "--padding/--no-padding",
    default=True,
    help="Whether or not to pad trajectories to max length.",
)
def train(
    input_filename: str,
    output_filename: str,
    num_epochs: int,
    learning_rate: float,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    model_version: str,
    padding: bool,
):
    """Command to train trajectory regressor model."""
    # Setup model and training parameters
    graphs = [
        create_rgg(np.random.randint(min_nodes, max_nodes))
        for _ in range(num_graphs)
    ]
    ds = create_trajectory_dataset(input_filename, graphs, model_version, padding)
    np.random.shuffle(ds)
    split_ratio = 0.8
    num_train_samples = int(split_ratio * len(ds))
    train_ds, test_ds = ds[:num_train_samples], ds[num_train_samples:]
    print(f"Training Data => {input_filename}")
    print(f"Model Version => {model_version}")
    print(f"Save Model to => {output_filename}")
    print(f"Number of training samples => {len(train_ds)}")
    print(f"Number of test samples => {len(test_ds)}")
    print(f"Max training trajectory size => {max((len(t) for t, _ in train_ds))}")
    print(f"Max test trajectory size => {max((len(t) for t, _ in test_ds))}")
    model = TrajectoryRegressor(
        gnn_hidden_dim=32,
        embedding_dim=16,
        rnn_hidden_dim=8,
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train model
    model, train_losses, train_accuracies, test_losses, test_accuracies = train_model(
        model,
        optimizer,
        train_ds,
        test_ds,
        criterion,
        num_epochs,
        target_tolerance_radius=0.05,
    )

    loss_df = pd.DataFrame(
        {"train_loss": train_losses, "test_loss": test_losses},
    )
    accuracy_df = pd.DataFrame(
        {"train_accuracy": train_accuracies, "test_accuracy": test_accuracies},
    )
    loss_df.to_csv(output_filename.replace(".pth", "_loss.csv"), index=False)
    accuracy_df.to_csv(output_filename.replace(".pth", "_accuracy.csv"), index=False)

    # Save model
    torch.save(model, output_filename)


@cli.command()
@click.option(
    "-f",
    "--input-filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Name of file to read sample trajectories from.",
)
@click.option(
    "-o",
    "--output-filename",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help="Name of file to write plots to.",
)
@click.option(
    "-m",
    "--model-filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to trained model file.",
)
@click.option(
    "-v",
    "--model-version",
    default="v1",
    type=str,
    help="Version of model to use.",
)
@click.option(
    "--padding/--no-padding",
    default=True,
    help="Whether or not to pad trajectories to max length.",
)
def explain(
    input_filename: str,
    output_filename: str,
    model_filename: str,
    model_version: str,
    padding: bool,
):
    """Command to explain model features."""
    import networkx as nx

    model: TrajectoryRegressor = torch.load(model_filename)
    explainer = Explainer(
        model=model.gnn,
        algorithm=GNNExplainer(epochs=30),
        explanation_type="model",
        node_mask_type="attributes",
        model_config={
            "mode": "regression",
            "task_level": "graph",
            "return_type": "raw",
        },
    )
    # Only sample from one graph for explanation
    graphs = [create_rgg(10)]
    ds = create_trajectory_dataset(input_filename, graphs, model_version, padding)
    traj, tgts = ds[0]
    sx, sy = traj[0].x[0, 2].item(), traj[0].x[0, 3].item()
    # Plot graph, plot trajectory on graph, and show feature importance for each trajectory
    G, radius = graphs[0]
    fig = plt.figure(figsize=(40, 20), dpi=100)
    gs = fig.add_gridspec(len(traj), 2)
    ax1 = fig.add_subplot(gs[:, 0])
    nx.draw(
        G,
        pos=nx.get_node_attributes(G, "pos"),
        ax=ax1,
        node_size=50,
        node_color="red",
        alpha=0.5,
        hide_ticks=False,
    )
    tx, ty = tgts[-1]
    tgt = plt.Circle(
        (tx, ty),
        radius=0.005,
        color="green",
        fill=True,
        alpha=0.5,
    )
    tgt_rad = plt.Circle(
        (tx, ty),
        radius=0.05,
        color="green",
        fill=False,
        linestyle="--",
    )
    for i in range(len(tgts)):
        px, py = sx, sy
        if i != 0:
            px, py = tgts[i-1]
        cx, cy = tgts[i]
        ax1.arrow(
            px, py,
            cx - px, cy - py,
            head_width=0.01, head_length=0.01,
            color="blue",
            linestyle="--",
            lw=0.5,
            alpha=0.5,
        )
        ax1.plot(
            cx, cy, 
            "x",
            color="purple",
            markersize=6,
            lw=1.1,
        )

    pred_trajs = [(sx, sy)]
    for _ in range(len(tgts)):
        x, y = predict_trajectory(
            model,
            G,
            radius,
            pred_trajs[-1],
            (tgts[-1][0].item(), tgts[-1][1].item()),
            model_version,
        )
        pred_trajs.append((x, y))

    for i in range(1, len(pred_trajs)):
        px, py = pred_trajs[i-1]
        cx, cy = pred_trajs[i]
        ax1.arrow(
            px, py,
            cx - px, cy - py,
            head_width=0.01, head_length=0.01,
            color="orange",
            linestyle="--",
            lw=0.5,
            alpha=0.5,
        )
        ax1.plot(
            cx, cy, 
            "x",
            color="blue",
            markersize=6,
            lw=1.1,
        )

    ax1.add_patch(tgt)
    ax1.add_patch(tgt_rad)
    ax1.set_axis_on()
    ax1.axvline(0.0, color="black", lw=0.5, linestyle="--")
    ax1.axhline(0.0, color="black", lw=0.5, linestyle="--")
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    
    features = get_features(model_version, traj[0])
    num_features = len(features)
    num_trajectories = len(traj)
    max_score = 0.0
    scores = torch.zeros((num_trajectories, num_features))
    for i, subgraph in enumerate(traj):
        x = subgraph.x.float()
        edge_index = subgraph.edge_index
        explanation = explainer(x, edge_index)
        feat_labels = get_features(model_version, subgraph)
        score = explanation.get("node_mask").sum(dim=0)
        sorted_indices = torch.argsort(score, descending=True)
        scores[i] = score
        max_score = max(max_score, score.max().item())

    for i, score in enumerate(scores):
        ax = fig.add_subplot(gs[i, 1])
        ax.barh(
            feat_labels,
            [score[i].item() for i in sorted_indices],
            label=[feat_labels[i] for i in sorted_indices],
            color="blue",
            alpha=0.5,
        )
        ax.set_xlim([0, max_score + 0.5])

    fig.tight_layout()
    plt.savefig(
        output_filename,
        format="png",
    )

 
if __name__ == "__main__":
    cli()