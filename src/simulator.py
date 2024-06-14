from __future__ import annotations

import typing as T

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox
import networkx as nx
import numpy as np
import torch

from utils import get_local_to_world_orientation, rad_to_deg
from model import predict_trajectory
from graph import (
    get_node_positions,
    select_start_node,
    select_target_node,
    get_node_weights,
    get_nodes_in_radius,
    set_node_weights,
    is_point_in_radius,
)

class SimulatorStrategy:
    """A class to define a navigation strategy for the robot in the simulator.
    
    Methods:
        get_next_pos: get the next position of the robot based on the current
            simulation state.
    """

    def get_next_pos(self, sim: Simulator) -> T.Tuple[float, float]:
        """Get the next position of the robot based on the current simulation state.
        
        Args:
            sim (Simulator): the simulator instance.
        
        Raises:
            NotImplementedError: if the method is not implemented.

        Returns:
            A tuple of the next position.
        """
        raise NotImplementedError


class ShortestPathStrategy(SimulatorStrategy):
    """A class to define a shortest path navigation strategy for the robot in
    the simulator.
    
    Methods:
        get_next_pos: get the next position of the robot based on the shortest
            path strategy.
    """

    def get_next_pos(self, sim: Simulator) -> T.Tuple[float]:
        """Get the next position of the robot based on the shortest path strategy.

        Args:
            sim (Simulator): the simulator instance.

        Returns:
            A tuple of the next position.
        """
        neigbors = get_nodes_in_radius(
            sim.graph, sim.radius, *sim.rpos,
        )
        if len(neigbors) == 0:
            return 0.0, 0.0
        weights = {
            node: weight 
            for node, weight in sim.node_weights.items()
            if node in neigbors
        }
        max_node = max(weights.items(), key=lambda item: item[1])[0]
        next_pos = sim.gpos[max_node]
        return next_pos


class TrajectoryModelStrategy(SimulatorStrategy):
    """A class to define a trajectory model navigation strategy for the robot in
    the simulator.
    
    Methods:
        get_next_pos: get the next position of the robot based on the trajectory
            model strategy.
    """

    VALID_MODEL_VERSIONS = ("v1", "v2")

    def __init__(self, model_filename: str, model_version: str = "v1"):
        """Inits a TrajectoryModelStrategy instance.
        
        Args:
            model_filename (str): the filename of the trajectory model.
            model_version (str): the version of the trajectory model.
        
        Raises:
            ValueError: if the model version is invalid.
        
        Returns:
            A TrajectoryModelStrategy instance.
        """
        self.model = torch.load(model_filename)
        self.model_version = model_version
        if model_version not in self.VALID_MODEL_VERSIONS:
            raise ValueError(f"Invalid model version: {model_version}")

    def get_next_pos(self, sim: Simulator) -> T.Tuple[float]:
        """Get the next position of the robot based on the trajectory model strategy.

        Args:
            sim (Simulator): the simulator instance.

        Returns:
            A tuple of the next position.
        """
        next_pos = predict_trajectory(
            self.model,
            sim.graph,
            sim.radius,
            sim.rpos,
            sim.tpos,
            self.model_version,
        )
        return next_pos


def create_simulator_strategy(
    name: str,
    strategy_kwargs: T.Optional[T.Mapping[str, T.Any]] = None,
) -> SimulatorStrategy:
    """Create a simulator strategy instance based on the given name.
    
    Args:
        name (str): the name of the strategy.
        strategy_kwargs (Mapping[str, Any]): the keyword arguments for the
            strategy.
    
    Returns:
        A SimulatorStrategy instance.
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}

    if name == "algo":
        return ShortestPathStrategy(**strategy_kwargs)
    if name == "model":
        return TrajectoryModelStrategy(**strategy_kwargs)
    raise ValueError(f"Unknown simulator strategy: {name}")


class Simulator:
    """A class to simulate a robot navigating a graph to reach a target node."""

    def __init__(
        self,
        G: nx.Graph,
        radius: float,
        nav_strategy_name: str = "algo",
        nav_strategy_kwargs: T.Optional[T.Mapping[str, T.Any]] = None,
        rtheta_init: float = 0.0,
        tau: float = 1.0,
        ts: int = 0,
        target_tolerance_radius: float = 0.05,
    ):
        """Inits a Simulator instance.

        Args:
            G (nx.Graph): the robot navigation graph.
            radius (float): the radius of the graph and robot field of view.
            nav_strategy_name (str): the name of the navigation strategy.
            nav_strategy_kwargs (Mapping[str, Any]): the keyword arguments for
                the navigation strategy.
            rtheta_init (float): the initial robot orientation in radians.
            tau (float): the time delta between timesteps for the simulation.
            ts (int): the current timestep of the simulation.
            target_tolerance_radius (float): the radius around the target node
                that is considered to be reached.

        Returns:
            A Simulator instance.
        """
        self.N = len(G.nodes)

        # Set the navigation strategy
        self.nav_strategy_name = nav_strategy_name
        self.nav_strategy = create_simulator_strategy(
            nav_strategy_name,
            nav_strategy_kwargs,
        )

        # Initialize the simulator
        self._init(G, radius, rtheta_init, tau, ts, target_tolerance_radius)

    @property
    def gpos(self) -> T.Mapping[int, T.Tuple[float, float]]:
        """Positions of all nodes in the graph.
        
        Returns:
            A mapping of node indices to their positions.
        """
        return get_node_positions(self.graph)
    
    @property
    def rpos(self) -> T.Tuple[float, float]:
        """The current robot position.
        
        Returns:
            A tuple of the robot position.
        """
        return self.visited_path[-1][:2]
    
    @property
    def rtheta(self) -> float:
        """The current robot orientation.
        
        Returns:
            A float of the robot orientation.
        """
        return self.visited_path[-1][2]
    
    @property
    def rpose(self) -> T.Tuple[float, float, float]:
        """The current robot pose.
        
        Returns:
            A tuple of the robot pose (x, y, theta).
        """
        return self.visited_path[-1]

    @property
    def tpos(self) -> T.Tuple[float, float]:
        """The target position.
        
        Returns:
            A tuple of the target position.
        """
        return self.gpos[self.tnode]

    @property
    def npos(self) -> T.Tuple[float, float]:
        """The next robot position.
        
        Returns:
            A tuple of the next robot position.
        """
        return self.projected_path[-1][:2]

    @property
    def npose(self) -> T.Tuple[float, float, float]:
        """The next robot pose.
        
        Returns:
            A tuple of the next robot pose (x, y, theta).
        """
        return self.projected_path[-1]
    
    @property
    def has_reached_target(self) -> bool:
        """Indicates whether the robot has reached the target node within 
        a given tolerance radius.

        Returns:
            A boolean indicating whether the robot has reached the target.
        """
        return is_point_in_radius(
            *self.rpos,
            *self.tpos,
            self.trad,
        )
    
    @property
    def tdist(self) -> float:
        """The distance between the robot and the target.
        
        Returns:
            A float of the distance.
        """
        return np.linalg.norm(np.array(self.rpos) - np.array(self.tpos))

    def get_node_color(self, node: int) -> str:
        """Get the color of a node based on its type.

        Args:
            node (int): the node index.

        Returns:
            A string of the color.
        """
        if node == self.tnode:
            return "green"
        if node == self.rnode:
            return "blue"
        return "red"
    
    def get_next_projected_pos(self) -> T.Tuple[float, float]:
        """Get the next ideal position that the robot should be guided to
        based on the current navigation strategy.
        
        Returns:
            A tuple of the next position.
        """
        return self.nav_strategy.get_next_pos(self)
    
    def get_next_differential(self, npos: T.Tuple[float, float]) -> T.Tuple[float, float]:
        """Get the next differential control inputs for the robot based on the
        current navigation strategy.
        
        Args:
            npos (Tuple[float, float]): the next projected position.

        Returns:
            A tuple of the next differential control inputs.
        """
        return get_differential_between_points(
            *self.rpose,
            *npos,
            self.tau,
        )
    
    def get_next_pose(self) -> T.Tuple[float, float, float]:
        """Get the next pose of the robot based on the current navigation strategy.
        
        Returns:
            A tuple of the next pose.
        """
        npos = self.get_next_projected_pos()
        vel, ang = self.get_next_differential(npos)
        return get_robot_pose(*self.rpose, vel, ang, self.tau)

    def update(self):
        """Update the simulator state based on the current navigation strategy."""
        self.ts += 1

        # Update robot position
        rpose = self.get_next_pose()
        self.visited_path.append(rpose)

        # Update next projected position
        npose = self.get_next_pose()
        self.projected_path.append(npose)

    def reset(
        self,
        G: T.Optional[nx.Graph] = None,
        radius: T.Optional[float] = None,
    ):
        """Reset the simulator state to the initial state.
        
        NOTE: This method can be used to reset the simulator to a new graph and
        radius without changing the navigation strategy.

        Args:
            G (nx.Graph): the new graph to use for the simulation.
            radius (float): the new radius to use for the simulation.
        """
        if G is not None and radius is not None:
            self._init(G, radius, 0.0, self.tau, 0, self.trad)
        else:
            self._init(self.graph, self.radius, 0.0, self.tau, 0, self.trad)

    def _init(
        self,
        G: nx.Graph,
        radius: float,
        rtheta_init: float,
        tau: float,
        ts: int,
        target_tolerance_radius: float,
    ):
        """Initialize the simulator.
        
        Args:
            G (nx.Graph): the robot navigation graph.
            radius (float): the radius of the graph and robot field of view.
            rtheta_init (float): the initial robot orientation in radians.
            tau (float): the time delta between timesteps for the simulation.
            ts (int): the current timestep of the simulation.
            target_tolerance_radius (float): the radius around the target node

        """
        # Set the simulation parameters
        self.trad = target_tolerance_radius
        self.ts = ts
        self.tau = tau
        self.graph = G
        self.radius = radius
        self.tnode = select_target_node(self.graph)
        self.rnode = select_start_node(self.graph, self.tnode)
        
        # Track robot and target positions
        rpos = self.gpos[self.rnode]
        
        # Remove robot node from graph after getting its position
        self.graph.remove_node(self.rnode)
        self.node_weights = get_node_weights(self.graph, self.tnode, self.rnode)
        self.graph = set_node_weights(self.graph, self.node_weights)

        # Keep track of robot paths
        rpose = (*rpos, rtheta_init)
        self.visited_path = [rpose]
        npose = self.get_next_pose()
        self.projected_path = [npose]


def get_differential_between_points(
    rx: float,
    ry: float,
    rtheta: float,
    tx: float,
    ty: float,
    tau: float,
) -> T.Tuple[float, float]:
    """Get the linear and angular velocity required to move the robot from its
    current position to a target position in a given timestep.
    
    Args:
        rx (float): the robot x-coordinate.
        ry (float): the robot y-coordinate.
        rtheta (float): the robot orientation in radians.
        tx (float): the target x-coordinate.
        ty (float): the target y-coordinate.
        tau (float): the time step.
    
    Returns:
        A tuple of the linear and angular velocity.
    """
    ang = get_local_to_world_orientation(
        rx, ry,
        rtheta,
        tx, ty,
    )
    dv = np.array([tx - rx, ty - ry])
    vel = np.linalg.norm(dv)
    return vel / tau, ang / tau


def get_robot_pose(
    rx: float,
    ry: float,
    rtheta: float,
    vel: float,
    ang: float,
    tau: float,
) -> T.Tuple[float, float, float]:
    """Get the next robot pose given its current pose and control inputs.

    Args:
        rx (float): the robot x-coordinate.
        ry (float): the robot y-coordinate.
        rtheta (float): the robot orientation in radians.
        vel (float): the robot velocity.
        ang (float): the robot angular velocity.
        tau (float): the time step.
    
    Returns:
        A tuple of the next robot pose (x, y, theta).
    """
    rtheta = rtheta + tau*ang
    rtheta = rtheta % (2 * np.pi)
    rx = rx + tau*vel*np.cos(rtheta)
    rx = min(rx, 1.0)
    rx = max(rx, 0.0)
    ry = ry + tau*vel*np.sin(rtheta)
    ry = min(ry, 1.0)
    ry = max(ry, 0.0)
    return rx, ry, rtheta


def draw_simulator(
    sim: Simulator,
    ax: plt.Axes,
    show_labels: bool = False,
    show_orient: bool = False,
    rpose: T.Optional[T.Tuple[float, float, float]] = None,
    cpath: T.Optional[T.Tuple[T.Tuple[float, float], T.Tuple[float, float]]] = None,
):
    """Draw the simulator state in the given axes.
    
    NOTE: A custom robot pose and path can be provided to animate the movement
    of the robot.

    Args:
        sim (Simulator): the simulator to draw.
        ax (plt.Axes): the axes to draw the simulator in.
        show_labels (bool): whether to show node labels.
        show_orient (bool): whether to show the robot orientation.
        rpose (Tuple[float, float, float]): the robot pose to draw.
        cpath (Tuple[Tuple[float, float], Tuple[float, float]]): the current path
            of the robot to draw.
    """
    # Clear the current axes
    ax.clear()

    # Draw the graph
    draw_graph(sim, ax, show_labels)

    # Draw the robot
    draw_robot(sim, ax, rpose=rpose)
    
    # Draw the visited path
    draw_visited_path(sim, ax, cpath=cpath)
    
    # Draw the next differential
    if show_orient:
        draw_next_position(sim, ax)
        draw_next_differential(sim, ax, rpose=rpose)

    # Draw and set the axis limits
    ax.set_axis_on()
    ax.axvline(0.0, color="black", lw=0.5, linestyle="--")
    ax.axhline(0.0, color="black", lw=0.5, linestyle="--")
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])


def draw_graph(
    sim: Simulator,
    ax: plt.Axes,
    show_labels: bool = False,
):
    """Draw the graph in the given axes.
    
    Args:
        sim (Simulator): the simulator to draw.
        ax (plt.Axes): the axes to draw the graph in.
        show_labels (bool): whether to show node labels.
    """
    node_colors = [sim.get_node_color(node) for node in sim.gpos]
    nx.draw(
        sim.graph,
        pos=sim.gpos,
        ax=ax,
        node_size=50,
        node_color=node_colors,
        alpha=0.5,
        hide_ticks=False,
    )

    # Add a tolerance radius around the target
    rad_color = "green" if sim.has_reached_target else "yellow"
    tx, ty = sim.tpos
    circle = plt.Circle(
        (tx, ty),
        radius=sim.trad,
        color=rad_color,
        fill=False,
        linestyle="--",
    )
    ax.add_patch(circle)

    if show_labels:
        nodes_with_weights = list(sim.node_weights.keys())
        pos_with_weights = {
            node: [px, py + 0.05]
            for node, (px, py) in sim.gpos.items()
            if node in nodes_with_weights
        }
        node_labels = {
            node: f"({node}) {weight:.2f}"
            for node, weight in sim.node_weights.items()
        }
        nx.draw_networkx_labels(
            sim.graph,
            pos=pos_with_weights,
            ax=ax,
            labels=node_labels,
            hide_ticks=False,
        )


def draw_robot(
    sim: Simulator,
    ax: plt.Axes,
    rpose: T.Optional[T.Tuple[float, float, float]] = None,
):
    """Draw the robot in the given axes.

    NOTE: A custom robot pose can be provided to animate the movement
    of the robot.

    Args:
        sim (Simulator): the simulator to draw.
        ax (plt.Axes): the axes to draw the robot in.
        rpose (Tuple[float, float, float]): the robot pose to draw.
    """
    if rpose is None:
        rpose = sim.rpose

    # Draw the robot
    rx, ry, rtheta = rpose
    orient_rad = 0.05
    robot = plt.Circle(
        (rx, ry),
        radius=0.015,
        color="blue",
        fill=True,
        alpha=0.5,
    )
    rfov = plt.Circle(
        (rx, ry),
        radius=sim.radius,
        color="r",
        fill=False,
        linestyle="--",
    )
    ax.add_patch(robot)
    ax.add_patch(rfov)
    ax.arrow(
        rx, ry, orient_rad*np.cos(rtheta), orient_rad*np.sin(rtheta),
        head_width=0.01, head_length=0.01,
    )


def draw_visited_path(
    sim: Simulator,
    ax: plt.Axes,
    cpath: T.Optional[T.Tuple[T.Tuple[float, float], T.Tuple[float, float]]] = None,
):
    """Draw the visited path of the robot in the given axes.
    
    NOTE: A custom path can be provided to animate the movement of the robot.

    Args:
        sim (Simulator): the simulator to draw.
        ax (plt.Axes): the axes to draw the visited path in.
        cpath (Tuple[Tuple[float, float], Tuple[float, float]]): the current path
            of the robot to draw.
    """
    if cpath is not None:
        spos, epos = cpath
        px, py = spos
        cx, cy = epos
        ax.arrow(
            px, py,
            cx - px, cy - py,
            head_width=0.01, head_length=0.01,
            color="blue",
            linestyle="--",
            lw=0.5,
            alpha=0.5,
        )

    if len(sim.visited_path) <= 1:
        return

    for i in range(1, len(sim.visited_path)):
        px, py, _ = sim.visited_path[i-1]
        cx, cy, _ = sim.visited_path[i]
        ax.arrow(
            px, py,
            cx - px, cy - py,
            head_width=0.01, head_length=0.01,
            color="blue",
            linestyle="--",
            lw=0.5,
            alpha=0.5,
        )
        ax.plot(
            cx, cy, 
            "x",
            color="purple",
            markersize=6,
            lw=1.1,
        )


def draw_projected_path(
    sim: Simulator,
    ax: plt.Axes,
):
    """Draw the projected path of the robot in the given axes.

    Args:
        sim (Simulator): the simulator to draw.
        ax (plt.Axes): the axes to draw the projected path in.
    """
    for px, py, _ in sim.projected_path:
        ax.plot(
            px, py, 
            "x",
            color="purple",
            markersize=6,
            lw=1.1,
        )


def draw_next_position(
    sim: Simulator,
    ax: plt.Axes,
):
    """Draw the next position of the robot in the given axes.
    
    Args:
        sim (Simulator): the simulator to draw.
        ax (plt.Axes): the axes to draw the next position in.
    """
    npos = sim.get_next_projected_pos()
    ax.plot(
        *npos, 
        "o",
        color="orange",
        markersize=6,
        lw=1.1,
    )


def draw_next_differential(
    sim: Simulator,
    ax: plt.Axes,
    rpose: T.Optional[T.Tuple[float, float, float]] = None,  
):
    """Draw the components that determine the control inputs to guide
    the robot to the next position.
    
    NOTE: A custom robot pose can be provided to animate the movement
    of the robot.

    Args:
        sim (Simulator): the simulator to draw.
        ax (plt.Axes): the axes to draw the next differential in.
        rpose (Tuple[float, float, float]): the robot pose to draw.
    """
    if rpose is None:
        rpose = sim.rpose
    rx, ry, rtheta = rpose
    ux, uy = sim.get_next_projected_pos()
    _, ang = get_differential_between_points(
        rx, ry,
        rtheta,
        ux, uy,
        sim.tau,
    )
    dv = np.array([ux - rx, uy - ry])
    d = np.linalg.norm(dv)
    ax.arrow(
        rx, ry,
        d*np.cos(rtheta), d*np.sin(rtheta),
        head_width=0.01, head_length=0.01,
        color="orange",
        linestyle="--",
        alpha=0.5,
    )
    ax.arrow(
        rx, ry,
        ux - rx, uy - ry,
        head_width=0.01, head_length=0.01,
        color="orange",
        linestyle="--",
        alpha=0.5,
    )
    AngleAnnotation(
        (rx, ry),
        (rx + d*np.cos(rtheta), ry + d*np.sin(rtheta)),
        (ux, uy),
        ax=ax,
        size=35,
        text=f"{rad_to_deg(ang):.2f}" + u"\u00b0",
    )


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.

    REFERENCE: https://matplotlib.org/stable/gallery/text_labels_and_annotations/angle_annotation.html#angleannotation-class

    """
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])