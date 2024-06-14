import functools
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import typing as T

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np

from simulator import Simulator, draw_simulator, get_local_to_world_orientation
from utils import rad_to_deg, get_angle_between_vectors
from graph import get_nodes_in_radius, create_rgg


class TextLog:
    """Manages a running message log for a tkinter ScrolledText widget.

    Methods:
        log: prints a new message to the screen.
        clear: clears the current messages on the screen.
        start: logs a start message to the screen.
    """

    START_MSG: str = "START OF LOG"
    END_MSG: str = "END OF LOG"

    def __init__(self, widget: ScrolledText):
        """Inits a TextLog."""
        self._widget = widget
        self._log = []

    def log(self, msg: str):
        """Displays a new message to the log.

        Args:
            msg (str): the message to display.
        """
        self._log.append(msg)
        self._widget.configure(state="normal")
        self._widget.insert(tk.END, f"{msg}\n")
        self._widget.configure(state="disabled")
        self._widget.yview(tk.END)
        
    def clear(self):
        """Clears the current log contents."""
        self._log = []
        self._widget.configure(state="normal")
        self._widget.delete(1.0, tk.END)
        self._widget.configure(state="disabled")
        self._widget.yview(tk.END)

    def start(self):
        """Displays a new start message to the log."""
        def banner(s: str) -> str:
            return f"{s}\n{'='*len(s)}"
        start_msg = banner(self.START_MSG)
        self.log(start_msg)

    def end(self):
        """Displays a new end message to the log."""
        def banner(s: str) -> str:
            return f"{s}\n{'='*len(s)}"
        end_msg = banner(self.END_MSG)
        self.log(end_msg)

    def hline(self):
        """Displays a horizontal line across the log."""
        self.log("-"*60)


def start_gui(
    gui_title: str,
    sim: Simulator,
    show_labels: bool = False,
    show_orient: bool = False,
):
    """Start a GUI for the robot navigation simulation.

    Args:
        gui_title (str): the name of the GUI window.
        sim (Simulator): the robot navigation simulation.
        show_labels (bool): whether to display node labels in the graph.
        show_orient (bool): whether to display robot orientation information.
    """
    # Setup root gui window
    root = tk.Tk()
    root.title(gui_title)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Create main frame
    frame = ttk.Frame(root)
    
    # Timestep
    # Create data labels
    data_frame = ttk.Frame(frame)
    robot_pos_func = lambda s: f"Robot Position (m): ({s.rpos[0]:.2f}, {s.rpos[1]:.2f})"
    robot_pos_var = tk.StringVar(data_frame, value=robot_pos_func(sim))
    robot_pos_label = ttk.Label(data_frame, textvariable=robot_pos_var)
    robot_theta_func = lambda s: f"Robot Orientation (deg): {rad_to_deg(s.rtheta):.2f}"
    robot_theta_var = tk.StringVar(data_frame, value=robot_theta_func(sim))
    robot_theta_label = ttk.Label(data_frame, textvariable=robot_theta_var)
    sim_ts_var = tk.StringVar(data_frame, value=f"Timestep: {sim.ts}")
    sim_ts_label = ttk.Label(data_frame, textvariable=sim_ts_var)
    target_pos_var = tk.StringVar(data_frame, value=f"Target Position (m): ({sim.tpos[0]:.2f}, {sim.tpos[1]:.2f})")
    target_pos_label = ttk.Label(data_frame, textvariable=target_pos_var)
    target_dist_var = tk.StringVar(frame, value=f"Target not yet reached. Distance to target: {sim.tdist:.2f} (m)")
    target_dist_label = ttk.Label(frame, textvariable=target_dist_var)

    # Create graph canvas
    canvas = create_graph_canvas(frame)
    draw_canvas(
        sim,
        canvas,
        show_labels=show_labels,
        show_orient=show_orient,
    )

    # Create text log
    scrolledtext = ScrolledText(frame, width=60, height=30, state="disabled")
    log = TextLog(scrolledtext)
    log_init(sim, log)

    btn_frame = ttk.Frame(frame)

    # Create action buttons
    update_btn = ttk.Button(btn_frame, text="Update")
    play_btn = ttk.Button(btn_frame, text="Play")
    reset_btn = ttk.Button(btn_frame, text="Reset")

    # Setup update button
    update_btn_func = functools.partial(
        update_btn_callback,
        *(sim, canvas, robot_pos_var, robot_pos_func, robot_theta_var, robot_theta_func, sim_ts_var, target_pos_var, log, update_btn, play_btn, target_dist_var),
        show_labels=show_labels, show_orient=show_orient,
    )
    update_btn["command"] = update_btn_func

    # Setup play button
    play_btn_func = functools.partial(
        play_btn_callback,
        *(sim, canvas, robot_pos_var, robot_theta_var, sim_ts_var, target_dist_var, log, update_btn, play_btn, reset_btn),
        show_labels=show_labels, show_orient=show_orient,
    )
    play_btn["command"] = play_btn_func

    # Setup reset button
    reset_btn_func = functools.partial(
        reset_btn_callback,
        *(sim, canvas, robot_pos_var, robot_pos_func, robot_theta_var, robot_theta_func, sim_ts_var, target_pos_var, log, update_btn, play_btn, reset_btn, target_dist_var),
        show_labels=show_labels, show_orient=show_orient,
    )
    reset_btn["command"] = reset_btn_func

    # Grid layouts
    #           Col 1        Col 2
    #   Row 1 | textarea  | textarea  |
    #   Row 2 | canvas    | textarea  |
    #   Row 3 | actions   |           |
    
    # Main Frame
    frame.grid(
        column=0,
        row=0,
        sticky=(tk.N, tk.S, tk.E, tk.W),
    )
    data_frame.grid(
        row=1,
        column=1,
        sticky=(tk.N, tk.S, tk.E, tk.W),
        padx=15,
    )
    btn_frame.grid(
        row=3,
        column=1,
        sticky=(tk.N, tk.S, tk.E, tk.W),
        padx=15,
    )
    # Position Data
    robot_pos_label.grid(
        row=1,
        column=1,
        sticky=(tk.W,),
    )
    robot_theta_label.grid(
        row=1,
        column=2,
        padx=5,
        sticky=(tk.W,),
    )
    sim_ts_label.grid(
        row=1,
        column=3,
        padx=5,
        sticky=(tk.W,),
    )
    target_pos_label.grid(
        row=2,
        column=1,
        sticky=(tk.W,),
    )
    target_dist_label.grid(
        row=1,
        column=2,
        sticky=(tk.W, tk.N,),
        padx=5,
    )
    # Canvas
    canvas.get_tk_widget().grid(
        row=2,
        column=1,
        sticky=(tk.N, tk.W, tk.E, tk.S),
        padx=15,
        pady=15,
    )
    # Action Buttons
    reset_btn.grid(
        row=1,
        column=1,
        pady=5,
        sticky=(tk.W, tk.S),
    )
    play_btn.grid(
        row=1,
        column=2,
        padx=15,
        pady=5,
        sticky=(tk.E, tk.S),
    )
    update_btn.grid(
        row=1,
        column=3,
        pady=5,
        sticky=(tk.E, tk.S),
    )
    # Text Log
    scrolledtext.grid(
        row=2,
        column=2,
        padx=5,
        pady=15,
        sticky=(tk.W, tk.N,),
    )
    root.mainloop()


class Animator:
    """Manages the animation of the robot navigation simulation.

    Methods:
        play: starts the animation.
        set_rotations: sets the rotation frames for the animation.
        set_moves: sets the movement frames for the animation.
        animate_rotate: animates the rotation of the robot.
        animate_move: animates the movement of the robot.
        start_next_scene: starts the next scene in the simulation.
    """

    def __init__(self,
        sim: Simulator,
        canvas: FigureCanvasTkAgg,
        robot_pos_var: tk.StringVar,
        robot_theta_var: tk.StringVar,
        sim_ts_var: tk.StringVar,
        target_dist_var: tk.StringVar,
        log: TextLog,
        update_btn: ttk.Button,
        play_btn: ttk.Button,
        reset_btn: ttk.Button,
        show_labels: bool = False,
        show_orient: bool = False,
        num_frames: int = 50,
        ts_limit: int = 10,
    ):
        """Inits an Animator.

        Args:
            sim (Simulator): the robot navigation simulation.
            canvas (FigureCanvasTkAgg): the canvas to draw the graph in.
            robot_pos_var (tk.StringVar): the variable to update with the robot position.
            robot_theta_var (tk.StringVar): the variable to update with the robot orientation.
            sim_ts_var (tk.StringVar): the variable to update with the current timestep.
            target_dist_var (tk.StringVar): the variable to update with the distance to the target.
            log (TextLog): the log to record contextual messages to.
            update_btn (ttk.Button): the update button to toggle the state of.
            play_btn (ttk.Button): the play button to toggle the state of.
            reset_btn (ttk.Button): the reset button to toggle the state of.
            show_labels (bool): whether to display node labels in the graph.
            show_orient (bool): whether to display robot orientation information.
            num_frames (int): the number of frames to animate the robot movement.
            ts_limit (int): the maximum number of timesteps to simulate.
        """
        self.sim = sim
        self.canvas = canvas
        self.robot_pos_var = robot_pos_var
        self.robot_theta_var = robot_theta_var
        self.sim_ts_var = sim_ts_var
        self.target_dist_var = target_dist_var
        self.log = log
        self.update_btn = update_btn
        self.play_btn = play_btn
        self.reset_btn = reset_btn
        self.ax = get_canvas_axes(canvas)
        self.show_labels = show_labels
        self.show_orient = show_orient
        self.num_frames = num_frames
        self.ts_limit = ts_limit

        self.rotations = []
        self.rotate = FuncAnimation(
            canvas.figure,
            self.animate_rotate,
            init_func=self.set_rotations,
            frames=self.num_frames,
            interval=1,
            blit=False,
            repeat=True,
            repeat_delay=10,
        )

        self.moves = []
        self.move = FuncAnimation(
            canvas.figure,
            self.animate_move,
            init_func=self.set_moves,
            frames=self.num_frames,
            interval=1,
            blit=False,
            repeat=True,
            repeat_delay=10,
        )

    def play(self):
        """Plays the animation."""
        draw_canvas(
            self.sim,
            self.canvas,
            show_labels=self.show_labels,
            show_orient=self.show_orient,
        )
        self.rotate.resume()
        self.move.pause()
        self.update_btn["state"] = tk.DISABLED
        self.play_btn["state"] = tk.DISABLED
        self.reset_btn["state"] = tk.DISABLED
        
    def set_rotations(self):
        """Sets the rotation frames for the animation."""
        npose = self.sim.get_next_pose()
        rx, ry, rtheta = self.sim.rpose
        _, _, etheta = npose
        rthetas = np.linspace(rtheta, etheta, self.num_frames)
        self.rotations = [(rx, ry, rtheta) for rtheta in rthetas] + [(rx, ry, etheta)]
        
    def set_moves(self):
        """Set the movement frames for the animation."""
        npose = self.sim.get_next_pose()
        rx, ry, _ = self.sim.rpose
        ex, ey, etheta = npose
        rxs = np.linspace(rx, ex, self.num_frames)
        rys = np.linspace(ry, ey, self.num_frames)
        self.moves = [(rx, ry, etheta) for rx, ry in zip(rxs, rys)] + [(ex, ey, etheta)]
        
    def animate_rotate(self, i: int):
        """Animate the rotation of the robot.
        
        Args:
            i (int): the current frame index.
        """
        rpose = self.rotations[i]
        draw_simulator(
            self.sim,
            ax=self.ax,
            show_labels=self.show_labels,
            show_orient=self.show_orient,
            rpose=rpose,
        )
        self.robot_theta_var.set(f"Robot Orientation (deg): {rad_to_deg(rpose[2]):.2f}")
        if i >= self.num_frames - 1:
            self.rotate.pause()
            self.move.resume()
        
    def animate_move(self, i: int):
        """Animate the movement of the robot.
        
        Args:
            i (int): the current frame index.
        """
        rpose = self.moves[i]
        spos = self.moves[0][:2]
        epos = rpose[:2]
        draw_simulator(
            self.sim,
            ax=self.ax,
            show_labels=self.show_labels,
            show_orient=self.show_orient,
            rpose=rpose,
            cpath=(spos, epos),
        )
        self.robot_pos_var.set(f"Robot Position (m): ({rpose[0]:.2f}, {rpose[1]:.2f})")
        tdist = np.linalg.norm(np.array(self.sim.tpos) - np.array(rpose[:2]))
        self.target_dist_var.set(f"Target not yet reached. Distance to target: {tdist:.2f} (m)")
        if i >= self.num_frames - 1:
            self.move.pause()
            self.sim.update()
            draw_canvas(
                self.sim,
                self.canvas,
                show_labels=self.show_labels,
                show_orient=self.show_orient,
            )
            self.sim_ts_var.set(f"Timestep: {self.sim.ts}")
            log_update(self.sim, self.log)
            self.start_next_scene()

    def start_next_scene(self):
        """Start the next scene in the simulation."""
        if self.sim.has_reached_target:
            self.target_dist_var.set(f"Target reached! Distance to target: {self.sim.tdist:.2f} (m)")
        if self.sim.has_reached_target or self.sim.ts >= self.ts_limit:
            return
        self.play()


def play_btn_callback(
    sim: Simulator,
    canvas: FigureCanvasTkAgg,
    robot_pos_var: tk.StringVar,
    robot_theta_var: tk.StringVar,
    sim_ts_var: tk.StringVar,
    target_dist_var: tk.StringVar,
    log: TextLog,
    update_btn: ttk.Button,
    play_btn: ttk.Button,
    reset_btn: ttk.Button,
    show_labels: bool = False,
    show_orient: bool = False,
    num_frames: int = 100,
    ts_limit: int = 10,
):
    """Play the simulation until the target is reached or a certain timestep is passed.
    
    Args:
        sim (Simulator): the robot navigation simulation.
        canvas (FigureCanvasTkAgg): the canvas to draw the graph in.
        robot_pos_var (tk.StringVar): the variable to update with the robot position.
        robot_theta_var (tk.StringVar): the variable to update with the robot orientation.
        sim_ts_var (tk.StringVar): the variable to update with the current timestep.
        target_dist_var (tk.StringVar): the variable to update with the distance to the target.
        log (TextLog): the log to record contextual messages to.
        update_btn (ttk.Button): the update button to toggle the state of.
        play_btn (ttk.Button): the play button to toggle the state of.
        reset_btn (ttk.Button): the reset button to toggle the state of.
        show_labels (bool): whether to display node labels in the graph.
        show_orient (bool): whether to display robot orientation information.
        num_frames (int): the number of frames to animate the robot movement.
        ts_limit (int): the maximum number of timesteps to simulate.
    """
    animator = Animator(
        sim,
        canvas,
        robot_pos_var,
        robot_theta_var,
        sim_ts_var,
        target_dist_var,
        log,
        update_btn,
        play_btn,
        reset_btn,
        show_labels=show_labels,
        show_orient=show_orient,
        num_frames=num_frames,
        ts_limit=ts_limit,
    )
    animator.play()


def update_btn_callback(
    sim: Simulator,
    canvas: FigureCanvasTkAgg,
    robot_pos_var: tk.StringVar,
    robot_pos_func: T.Callable[[Simulator], str],
    robot_theta_var: tk.StringVar,
    robot_theta_func: T.Callable[[Simulator], str],
    sim_ts_var: tk.StringVar,
    target_pos_var: tk.StringVar,
    log: TextLog,
    update_btn: ttk.Button,
    play_btn: ttk.Button,
    target_dist_var: tk.StringVar,
    show_labels: bool = False,
    show_orient: bool = False,
):
    """Update the simulation state and log messages.

    Args:
        sim (Simulator): the robot navigation simulation.
        canvas (FigureCanvasTkAgg): the canvas to draw the graph in.
        robot_pos_var (tk.StringVar): the variable to update with the robot position.
        robot_pos_func (Callable[[Simulator], str]): the function to update the robot position.
        robot_theta_var (tk.StringVar): the variable to update with the robot orientation.
        robot_theta_func (Callable[[Simulator], str]): the function to update the robot orientation.
        sim_ts_var (tk.StringVar): the variable to update with the current timestep.
        target_pos_var (tk.StringVar): the variable to update with the target position.
        log (TextLog): the log to record contextual messages to.
        update_btn (ttk.Button): the update button to toggle the state of.
        play_btn (ttk.Button): the play button to toggle the state of.
        target_dist_var (tk.StringVar): the variable to update with the distance to the target.
        show_labels (bool): whether to display node labels in the graph.
        show_orient (bool): whether to display robot orientation information.
    """
    sim.update()
    draw_canvas(sim, canvas, show_labels=show_labels, show_orient=show_orient)
    robot_pos_var.set(robot_pos_func(sim))
    robot_theta_var.set(robot_theta_func(sim))
    sim_ts_var.set(f"Timestep: {sim.ts}")
    target_pos_var.set(f"Target Position (m): ({sim.tpos[0]:.2f}, {sim.tpos[1]:.2f})")
    log_update(sim, log)
    if sim.has_reached_target:
        target_dist_var.set(f"Target reached! Distance to target: {sim.tdist:.2f} (m)")
        update_btn["state"] = tk.DISABLED
        play_btn["state"] = tk.DISABLED
    else:
        target_dist_var.set(f"Target not yet reached. Distance to target: {sim.tdist:.2f} (m)")


def reset_btn_callback(
    sim: Simulator,
    canvas: FigureCanvasTkAgg,
    robot_pos_var: tk.StringVar,
    robot_pos_func: T.Callable[[Simulator], str],
    robot_theta_var: tk.StringVar,
    robot_theta_func: T.Callable[[Simulator], str],
    sim_ts_var: tk.StringVar,
    target_pos_var: tk.StringVar,
    log: TextLog,
    update_btn: ttk.Button,
    play_btn: ttk.Button,
    reset_btn: ttk.Button,
    target_dist_var: tk.StringVar,
    show_labels: bool = False,
    show_orient: bool = False,
):
    """Reset the simulation state and log messages.

    Args:
        sim (Simulator): the robot navigation simulation.
        canvas (FigureCanvasTkAgg): the canvas to draw the graph in.
        robot_pos_var (tk.StringVar): the variable to update with the robot position.
        robot_pos_func (Callable[[Simulator], str]): the function to update the robot position.
        robot_theta_var (tk.StringVar): the variable to update with the robot orientation.
        robot_theta_func (Callable[[Simulator], str]): the function to update the robot orientation.
        sim_ts_var (tk.StringVar): the variable to update with the current timestep.
        target_pos_var (tk.StringVar): the variable to update with the target position.
        log (TextLog): the log to record contextual messages to.
        update_btn (ttk.Button): the update button to toggle the state of.
        play_btn (ttk.Button): the play button to toggle the state of.
        reset_btn (ttk.Button): the reset button to toggle the state of.
        target_dist_var (tk.StringVar): the variable to update with the distance to the target.
        show_labels (bool): whether to display node labels in the graph.
        show_orient (bool): whether to display robot orientation information.
    """
    G, radius = create_rgg(sim.N)
    sim.reset(G, radius)
    draw_canvas(sim, canvas, show_labels=show_labels, show_orient=show_orient)    
    robot_pos_var.set(robot_pos_func(sim))
    robot_theta_var.set(robot_theta_func(sim))
    sim_ts_var.set(f"Timestep: {sim.ts}")
    target_pos_var.set(f"Target Position (m): ({sim.tpos[0]:.2f}, {sim.tpos[1]:.2f})")
    target_dist_var.set(f"Target not yet reached. Distance to target: {sim.tdist:.2f} (m)")
    log_init(sim, log)
    update_btn["state"] = tk.NORMAL
    play_btn["state"] = tk.NORMAL
    reset_btn["state"] = tk.NORMAL


def create_graph_canvas(
    frame: ttk.Frame,
    figsize: T.Tuple[int, int] = (5, 5),
    dpi: int = 100,
) -> FigureCanvasTkAgg:
    """Create the canvas containing the robot navigation graph display.

    Args:
        frame (Frame): the parent frame of the canvas.
        figsize (tuple[int, int]): the size of the figure placed in the canvas.
        dpi (int): the dots per inch of the resulting figure.
    
    Returns:
        A FigureCanvasTkAgg instance.
    """
    fig = plt.Figure(figsize=figsize, dpi=dpi)
    return FigureCanvasTkAgg(fig, frame)


def get_canvas_axes(canvas: FigureCanvasTkAgg) -> plt.Axes:
    """Get the axes of the canvas figure.

    Args:
        canvas (FigureCanvasTkAgg): the canvas to get the axes from.
    
    Returns:
        The axes of the canvas figure.
    """
    fig = canvas.figure
    axes = fig.get_axes()
    if len(axes) == 0:
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = axes[0]
    return ax


def draw_canvas(
    sim: Simulator,
    canvas: FigureCanvasTkAgg,
    show_labels: bool = False,
    show_orient: bool = False,
    rpose: T.Tuple[float, float, float] = None,
    cpath: T.Tuple[T.Tuple[float, float], T.Tuple[float, float]] = None,
):
    """Draws the latest graph contents in the canvas figure.

    NOTE: A robot pose and current path can be provided to animate the robot movement.

    Args:
        G (MultiAgentGraph): the robot navigation graph.
        canvas (FigureCanvasTkAgg): the canvas to draw the graph in.
        show_labels (bool): whether to display node labels in the graph.
        rpose (tuple[float, float, float]): the robot pose to draw.
        cpath (tuple[tuple[float, float], tuple[float, float]]): the path to draw.
    """
    fig = canvas.figure
    ax = get_canvas_axes(canvas)
    draw_simulator(
        sim,
        ax=ax,
        show_labels=show_labels,
        show_orient=show_orient,
        rpose=rpose,
        cpath=cpath,
    )
    fig.tight_layout()
    canvas.draw()


def log_init(
    sim: Simulator,
    log: TextLog,
):
    """Log the initial simulation state.

    Args:
        sim (Simulator): the robot navigation simulation.
        log (TextLog): the log to record contextual messages to.
    """
    log.clear()
    log.start()
    log.log(f"Navigation Strategy => {sim.nav_strategy_name}")
    if sim.nav_strategy_name == "model":
        log.log(f"Model Version => {sim.nav_strategy.model_version}")
    log_update(sim, log)


def log_update(
    sim: Simulator,
    log: TextLog,
):
    """Update the log with the latest simulation state.

    Args:
        sim (Simulator): the robot navigation simulation.
        log (TextLog): the log to record contextual messages to.
    """
    log_robot_data(sim, log)
    log_neighbors_data(sim, log)
    log.hline()
    if sim.has_reached_target:
        log.end()


def log_robot_data(
    sim: Simulator,
    log: TextLog,
):
    """Displays the positional and simulation information about the current
    robot state.

    Args:
        sim (Simulator): the robot navigation simulation.
        log (TextLog): the log to record contextual messages to.
    """
    log.log(f"Robot (ts = {sim.ts})")
    if sim.has_reached_target:
        log.log(f"Robot has reached target! Distance to target: {sim.tdist:.2f} (m)")

    log.log(f"({sim.rpos[0]:.2f}, {sim.rpos[1]:.2f}) (m), {rad_to_deg(sim.rtheta):.2f} (deg)")

    # Log ideal next position
    npos = sim.get_next_projected_pos()
    log.log(f"Next Ideal Position: ({npos[0]:.2f}, {npos[1]:.2f}) (m)")

    # Log differential to get to ideal next position
    vel, ang = sim.get_next_differential(npos)
    log.log(f"Next Differential: ({vel:.2f} (m/s), {rad_to_deg(ang):.2f} (deg/s))")

    # Log next position based on differential
    log.log(f"Next Actual Position: {sim.npos[0]:.2f}, {sim.npos[1]:.2f} (m)")



def log_neighbors_data(
    sim: Simulator,
    log: TextLog,
):
    """Displays positional and simulation information about the current robot
    neighbors.

    Args:
        sim (Simulator): the robot navigation simulation.
        log (TextLog): the log to record contextual messages to.
    """
    neighbor_nodes = get_nodes_in_radius(sim.graph, sim.radius, *sim.rpos)
    npoints = {
        node: pos
        for node, pos in sim.gpos.items()
        if node in neighbor_nodes
    }
    nangles = {
        node: get_angle_between_vectors(
            np.array([1, 0]),
            pos,
        )
        for node, pos in npoints.items()
    }
    nweights = {
        node: weight
        for node, weight in sim.node_weights.items()
        if node in neighbor_nodes
    }
    
    # Neighbor positional data
    ndists = {
        node: np.linalg.norm(np.array(sim.rpos) - np.array(npos))
        for node, npos in npoints.items()
    }
    nrelangles = {
        node: get_local_to_world_orientation(
            *sim.rpos,
            sim.rtheta,
            *npos,
        )
        for node, npos in npoints.items()
    }

    # Neighbor log messages
    neighbor_msgs = [
        (
            f"{node} - dist. ({ndists[node]:.2f} (m), rel ang. "
            f"{rad_to_deg(nrelangles[node]):.2f} (deg),\n({npos[0]:.2f}, {npos[1]:.2f}) (m), "
            f"{rad_to_deg(nangles[node]):.2f} (deg), w={nweights[node]:.2f}"
        )
        for node, npos in npoints.items()
    ]
    log.log(f"Neighbors (ts = {sim.ts})")
    log.log("\n".join(neighbor_msgs))
