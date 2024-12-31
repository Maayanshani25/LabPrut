"""
createPlots.py

This module provides functions to visualize the HDF5 dataset data.

Functions:
- plot_trajectory: Plot 2D trajectories of a node with a gradient representing time.
- interactive_plot_all: Create an interactive plot for all nodes using sliders.
- show_joint_graphs: Visualize a specific joint's trajectory and analysis.
- plot_density_heatmap: Plot density heatmap for node positions.
- pairwise_distance_heatmap: Plot pairwise distances between nodes as a heatmap.
- velocity_acceleration_analysis: Plot velocity and acceleration of a node over time.
- overlay_trajectories: Overlay trajectories of all nodes in a single plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from constants import Nodes


def plot_trajectory(McpOrTipsLoc: np.ndarray, title: str):
    """
    Plot 2D trajectory of McpOrTipsLoc with a gradient representing time.

    Parameters:
    McpOrTipsLoc (np.ndarray): McpOrTipsLoc location data.
    title (str): Title of the plot.
    """
    points = np.array([McpOrTipsLoc[:, 0, 0], McpOrTipsLoc[:, 1, 0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0, len(McpOrTipsLoc[:, 0, 0]) - 1)
    cmap = plt.cm.viridis
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(len(McpOrTipsLoc[:, 0, 0])))

    plt.figure(figsize=(7, 7))
    plt.gca().add_collection(lc)
    plt.title(f'2D Trajectory of {title} Node (Color = Time)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.colorbar(lc, label='Time (Frames)')
    plt.show()


def interactive_plot_all(locations: np.ndarray, frame_count: int, node_names: list[str]):
    """
    Create an interactive plot with sliders to control the time range, showing all nodes together.

    Parameters:
    locations (np.ndarray): Location data of all nodes.
    frame_count (int): Total number of frames.
    node_names (list[str]): Names of all nodes.
    """
    time_window = 100  # Default window size for the time range

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    scatter_plots = []
    colors = sns.color_palette("husl", len(node_names))

    for i, name in enumerate(node_names):
        sc = ax.scatter([], [], s=50, c=[colors[i]], alpha=0.6, label=name)
        scatter_plots.append(sc)

    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.legend()

    ax_time = plt.axes([0.2, 0.1, 0.65, 0.03])
    ax_window = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider_time = Slider(ax_time, 'Time', 0, frame_count - 1, valinit=0, valstep=1)
    slider_window = Slider(ax_window, 'Window', 1, 500, valinit=time_window, valstep=1)

    def update(val):
        current_time = int(slider_time.val)
        window_size = int(slider_window.val)

        start_time = max(0, current_time - window_size)
        end_time = min(frame_count, current_time + window_size)

        for i, sc in enumerate(scatter_plots):
            x_positions = locations[start_time:end_time, i, 0, 0]
            y_positions = locations[start_time:end_time, i, 1, 0]
            sc.set_offsets(np.c_[x_positions, y_positions])

        fig.canvas.draw_idle()

    slider_time.on_changed(update)
    slider_window.on_changed(update)
    plt.show()


def show_joint_graphs(node: Nodes, locations: np.ndarray, frame_count: int):
    """
    Visualize a specific joint's trajectory and create an interactive plot.

    Parameters:
    node (Nodes): Enum value for the specific joint.
    locations (np.ndarray): Location data for all nodes.
    frame_count (int): Total number of frames in the dataset.
    """
    index = node.value
    name = node.name
    node_loc = locations[:, index, :, :]

    # Plot trajectory
    plot_trajectory(node_loc, name)

    # Interactive plot
    interactive_plot_all(locations, frame_count, [name])


def plot_density_heatmap(node_loc: np.ndarray, title: str):
    """
    Plot a density heatmap of node positions.

    Parameters:
    node_loc (np.ndarray): Node location data.
    title (str): Title of the plot.
    """
    
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=node_loc[:, 0, 0], y=node_loc[:, 1, 0], cmap="Blues", fill=True, alpha=0.7)
    plt.title(f"Density Heatmap of {title} Node Positions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(0, 1024)
    plt.ylim(0, 1024)
    plt.show()

def plot_density_heatmap(node_loc: np.ndarray):
    """
    Plot a density heatmap of node positions.

    Parameters:
    node_loc (np.ndarray): Node location data.
    title (str): Title of the plot.
    """
    
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=node_loc[:, 0, 0], y=node_loc[:, 1, 0], cmap="Blues", fill=True, alpha=0.7)
    plt.title(f"Density Heatmap of {title} Node Positions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(0, 1024)   
    plt.ylim(0, 1024)
    plt.show()
    
def plot_density_heatmaps_all(locations: np.ndarray, node_names: list[str]):
    """
    Plot density heatmaps for all joints.

    Parameters:
    locations (np.ndarray): Locations data of all nodes.
    node_names (list[str]): Names of all nodes.
    """
    for i, name in enumerate(node_names):
        node_loc = locations[:, i, :, :]  # Extract location data for the current node
        plt.figure(figsize=(10, 8))
        sns.kdeplot(x=node_loc[:, 0, 0], y=node_loc[:, 1, 0], cmap="Blues", fill=True, alpha=0.7)
        plt.title(f"Density Heatmap of {name} Node Positions")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.xlim(0, 1024)
        plt.ylim(0, 1024)
        plt.show()



def pairwise_distance_heatmap(locations: np.ndarray, node_names: list[str]):
    """
    Plot a heatmap of average pairwise distances between nodes.

    Parameters:
    locations (np.ndarray): Locations data of all nodes.
    node_names (list[str]): Names of all nodes.
    """
    node_count = locations.shape[1]
    pairwise_distances = np.zeros((node_count, node_count))

    for i in range(node_count):
        for j in range(node_count):
            pairwise_distances[i, j] = np.mean(
                np.linalg.norm(locations[:, i, :, 0] - locations[:, j, :, 0], axis=1)
            )

    plt.figure(figsize=(10, 8))
    sns.heatmap(pairwise_distances, annot=True, xticklabels=node_names, yticklabels=node_names, cmap="coolwarm")
    plt.title("Pairwise Distance Heatmap Between Nodes")
    plt.show()


def velocity_acceleration_analysis(node_loc: np.ndarray, title: str):
    """
    Plot velocity and acceleration of a node over time.

    Parameters:
    node_loc (np.ndarray): Node location data.
    title (str): Title of the plot.
    """
    velocity = np.linalg.norm(np.diff(node_loc[:, :, 0], axis=0), axis=1)
    acceleration = np.diff(velocity)

    plt.figure(figsize=(10, 5))
    plt.plot(velocity, label="Velocity", color="blue")
    plt.title(f"Velocity of {title} Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Velocity")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(acceleration, label="Acceleration", color="orange")
    plt.title(f"Acceleration of {title} Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.show()


def overlay_trajectories(locations: np.ndarray, node_names: list[str]):
    """
    Overlay trajectories for all nodes.

    Parameters:
    locations (np.ndarray): Locations data of all nodes.
    node_names (list[str]): Names of all nodes.
    """
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(node_names):
        plt.plot(locations[:, i, 0, 0], locations[:, i, 1, 0], label=name)

    plt.title("Overlay of All Node Trajectories")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.xlim(0, 1024)
    plt.ylim(0, 1024)
    plt.show()
