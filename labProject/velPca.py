"""
pcaVel.py

This script loads wrist motion data from CONTROL and HFS conditions,
reshapes it for PCA, and logs key information at different levels.

Usage:
    python3 pcaVel.py                      # Default (WARNING level)
    python3 pcaVel.py --log-level info     # Info logs
    python3 pcaVel.py --log-level debug    # Debug logs

Log Levels:
    debug, info, warning, error, critical
"""

import logging
import argparse
from readData import *
import numpy as np
from utils.constants import Nodes, FileNames
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Change this to the files you want to analyze
CONTROL_FILE = FileNames.CONTROL_DAY1TRY1.value
HFS_FILE = FileNames.HFS_DAY1TRY1.value
CONTROL_FILES = [f.value for f in FileNames if f.name.startswith("CONTROL")]
HFS_FILES = [f.value for f in FileNames if f.name.startswith("HFS")]
WINDOW_SIZE = 1000
NUM_JOINTS = 11

def smooth_data(data: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """
    Smooths the data along the time axis using a moving average filter.

    Parameters:
        data (np.ndarray): Input array of shape (frames, features), e.g. (frames, 22).
        window_size (int): Size of the smoothing window.

    Returns:
        np.ndarray: Smoothed data with the same shape as the input.
    """
    assert data.ndim == 2, "Data must be 2D (frames x features)"
    assert window_size >= 1, "Window size must be >= 1"

    logging.info("Smoothing data with window size = %d", window_size)
    logging.debug("Original data shape: %s", data.shape)

    smoothed = np.empty(data.shape, dtype=np.float64)
    half_win = window_size // 2

    for i in range(data.shape[0]):
        start = max(0, i - half_win)
        end = min(data.shape[0], i + half_win + 1)
        smoothed[i] = np.mean(data[start:end], axis=0)

    return smoothed


def convert_velocity_to_speed(velocity_data: np.ndarray) -> np.ndarray:
    """
    Converts velocity vectors (Vx, Vy) for each joint to scalar speed.

    Parameters:
        velocity_data (np.ndarray): Array of shape (frames, 22) where each joint has (x, y) velocity.

    Returns:
        np.ndarray: Speed array of shape (frames, 11), one scalar speed per joint.
    """
    assert velocity_data.ndim == 2 and velocity_data.shape[1] == 22, "Expected velocity shape (frames, 22)"

    vx = velocity_data[:, 0::2]  # every even column: Vx
    vy = velocity_data[:, 1::2]  # every odd column: Vy

    speed = np.sqrt(vx**2 + vy**2)  # shape: (frames, 11)
    return speed


# function to analyze data and run PCA
def pcaPipeline():
    logging.info("Loading CONTROL data...")
    dset_names_control, locations_control, node_names_control = load_hdf5_data(CONTROL_FILE)
    locations_control = fill_missing(locations_control)
    logging.debug("location control shape: %s", locations_control.shape)

    logging.info("Loading HFS data...")
    dset_names_hfs, locations_hfs, node_names_hfs = load_hdf5_data(HFS_FILE)
    locations_hfs = fill_missing(locations_hfs)
    logging.debug("location hfs shape: %s", locations_hfs.shape)

    logging.info("Reshaping data for PCA...")
    # [x1, y1, x2, y2, ..., x11, y11]
    location_data = [
        locations_control.reshape(locations_control.shape[0], -1),
        locations_hfs.reshape(locations_hfs.shape[0], -1)
    ]
    logging.debug("new control location data shape: %s", location_data[0].shape)
    logging.debug("new hfs location data shape: %s", location_data[1].shape)

    logging.info("Smoothing data...")
    smooth_location_data_control = smooth_data(location_data[0])
    smooth_location_data_hfs = smooth_data(location_data[1])
    smooth_location_data = [smooth_location_data_control, smooth_location_data_hfs]
    logging.info("Smoothing complete.")
        
    # plot_before_after(location_data[0], smooth_location_data[0])
    
    # convert x, y to velocity
    logging.info("Calculating velocity...")
    velocity_data_control = np.gradient(smooth_location_data[0], axis=0) # todo: how do i know what is the time step?
    velocity_data_hfs = np.gradient(smooth_location_data[1], axis=0)
    velocity_data = [velocity_data_control, velocity_data_hfs]
    
    # plot_all_velocities(velocity_data_control)
    
    logging.info("Velocity calculation complete.")
    speed_data_control = convert_velocity_to_speed(velocity_data_control)
    speed_data_hfs = convert_velocity_to_speed(velocity_data_hfs)
    
    # plot_speed(speed_data_control, "Control")
    # plot_speed(speed_data_hfs, "HFS")
    
    pca_control = run_pca_and_plot_variance(speed_data_control, "Control")
    pca_hfs = run_pca_and_plot_variance(speed_data_hfs, "HFS")

####### Plotting functions #######
    
def plot_before_after(original, smoothed, joint_index=5, coord='x'):
    """
    Plots the original vs. smoothed data for a specific joint coordinate over time.
    
    Parameters:
        original (np.ndarray): Original data of shape (frames, 22).
        smoothed (np.ndarray): Smoothed data (same shape).
        joint_index (int): Which of the 11 joints to look at (0–10).
        coord (str): 'x' or 'y' coordinate to extract (x is 0, y is 1).
    """
    coord_offset = 0 if coord == 'x' else 1
    col = joint_index * 2 + coord_offset

    plt.figure(figsize=(10, 4))
    plt.plot(original[:, col], label="Original", alpha=0.5)
    plt.plot(smoothed[:, col], label="Smoothed window size: " + str(WINDOW_SIZE), linewidth=2)
    plt.title(f"Joint {joint_index} - {coord.upper()} Coordinate")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_all_velocities(velocity_data: np.ndarray):
    """
    Plots all 22 velocity components (x and y for 11 joints) in the same graph.

    Parameters:
        velocity_data (np.ndarray): Array of shape (frames, 22), containing velocity over time.
    """
    assert velocity_data.ndim == 2 and velocity_data.shape[1] == 22, "Expected shape (frames, 22)"

    plt.figure(figsize=(14, 6))
    for i in range(velocity_data.shape[1] // 2):
        x_col = i * 2
        y_col = i * 2 + 1
        plt.plot(velocity_data[:, x_col], label=f'Joint {i} - X', alpha=0.7)
        plt.plot(velocity_data[:, y_col], label=f'Joint {i} - Y', linestyle='--', alpha=0.7)

    plt.xlabel("Frame")
    plt.ylabel("Velocity")
    plt.title("Velocity of All Joints (X and Y)")
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_speed(speed_data: np.ndarray, kind: str):
    """
    Plots speed over time for all 11 joints.

    Parameters:
        speed_data (np.ndarray): Speed array of shape (frames, 11),
                                 where each column is the speed of one joint.
    """
    assert speed_data.ndim == 2 and speed_data.shape[1] == NUM_JOINTS, "Expected speed shape (frames, 11)"

    plt.figure(figsize=(14, 6))
    for joint_index in range(NUM_JOINTS):
        plt.plot(speed_data[:, joint_index], label=f"Joint {joint_index}", alpha=0.8)

    plt.xlabel("Frame")
    plt.ylabel("Speed (√(Vx² + Vy²))")
    plt.title("Speed of All Joints Over Time - " + kind)
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def run_pca_and_plot_variance(data: np.ndarray, title="Control", show_plot: bool=False):
    """
    Runs PCA on the data and plots the explained variance of each component.

    Parameters:
        data (np.ndarray): Input data of shape (frames, features), e.g., (frames, 11).
        title (str): Title for the plot.
    """
    assert data.ndim == 2, "Data must be 2D"
    
    pca = PCA(n_components=data.shape[1])  # up to 11 components
    pca.fit(data)

    explained_variance_ratio = pca.explained_variance_ratio_

    # Plotting
    if show_plot:
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, alpha=0.7)
        plt.plot(
            range(1, len(explained_variance_ratio)+1),
            np.cumsum(explained_variance_ratio),
            marker='o', linestyle='--', color='black',
            label='Cumulative Variance'
        )

        # Add value labels on bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title("Explained Variance by PCA: " + title)
        plt.xticks(range(1, len(explained_variance_ratio)+1))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return pca

def main():
    # run the pipeling on all the files.
    # modify the pipeline to return a value
    # Go through the tasks in one note
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA on motion data.")
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (default: warning)"
    )
    args = parser.parse_args()

    # Set logging level from user input
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    pcaPipeline()
