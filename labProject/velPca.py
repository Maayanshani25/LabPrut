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
import os
import re


# Change this to the files you want to analyze
CONTROL_FILE = FileNames.CONTROL_DAY2TRY6.value
HFS_FILE = FileNames.HFS_DAY2TRY6.value
CONTROL_FILES = [f.value for f in FileNames if f.name.startswith("CONTROL")]
HFS_FILES = [f.value for f in FileNames if f.name.startswith("HFS")]
WINDOW_SIZE = 150
NUM_JOINTS = 11
DONT_SHOW_PLOTS = False 
SHOW_PLOTS = True
CONTROL_INDEX = 0
HFS_INDEX = 1

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

    logging.debug("Smoothing data with window size = %d", window_size)
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

def format_path_nicely(path: str) -> str:
    """
    Given a full file path like 'data/h5/.../dataControlDay2Try10.h5',
    returns a formatted string like 'Control: Day 2 Try 10'.
    """
    filename = os.path.splitext(os.path.basename(path))[0]  # 'dataControlDay2Try10'

    # Match: data + (Control or HFS) + Day + number + Try + number
    match = re.match(r"data(Control|HFS)Day(\d+)Try(\d+)", filename)
    if not match:
        return filename  # fallback

    group, day, trial = match.groups()
    return f"{group}: Day {day} Try {trial}"

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
    
def run_pca_and_plot_variance(data: np.ndarray, title: str, show_plot: bool=False) -> PCA:
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
    logging.info(title + ": First PCA component explains: %.1f%% of variance", explained_variance_ratio[0] * 100)

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


def run_pca_and_plot_variance_both(data_control: np.ndarray,data_hfs: np.ndarray, title_control: str, title_hfs:str, show_plot: bool=False):
    """
    Runs PCA on the data and plots the explained variance of each component.

    Parameters:
        data (np.ndarray): Input data of shape (frames, features), e.g., (frames, 11).
        title (str): Title for the plot.
    """
    assert data_control.ndim == 2, "Data must be 2D"
    assert data_hfs.ndim == 2, "Data must be 2D"
    
    pca_control = PCA(n_components=data_control.shape[1])  # up to 11 components
    pca_control.fit(data_control)
    pca_hfs = PCA(n_components=data_hfs.shape[1])  # up to 11 components
    pca_hfs.fit(data_hfs)
    

    explained_variance_ratio_control = pca_control.explained_variance_ratio_
    logging.info(title_control + ": First PCA component explains: %.1f%% of variance", explained_variance_ratio_control[0] * 100)
    explained_variance_ratio_hfs = pca_hfs.explained_variance_ratio_
    logging.info(title_hfs + ": First PCA component explains: %.1f%% of variance", explained_variance_ratio_hfs[0] * 100)

    # Plotting
    if show_plot:
        # control
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(1, len(explained_variance_ratio_control)+1), explained_variance_ratio_control, alpha=0.7)
        plt.plot(
            range(1, len(explained_variance_ratio_control)+1),
            np.cumsum(explained_variance_ratio_control),
            marker='o', linestyle='--', color='black',
            label='Cumulative Variance'
        )

        # Add value labels on bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

        #  hfs
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(1, len(explained_variance_ratio_hfs)+1), explained_variance_ratio_hfs, alpha=0.7)
        plt.plot(
            range(1, len(explained_variance_ratio_hfs)+1),
            np.cumsum(explained_variance_ratio_hfs),
            marker='o', linestyle='--', color='black',
            label='Cumulative Variance'
        )

        # Add value labels on bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

        
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title("Explained Variance by PCA: " + title_control)
        plt.xticks(range(1, len(explained_variance_ratio_control)+1))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_combined_pca_and_plot_variance(data_control: np.ndarray,
                                       data_hfs: np.ndarray,
                                       label_control: str = "Control",
                                       label_hfs: str = "HFS",
                                       show_plots: bool = False) -> tuple:
    """
    Runs PCA on both control and HFS datasets and plots their explained variance side-by-side.

    Parameters:
        data_control (np.ndarray): Control condition data of shape (frames, features).
        data_hfs (np.ndarray): HFS condition data of shape (frames, features).
        label_control (str): Label to use for the control group.
        label_hfs (str): Label to use for the HFS group.

    Returns:
        Tuple[PCA, PCA]: Fitted PCA models for control and HFS.
    """
    assert data_control.ndim == 2 and data_hfs.ndim == 2, "Both datasets must be 2D arrays."

    # Run PCA
    pca_control = PCA(n_components=data_control.shape[1])
    pca_control.fit(data_control)
    var_control = pca_control.explained_variance_ratio_

    pca_hfs = PCA(n_components=data_hfs.shape[1])
    pca_hfs.fit(data_hfs)
    var_hfs = pca_hfs.explained_variance_ratio_
    if show_plots:
        # Plot
        plt.figure(figsize=(10, 5))
        x = np.arange(1, len(var_control) + 1)
        
        plt.bar(x - 0.15, var_control, width=0.3, alpha=0.7, label=label_control, color='blue')
        plt.bar(x + 0.15, var_hfs, width=0.3, alpha=0.7, label=label_hfs, color='orange')

        plt.plot(x, np.cumsum(var_control), 'o--', color='blue', label=f'Cumulative {label_control}')
        plt.plot(x, np.cumsum(var_hfs), 'o--', color='orange', label=f'Cumulative {label_hfs}')

        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Explained Variance: Control vs HFS")
        plt.xticks(x)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return var_control, var_hfs


def plot_first_pca_distribution(control_vals, hfs_vals):
    """
    Creates a vertical 1D scatter plot of the first PCA component variance.
    Control = blue X, HFS = green O.
    """
    plt.figure(figsize=(2.5, 10))

    # Constant x-position so they all appear in a line
    x_control = [1] * len(control_vals)
    x_hfs = [2] * len(hfs_vals)

    # Plot control as blue X
    plt.scatter(x_control, control_vals, color='blue', marker='x', label='Control', s=150)

    # Plot HFS as green X
    plt.scatter(x_hfs, hfs_vals, color='green', marker='x', label='HFS', s=150)
    
    avg_control = np.mean(control_vals)
    avg_hfs = np.mean(hfs_vals)
    plt.scatter(1, avg_control, color='blue', marker='o', s=250, label='Avg Control')
    plt.scatter(2, avg_hfs, color='green', marker='o', s=250, label='Avg HFS')

    # Formatting
    plt.xticks([1, 2], ['Control', 'HFS'], fontsize=22)
    plt.ylabel('Explained Variance of 1st PCA (%)', fontsize=22)
    plt.title(f'Explained Variance Distribution of First PCA Component with smoothing window of {WINDOW_SIZE}', fontsize=22)
    plt.tick_params(axis='y', labelsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

# function to analyze data and run PCA
# todo: add hint for output_pca
def pcaPipeline(control_path: str, hfs_path: str, output_pca: list[list[PCA]], show_plots_bool: bool = False):
    control_name = format_path_nicely(control_path)
    hfs_name = format_path_nicely(hfs_path)
    
    logging.info("Loading CONTROL data...")
    _, locations_control, _ = load_hdf5_data(control_path)
    locations_control = fill_missing(locations_control)
    logging.debug("location control shape: %s", locations_control.shape)

    logging.debug("Loading HFS data...")
    _, locations_hfs, _ = load_hdf5_data(hfs_path)
    locations_hfs = fill_missing(locations_hfs)
    logging.debug("location hfs shape: %s", locations_hfs.shape)

    logging.debug("Reshaping data for PCA...")
    # [x1, y1, x2, y2, ..., x11, y11]
    location_data = [
        locations_control.reshape(locations_control.shape[0], -1),
        locations_hfs.reshape(locations_hfs.shape[0], -1)
    ]
    logging.debug("new control location data shape: %s", location_data[CONTROL_INDEX].shape)
    logging.debug("new hfs location data shape: %s", location_data[HFS_INDEX].shape)

    logging.debug("Smoothing data...")
    smooth_location_data = [None, None]
    smooth_location_data[CONTROL_INDEX] = smooth_data(location_data[CONTROL_INDEX])
    smooth_location_data[HFS_INDEX] = smooth_data(location_data[HFS_INDEX])
    logging.debug("Smoothing complete.")
    
    if show_plots_bool:
        plot_before_after(location_data[CONTROL_INDEX], smooth_location_data[HFS_INDEX])
    
    # convert x, y to velocity
    logging.debug("Calculating velocity...")
    velocity_data = [None, None]
    velocity_data[CONTROL_INDEX] = np.gradient(smooth_location_data[CONTROL_INDEX], axis=0) # todo: how do i know what is the time step?
    velocity_data[HFS_INDEX] = np.gradient(smooth_location_data[HFS_INDEX], axis=0)
    logging.debug("Velocity calculation complete.")
    
    if show_plots_bool:
        plot_all_velocities(velocity_data[CONTROL_INDEX])
    
    logging.debug("Converting velocity to speed...")
    speed_data_control = convert_velocity_to_speed(velocity_data[CONTROL_INDEX])
    speed_data_hfs = convert_velocity_to_speed(velocity_data[HFS_INDEX])
    
    if show_plots_bool:
        plot_speed(speed_data_control, "Control")
        plot_speed(speed_data_hfs, "HFS")
    
    logging.info("Running PCA...")
    pca_control = run_pca_and_plot_variance(speed_data_control, control_name, show_plots_bool)
    pca_hfs = run_pca_and_plot_variance(speed_data_hfs, hfs_name, show_plots_bool)
    
    var_control, var_hfs = run_combined_pca_and_plot_variance(speed_data_control, speed_data_hfs, control_name, hfs_name)
    output_pca[CONTROL_INDEX].append(var_control[0])
    output_pca[HFS_INDEX].append(var_hfs[0])


def main():
    # control_variances = [
    # 90.6, 80.6, 91.4, 83.3, 90.9, 90.7, 89.1, 90.1, 87.3, 85.7, 87.3, 81.8, 83.7
    # ]
    # hfs_variances = [
    #     88.5, 90.7, 92.6, 91.3, 83.7, 87.2, 91.5, 91.0, 92.0, 91.4, 93.3, 90.3, 93.7
    # ]
    # plot_first_pca_distribution(control_variances, hfs_variances)
    
    # run the pipeling on all the files.
    # modify the pipeline to return a value
    # Go through the tasks in one note
    output_pca = [[], []]
    print(CONTROL_FILES)
    print(HFS_FILES)
    for control_path, hfs_path in zip(CONTROL_FILES, HFS_FILES):
        control_name = format_path_nicely(control_path)
        hfs_name = format_path_nicely(hfs_path)
        logging.info(f"Running PCA for:\n {control_name}\n {hfs_name}")
        pcaPipeline(control_path, hfs_path, output_pca, DONT_SHOW_PLOTS)
        
    print("PCA results: {}", output_pca)
    plot_first_pca_distribution(output_pca[CONTROL_INDEX], output_pca[HFS_INDEX])

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
    
    main()
