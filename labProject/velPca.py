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

# Change this to the files you want to analyze
CONTROL_FILE = FileNames.CONTROL_DAY1TRY1.value
HFS_FILE = FileNames.HFS_DAY1TRY1.value
WINDOW_SIZE = 200


def plot_before_after(original, smoothed, joint_index=0, coord='x'):
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


def smooth_data_in_place(data: np.ndarray, window_size: int = WINDOW_SIZE):
    """
    Smooths the data along the time axis using a moving average filter.

    Parameters:
        data (np.ndarray): The input array of shape (frames, features), e.g. (frames, 22).
        window_size (int): The size of the smoothing window. Default is WINDOW_SIZE const.

    This function modifies the data in-place.
    """
    assert data.ndim == 2, "Data must be 2D (frames x features)"
    assert window_size >= 1, "Window size must be >= 1"

    logging.info("Smoothing data in-place with window size = %d", window_size)
    logging.debug("Original data shape: %s", data.shape)

    smoothed = np.copy(data)  # Avoid modifying data during iteration
    half_win = window_size // 2

    for i in range(data.shape[0]):
        start = max(0, i - half_win)
        end = min(data.shape[0], i + half_win + 1)
        smoothed[i] = np.mean(data[start:end], axis=0)

    data[:] = smoothed  # Modify original data in-place
    logging.info("Smoothing complete.")

# The main of the file 
def pcaVel():
    logging.info("Loading CONTROL data...")
    dset_names_control, locations_control, node_names_control = load_hdf5_data(CONTROL_FILE)
    locations_control = fill_missing(locations_control)
    logging.debug("location control shape: %s", locations_control.shape)

    logging.info("Loading HFS data...")
    dset_names_hfs, locations_hfs, node_names_hfs = load_hdf5_data(HFS_FILE)
    locations_hfs = fill_missing(locations_hfs)
    logging.debug("location hfs shape: %s", locations_hfs.shape)

    logging.info("Reshaping data for PCA...")
    location_data = [
        locations_control.reshape(locations_control.shape[0], -1),
        locations_hfs.reshape(locations_hfs.shape[0], -1)
    ]
    logging.debug("new control location data shape: %s", location_data[0].shape)
    logging.debug("new hfs location data shape: %s", location_data[1].shape)

    logging.info("Smoothing data...")
    original_data = location_data[0].copy()

    smooth_data_in_place(location_data[0])
    smooth_data_in_place(location_data[1])
    
    plot_before_after(original_data, location_data[0], joint_index=4, coord='x')



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
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")

    pcaVel()
