"""
readData.py

This module provides functions to load and preprocess data from HDF5 files.

Functions:
- load_hdf5_data: Loads dataset names, locations, and node names from an HDF5 file.
- print_details: Prints details about the loaded datasets, their shapes, and node names.
- fill_missing: Handles missing values (NaNs) in data arrays using interpolation.
"""
import h5py
import numpy as np
from scipy.interpolate import interp1d

def load_hdf5_data(filename: str) -> tuple[list[str], np.ndarray, list[str]]:
    """
    Load data from the HDF5 file.

    Parameters:
    filename (str): Path to the HDF5 file.

    Returns:
    tuple: Dataset names, locations data, and node names.
    """
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
    return dset_names, locations, node_names

def print_details(dset_names: list[str], locations: np.ndarray, node_names: list[str]) -> None:
    """
    Print details of HDF5 datasets, locations data shape, and nodes.

    Parameters:
    dset_names (list[str]): List of dataset names in the HDF5 file. 
    locations (np.ndarray): NumPy array containing location data. 
    node_names (list[str]): List of node names.

    Returns:
    None
    """
    print("=== HDF5 datasets ===")
    print(dset_names)
    print()
    print("=== Locations data shape ===")
    print(locations.shape)
    print()
    print("=== Nodes ===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()

def fill_missing(Y: np.ndarray, kind: str = "linear") -> np.ndarray:
    """
    Fills missing values (NaNs) in a multi-dimensional array using interpolation.

    Parameters:
    Y (np.ndarray): Input array with missing values (NaNs).
    kind (str): Interpolation type, e.g., 'linear', 'nearest', etc.

    Returns:
    np.ndarray: Array with missing values filled.
    """
    initial_shape = Y.shape
    Y = Y.reshape((initial_shape[0], -1))

    for i in range(Y.shape[-1]):
        y = Y[:, i]
        x = np.flatnonzero(~np.isnan(y))
        if len(x) > 0:
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)
            mask = np.isnan(y)
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        Y[:, i] = y

    return Y.reshape(initial_shape)
