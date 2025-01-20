"""
main.py

This script serves as the entry point for analyzing and visualizing HDF5 dataset data.

Workflow:
1. Load data from the HDF5 file.
2. Print details about the data (datasets, shape, nodes).
3. Fill missing values in the data.
4. Perform data visualization:
    - Joint-specific interactive visualization.
    - Density heatmap.
    - Pairwise distance heatmap.
    - Velocity and acceleration analysis.
    - Overlay all node trajectories.
"""
import matplotlib
matplotlib.use('TkAgg')  # Or "Qt5Agg", "MacOSX", etc., depending on your environment

from readData import *  # Data loading and preprocessing
from createPlots import *  # Visualization functions
from utils.constants import Nodes, FileNames  # Enumerations
from pca import *  # PCA function

PCA_DIMENSIONS = 3


def process_all_files():
    filenames = [file.value for file in FileNames]
    

def main():
    # 1. File Path    
    filenameControl = FileNames.CONTROL_DAY1TRY1.value
    filenameHfs = FileNames.HFS_DAY1TRY1.value

     # ==========================
    # Load and Preprocess CONTROL Data
    # ==========================
    print("\n=== Processing CONTROL Data ===")
    dset_names_control, locations_control, node_names_control = load_hdf5_data(filenameControl)
    print_details(dset_names_control, locations_control, node_names_control)

    frame_count_control, node_count_control, _, instance_count_control = locations_control.shape
    locations_control = fill_missing(locations_control)

    # ==========================
    # Load and Preprocess HFS Data
    # ==========================
    print("\n=== Processing HFS Data ===")
    dset_names_hfs, locations_hfs, node_names_hfs = load_hdf5_data(filenameHfs)
    print_details(dset_names_hfs, locations_hfs, node_names_hfs)

    frame_count_hfs, node_count_hfs, _, instance_count_hfs = locations_hfs.shape
    locations_hfs = fill_missing(locations_hfs)

    # ==========================
    # Visualizations
    # ==========================
    nodeNames = [Nodes.WRIST, Nodes.MCP1, Nodes.MCP2, Nodes.MCP3, Nodes.MCP4, Nodes.MCP5, Nodes.TIPS1, Nodes.TIPS2, Nodes.TIPS3, Nodes.TIPS4, Nodes.TIPS5]
    # plot_density_heatmaps_all(locations_control, nodeNames)
    
    
    # ==========================
    # PCA 
    # ==========================
    # explain variance
    labels = ["Control", "HFS"]
    data = [locations_control[:, :, :, :].reshape(locations_control.shape[0], -1), locations_hfs[:, :, :, :].reshape(locations_hfs.shape[0], -1)]
    # plot_explained_variance_some(data, labels)
    dataBeginning = [locations_control[:100, :, :, :].reshape(100, -1), locations_hfs[:100, :, :, :].reshape(100, -1)]
    # print the shape of both datasets
    print(dataBeginning[0].shape)
    print(dataBeginning[0])
    print(dataBeginning[1].shape)
    labelsBeginning = ["Control Beginning", "HFS Beginning"]
    plot_explained_variance_some(dataBeginning, labelsBeginning)
    
    
    # Control Data
    # plot_explained_variance(locations_control[:, :, :, :].reshape(locations_control.shape[0], -1), "Control")
    plot_pca(locations_control[:, :, :, :].reshape(locations_control.shape[0], -1), "Control", dimensions=PCA_DIMENSIONS)
    plot_pca(dataBeginning[0], labelsBeginning[0], dimensions=PCA_DIMENSIONS)
    
    # HFS Data
    # plot_explained_variance(locations_hfs[:, :, :, :].reshape(locations_hfs.shape[0], -1), "HFS")
    plot_pca(locations_hfs[:, :, :, :].reshape(locations_hfs.shape[0], -1), "HFS", dimensions=PCA_DIMENSIONS)
    


    # # Joint-specific Interactive Visualization
    # print("\n=== Joint Graphs ===")
    # print("CONTROL:")
    # show_joint_graphs(Nodes.MCP1, locations_control, frame_count_control)
    # print("HFS:")
    # show_joint_graphs(Nodes.MCP1, locations_hfs, frame_count_hfs)

    # # Density Heatmaps
    # print("\n=== Density Heatmaps ===")
    # print("CONTROL:")
    # plot_density_heatmap(locations_control[:, Nodes.MCP1.value, :, :], "MCP1 (Control)")
    # print("HFS:")
    # plot_density_heatmap(locations_hfs[:, Nodes.MCP1.value, :, :], "MCP1 (HFS)")

    # # Pairwise Distance Heatmaps
    # print("\n=== Pairwise Distance Heatmaps ===")
    # print("CONTROL:")
    # pairwise_distance_heatmap(locations_control, node_names_control)
    # print("HFS:")
    # pairwise_distance_heatmap(locations_hfs, node_names_hfs)

    # # Velocity and Acceleration Analyses
    # print("\n=== Velocity and Acceleration ===")
    # print("CONTROL:")
    # velocity_acceleration_analysis(locations_control[:, Nodes.MCP1.value, :, :], "MCP1 (Control)")
    # print("HFS:")
    # velocity_acceleration_analysis(locations_hfs[:, Nodes.MCP1.value, :, :], "MCP1 (HFS)")

    # # Overlay Trajectories
    # print("\n=== Overlay Trajectories ===")
    # print("CONTROL:")
    # overlay_trajectories(locations_control, node_names_control)
    # print("HFS:")
    # overlay_trajectories(locations_hfs, node_names_hfs)

if __name__ == "__main__":
    main()
