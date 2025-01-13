'''
this it the translation of the matlab code of Chen. not workins so good but 
supposed to be the idea of the purpose of the task
'''


# # File paths
# file1: str = "/mnt/c/Users/maaya/Desktop/University/labPrut/data/h5/dataControlDay1Try2.h5"
# file2: str = "/mnt/c/Users/maaya/Desktop/University/labPrut/data/h5/dataHfsDay1Try2.h5"

# # Read the 'tracks' dataset
# with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
#     combined_tracks = f1['/tracks'][:].T  # Shape: (1, 2, 11, 19735)
#     combined_tracks2 = f2['/tracks'][:]

# # Extract the tip coordinates
# TIPS = [np.squeeze(combined_tracks[:, i, :]) for i in range(7, 12)]
# MCP = [np.squeeze(combined_tracks[:, i, :]) for i in range(2, 7)]
# TIPS_HFS = [np.squeeze(combined_tracks2[:, i, :]) for i in range(7, 12)]
# MCP_HFS = [np.squeeze(combined_tracks2[:, i, :]) for i in range(2, 7)]

# # Verify the shapes of TIPS
# for i, tip in enumerate(TIPS, start=1):
#     print(f"TIPS{i} shape: {tip.shape}")

# for i, mcp in enumerate(MCP, start=1):
#     print(f"MCP{i} shape: {mcp.shape}")

# # Calculate the distances between the tip pairs
# def calculate_distances(tips: List[np.ndarray]) -> List[np.ndarray]:
#     return [
#         np.sqrt(np.sum((tips[i] - tips[j]) ** 2, axis=1))
#         for i in range(len(tips))
#         for j in range(i + 1, len(tips))
#     ]

# distances_control: List[np.ndarray] = calculate_distances(TIPS)
# distances_HFS: List[np.ndarray] = calculate_distances(TIPS_HFS)

# # Calculate distances between tips and MCP
# distances_control_tips_mcp: List[np.ndarray] = [np.sqrt(np.sum((TIPS[i] - MCP[i]) ** 2, axis=1)) for i in range(5)]
# distances_HFS_tips_mcp: List[np.ndarray] = [np.sqrt(np.sum((TIPS_HFS[i] - MCP_HFS[i]) ** 2, axis=1)) for i in range(5)]

# # Compute means and standard deviations for each distance category
# means_control: np.ndarray = np.nanmean(distances_control, axis=1)
# means_HFS: np.ndarray = np.nanmean(distances_HFS, axis=1)
# means_control_tips_mcp: np.ndarray = np.nanmean(distances_control_tips_mcp, axis=1)
# means_HFS_tips_mcp: np.ndarray = np.nanmean(distances_HFS_tips_mcp, axis=1)

# std_control: np.ndarray = np.nanstd(distances_control, axis=1)
# std_HFS: np.ndarray = np.nanstd(distances_HFS, axis=1)
# std_control_tips_mcp: np.ndarray = np.nanstd(distances_control_tips_mcp, axis=1)
# std_HFS_tips_mcp: np.ndarray = np.nanstd(distances_HFS_tips_mcp, axis=1)

# # Combine means into matrices for plotting
# means_control_all: np.ndarray = np.concatenate([means_control, means_control_tips_mcp])
# means_HFS_all: np.ndarray = np.concatenate([means_HFS, means_HFS_tips_mcp])

# # Find the maximum value for y-axis limit
# max_value: float = max(np.nanmax(means_control_all), np.nanmax(means_HFS_all))

# # Plotting figures with increased font sizes
# def plot_bars_with_error(
#     x: np.ndarray,
#     means1: np.ndarray,
#     means2: np.ndarray,
#     std1: np.ndarray,
#     std2: np.ndarray,
#     labels: List[str],
#     title: str,
#     xlabel: str,
#     ylabel: str,
#     ylim: Tuple[float, float],
# ) -> None:
#     plt.bar(x - 0.15, means1, width=0.3, label='Control', color='b')
#     plt.bar(x + 0.15, means2, width=0.3, label='HFS', color='r')
#     plt.errorbar(x - 0.15, means1, yerr=std1, fmt='k.', linewidth=1.5)
#     plt.errorbar(x + 0.15, means2, yerr=std2, fmt='k.', linewidth=1.5)

#     plt.title(title, fontsize=20)
#     plt.xlabel(xlabel, fontsize=16)
#     plt.ylabel(ylabel, fontsize=16)
#     plt.legend(fontsize=14)
#     plt.xticks(x, labels, fontsize=12)
#     plt.ylim(ylim)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)

# x1 = np.arange(1, len(means_control) + 1)
# plot_bars_with_error(
#     x1,
#     means_control,
#     means_HFS,
#     std_control,
#     std_HFS,
#     ['1-2', '1-3', '1-4', '1-5', '2-3', '2-4', '2-5', '3-4', '3-5', '4-5'],
#     'Means of Distances Between Tips - Control vs. HFS',
#     'TIPS Pairs',
#     'Mean Distance',
#     (0, max_value + 1)
# )
# plt.show()

# x2 = np.arange(1, len(means_control_tips_mcp) + 1)
# plot_bars_with_error(
#     x2,
#     means_control_tips_mcp,
#     means_HFS_tips_mcp,
#     std_control_tips_mcp,
#     std_HFS_tips_mcp,
#     ['Tips1-MCP1', 'Tips2-MCP2', 'Tips3-MCP3', 'Tips4-MCP4', 'Tips5-MCP5'],
#     'Means of Distances Between TIPS and MCP - Control vs. HFS',
#     'TIPS - MCP Pairs',
#     'Mean Distance',
#     (0, max_value + 1)
# )
# plt.show()
