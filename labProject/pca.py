import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.widgets import Slider


def plot_explained_variance(data: np.ndarray, kindOfData: str):
    """
    Plot the cumulative explained variance ratio for all components.
    
    Parameters:
        data (np.ndarray): Input data to perform PCA on. Shape (n_samples, n_features).
    """
    pca = PCA()
    pca.fit(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Plot explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    # plt.title('Cumulative Explained Variance Ratio by Principal Components')
    plt.title('Cumulative Explained Variance Ratio by Principal Components for ' + kindOfData)
    plt.grid(True)
    plt.show()

def plot_explained_variance_some(data: list[np.ndarray], kindOfDatas: list[str]):
    """
    Plot the cumulative explained variance ratio for multiple datasets,
    and display variance and standard deviation for each dataset.

    Parameters:
        data (list[np.ndarray]): A list of datasets to perform PCA on.
        kindOfDatas (list[str]): A list of labels corresponding to each dataset.
    """
    if len(data) != len(kindOfDatas):
        raise ValueError("The number of datasets must match the number of labels.")
    
    plt.figure(figsize=(10, 6))
    
    for i, (dataset, label) in enumerate(zip(data, kindOfDatas)):
        # Compute variance and standard deviation
        variances = np.var(dataset, axis=0)  # Variance of each feature
        std_devs = np.std(dataset, axis=0)  # Std dev of each feature
        
        # Perform PCA on the dataset
        pca = PCA()
        pca.fit(dataset)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Plot cumulative explained variance for this dataset
        plt.plot(
            range(1, len(cumulative_variance_ratio) + 1), 
            cumulative_variance_ratio, 
            marker='o', 
            label=(
                f"{label}\n"
                f"Total Variance: {np.sum(variances):.2f}\n"
                f"Mean Std Dev: {np.mean(std_devs):.2f}"
            )
        )
    
    # Customize the plot
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Cumulative Explained Variance Ratio by Principal Components")
    plt.legend(loc='lower right', fontsize='small')  # Adjust legend position and size
    plt.grid(True)
    plt.show()


def plot_pca(data: np.ndarray,kindOfData: str, dimensions: int = 2):
    """
    Perform PCA on the given ndarray and plot the reduced data.

    Parameters:
        data (np.ndarray): Input data to perform PCA on. Shape (n_samples, n_features).
        dimensions (int): Number of dimensions to reduce to (2 or 3). Default is 2.

    Raises:
        ValueError: If dimensions is not 2 or 3.
    """
    if dimensions not in [2, 3]:
        raise ValueError("Dimensions argument must be 2 or 3.")
    
    # Perform PCA
    print(f"Performing PCA for {dimensions}D projection...")
    pca = PCA(n_components=dimensions)
    reduced_data = pca.fit_transform(data)

    # Plotting
    if dimensions == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', edgecolor='k', s=50)
        plt.title("2D PCA Projection for " + kindOfData)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)
        plt.show()
    elif dimensions == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c='blue', edgecolor='k', s=50)
        ax.set_title("3D PCA Projection for " + kindOfData)
        ax.set_zlabel("Principal Component 3")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        plt.show()

        

# def plot_pca_interactive(data: np.ndarray, dimensions: int = 2):
#     """
#     Perform PCA on the given ndarray and plot the reduced data interactively.

#     Parameters:
#         data (np.ndarray): Input data to perform PCA on. Shape (n_samples, n_features).
#         dimensions (int): Number of dimensions to reduce to (2 or 3). Default is 2.

#     Raises:
#         ValueError: If dimensions is not 2 or 3.
#     """
#     if dimensions not in [2, 3]:
#         raise ValueError("Dimensions argument must be 2 or 3.")
    
#     # Perform PCA
#     print(f"Performing PCA for {dimensions}D projection...")
#     # pca = PCA(n_components=dimensions)
#     pca = PCA()
#     reduced_data = pca.fit_transform(data)
    
#     # Interactive Plot
#     if dimensions == 2:
#         # Initial plot for the entire dataset
#         fig, ax = plt.subplots(figsize=(8, 6))
#         plt.subplots_adjust(bottom=0.25)  # Leave space for the slider
        
#         scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', edgecolor='k', s=50)
#         ax.set_title("2D PCA Projection")
#         ax.set_xlabel("Principal Component 1")
#         ax.set_ylabel("Principal Component 2")
#         ax.grid(True)
        
#         # Slider for time window
#         ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of the slider
#         slider = Slider(ax_slider, 'Time', 0, len(reduced_data)-1, valinit=0, valstep=1)

#         # Update function for the slider
#         def update(val):
#             time_index = int(slider.val)
#             scatter.set_offsets(reduced_data[:time_index, :])  # Update the data
#             fig.canvas.draw_idle()

#         slider.on_changed(update)
#         plt.show()
    
#     elif dimensions == 3:
#         from mpl_toolkits.mplot3d import Axes3D
#         # Initial 3D plot for the entire dataset
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c='blue', edgecolor='k', s=50)
#         ax.set_title("3D PCA Projection")
#         ax.set_xlabel("Principal Component 1")
#         ax.set_ylabel("Principal Component 2")
#         ax.set_zlabel("Principal Component 3")
        
#         # Slider for time window
#         ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of the slider
#         slider = Slider(ax_slider, 'Time', 0, len(reduced_data)-1, valinit=0, valstep=1)

#         # Update function for the slider
#         def update(val):
#             time_index = int(slider.val)
#             ax.cla()  # Clear and redraw for 3D
#             ax.scatter(reduced_data[:time_index, 0], reduced_data[:time_index, 1], reduced_data[:time_index, 2],
#                        c='blue', edgecolor='k', s=50)
#             ax.set_title("3D PCA Projection")
#             ax.set_xlabel("Principal Component 1")
#             ax.set_ylabel("Principal Component 2")
#             ax.set_zlabel("Principal Component 3")
#             fig.canvas.draw_idle()

#         slider.on_changed(update)
#         plt.show()

# # Example Usage
# if __name__ == "__main__":
#     # Generate some example data (100 samples, 5 features)
#     np.random.seed(42)
#     example_data = np.random.rand(100, 5)
    
#     # Call the function for 2D and 3D PCA
#     plot_pca_interactive(example_data, dimensions=2)
#     plot_pca_interactive(example_data, dimensions=3)



# # Example Usage
# if __name__ == "__main__":
#     # Generate some example data (100 samples, 5 features)
#     np.random.seed(42)
#     example_data = np.random.rand(100, 5)
    
#     # Call the function for 2D and 3D PCA
#     plot_pca(example_data, dimensions=2)
#     plot_pca(example_data, dimensions=3)

