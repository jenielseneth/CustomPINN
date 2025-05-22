
from matplotlib import pyplot as plt
import torch

def plot_points_3d(points, values = None, cmap='viridis', title=""):
    if values is not None and type(values) == torch.Tensor:
        values = values.detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')        # Add one subplot to it
    scatter = ax.scatter(points[:, 0].detach().numpy(), points[:,1].detach().numpy(), points[:,2].detach().numpy(), c=values, cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

def plot_multiple_points_3d(points_list, values_list, title_list = None, cmap_list = None, main_title = None):
    fig = plt.figure() 
    fig.suptitle(main_title)
    for i, points in enumerate(points_list):
        ax = fig.add_subplot(3, 2, i+1, projection='3d')   
        ax.set_title(title_list[i] if title_list is not None else "")
        if values_list[i] is not None and type(values_list[i]) == torch.Tensor:
            values = values_list[i].detach().numpy()
        cmap = cmap_list[i] if cmap_list is not None else "viridis"
        scatter = ax.scatter(points[:, 0].detach().numpy(), points[:,1].detach().numpy(), points[:,2].detach().numpy(), c=values, cmap=cmap)
        plt.colorbar(scatter)
    plt.show()

    