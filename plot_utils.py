
from matplotlib import pyplot as plt

def plot_points(points, values = None, cmap='viridis', title=""):
    if values is not None:
        values = values.detach().numpy()
    scatter = plt.scatter(points[:, 0].detach().numpy(), points[:,1].detach().numpy(), c=values, cmap=cmap)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

def plot_multiple_points(points_list, values_list, title_list = None, cmap_list = None):
    for i, points in enumerate(points_list):
        plt.subplot(3, 2, i+1).set_title(title_list[i] if title_list is not None else "")
        if values_list[i] is not None:
            values = values_list[i].detach().numpy()
        cmap = cmap_list[i] if cmap_list is not None else "viridis"
        scatter = plt.scatter(points[:, 0].detach().numpy(), points[:,1].detach().numpy(), c=values, cmap=cmap)
        plt.colorbar(scatter)
    plt.show()

    