
import os
from matplotlib import animation, pyplot as plt
import numpy as np
import torch

plt.rcParams["figure.figsize"] = (10, 10)

def plot_points(points, values = None, cmap='viridis', title="", save_dir = None, save_name = None):
    if values is not None:
        values = values.detach().numpy()
    scatter = plt.scatter(points[:, 0].detach().numpy(), points[:,1].detach().numpy(), c=values, cmap=cmap)
    plt.colorbar(scatter)
    plt.title(title)

    if save_dir is not None and save_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + save_name + ".png")

    plt.show()

        
def plot_multiple_points(points_list, values_list, title_list = None, cmap_list = None, axs_size: tuple = (3,2),
                        main_title = None, save_dir = None, save_name = None, show: bool = True, log_info: str = ""):
    '''
    Plots multiple meshes and their corresponding values.
    '''
    assert len(values_list) == len(points_list), "Points list must have same number of elements as corresponding values in values list."
    assert title_list is None or len(points_list) == len(title_list), "Must have a title for each plotted mesh in points_list."
    assert cmap_list is None or len(points_list) == len(cmap_list), "Must have a color map for each plotted mesh in points_list."
    fig, axs = plt.subplots(*axs_size)
    plt.suptitle(main_title)
    for i, points in enumerate(points_list):
        if title_list is not None and len(title_list) > i:
            axs[i//axs_size[1], i%axs_size[1]].set_title(title_list[i])
        else: 
            axs[i//axs_size[1], i%axs_size[1]].set_title("")
        if values_list[i] is not None and not type(values_list[i]) == list:
            values = values_list[i].detach().numpy()
        elif values_list[i] is not None and type(values_list[i]) == list:
            values = []
            for subvalues in values_list[i]:
                subvalues = subvalues.detach().numpy() if type(subvalues) == torch.Tensor else subvalues
                values.append(subvalues)
        if cmap_list is not None and len(cmap_list) > i:
            cmap = cmap_list[i] 
        else:
            cmap = "viridis"
        
        if type(points) == list:
            for i, sub_plot_points in enumerate(points):
                scatter = axs[i//axs_size[1], i%axs_size[1]].scatter(sub_plot_points[:, 0].detach().numpy(), sub_plot_points[:,1].detach().numpy(), c=values[i], cmap=cmap[i])

        else: 
            scatter = axs[i//axs_size[1], i%axs_size[1]].scatter(points[:, 0].detach().numpy(), points[:,1].detach().numpy(), c=values, cmap=cmap)
        fig.colorbar(scatter, ax=axs[i//axs_size[1], i%axs_size[1]])

    
    plt.text(0.2, 0.92, log_info,
            transform=fig.transFigure,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    
    if save_dir is not None and save_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + save_name + ".png")
        print("Saved figure as " + save_dir + save_name + ".png.")
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_rate(h, error, discrete_h_values = None, discrete_error_values = None, p: float = None):
    plt.title("Convergence Rate p = " + str(p) if p is not None else "")
    plt.plot(h, error)
    if discrete_h_values is not None and discrete_error_values is not None:
        plt.scatter(discrete_h_values, discrete_error_values)
    plt.xlabel('Discretization Parameter h')
    plt.ylabel('Error Rate E(h)')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which='major', color="#4D4D4D", linewidth=0.8)
    plt.grid(which='minor', color="#BCBABA", linestyle=':', linewidth=1)
    plt.show()