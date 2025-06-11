
import os
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (11, 11)

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

        
def plot_multiple_points(points_list, values_list, title_list = None, cmap_list = None, 
                        main_title = None, save_dir = None, save_name = None, show: bool = True):

    plt.suptitle(main_title)
    for i, points in enumerate(points_list):
        if len(title_list) > i and title_list is not None:
            plt.subplot(3, 2, i+1).set_title(title_list[i])
        else: 
            plt.subplot(3, 2, i+1).set_title("")
        if values_list[i] is not None:
            values = values_list[i].detach().numpy()
        if len(title_list) > i and cmap_list is not None:
            cmap = cmap_list[i] 
        else:
            cmap = "viridis"
        scatter = plt.scatter(points[:, 0].detach().numpy(), points[:,1].detach().numpy(), c=values, cmap=cmap)
        plt.colorbar(scatter)
    
    if save_dir is not None and save_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + save_name + ".png")
        print("Saved figure as " + save_dir + save_name + ".png.")
    if show:
        plt.show()
    else:
        plt.close()


    