import os
import torch
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from constants_utils import Hyperparameters, mesh_type
from dataset_utils import GreenPINNDataset
from plot_utils import plot_multiple_points
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_visualiser():
    train_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_train.pt")


    num=3
    if len(test_data.f_meshes) < num:
        random_f_mesh_idx = random.choices(range(0, len(test_data.f_meshes)), k=num)
    elif len(test_data.f_meshes) == num:
        random_f_mesh_idx = range(0, num)
    else:
        random_f_mesh_idx = random.sample(range(0, len(test_data.f_meshes)), num)
    random_f_values_idx = [random.sample(range(0, len(test_data.f_values[idx])), 1)[0] for idx in random_f_mesh_idx]
    approx_values = []
    u_gt = []
    eval_points = []
    
    num_f_terms = train_data.num_f_terms
    sample_1 = train_data[slice(*train_data.u_data_addresses[num_f_terms[2]])]
    sample_2 = train_data[slice(*train_data.u_data_addresses[num_f_terms[3]])]
    sample_3 = train_data[slice(*train_data.u_data_addresses[num_f_terms[4]])]
    plot_multiple_points(points_list=[sample_1["crd"], sample_1["f_mesh"][0],
                                      sample_2["crd"], sample_2["f_mesh"][0],
                                      sample_3["crd"], sample_3["f_mesh"][0]],
                        values_list=[sample_1["u_vals"], sample_1["f_vals"][0],
                                     sample_2["u_vals"], sample_2["f_vals"][0],
                                     sample_3["u_vals"], sample_3["f_vals"][0],],
                        cmap_list=["viridis", "plasma",
                                   "viridis", "plasma",
                                   "viridis", "plasma",],
                        title_list=["Train Sample 1 u(x)", "Train Sample 1 f(x)",
                                    "Train Sample 2 u(x)", "Train Sample 2 f(x)",
                                    "Train Sample 3 u(x)", "Train Sample 3 f(x)"],
                    axs_size=(3,2),
                    main_title="Train Data Visualisation",
                    figsize=(18, 10),
                    )

    test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt")
    num_f_terms = test_data.num_f_terms
    sample_1 = test_data[slice(*test_data.u_data_addresses[num_f_terms[2]])]
    sample_2 = test_data[slice(*test_data.u_data_addresses[num_f_terms[3]])]
    sample_3 = test_data[slice(*test_data.u_data_addresses[num_f_terms[4]])]
    plot_multiple_points(points_list=[sample_1["crd"], sample_1["f_mesh"][0],
                                      sample_2["crd"], sample_2["f_mesh"][0],
                                      sample_3["crd"], sample_3["f_mesh"][0]],
                        values_list=[sample_1["u_vals"], sample_1["f_vals"][0],
                                     sample_2["u_vals"], sample_2["f_vals"][0],
                                     sample_3["u_vals"], sample_3["f_vals"][0],],
                        cmap_list=["viridis", "plasma",
                                   "viridis", "plasma",
                                   "viridis", "plasma",],
                        title_list=["Test Sample 1 u(x)", "Test Sample 1 f(x)",
                                    "Test Sample 2 u(x)", "Test Sample 2 f(x)",
                                    "Test Sample 3 u(x)", "Test Sample 3 f(x)"],
                    axs_size=(3,2),
                    main_title="Test Data Visualisation",
                    figsize=(18, 10),
                    )

if __name__ == "__main__":
    data = {
        "data_vis": data_visualiser
    }

    tasks = {
        **data
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--rd', type=str, required=True, help='Which res folder to use.')
    # Which tests to run
    parser.add_argument('--all', action='store_true', help='Run all tests.') 
    parser.add_argument('--data', action='store_true', help='Run all data checker tests.') 
    parser.add_argument('--run', nargs='+', choices=tasks.keys(), help="Tasks to run. Choose from: " + ", ".join(tasks.keys()))
    args = parser.parse_args()

    main_dir = "./res/" + args.rd + "/"
    if not os.path.exists(main_dir):
        raise IsADirectoryError(f'The directory {main_dir} does not exist.')
    
    data_dir = main_dir + "data/"

    train_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_train.pt")
    test_data = GreenPINNDataset(data_file_path=data_dir, data_file_name="data_test.pt")
    domain = train_data.constants.domain


    universal_integration_mesh_size = train_data.constants.integration_mesh_sizes
    universal_integration_mesh_type: mesh_type = train_data.constants.integration_mesh_type
    universal_evaluation_mesh_size = train_data.constants.evaluation_mesh_sizes
    universal_evaluation_mesh_type: mesh_type = train_data.constants.evaluation_mesh_type

    if args.data:
        for func in tqdm(data.values(), "Running all data tests..."):
            func()
    
    if args.all:
        for key, func in tqdm(tasks.items(), "Running all tests..."):
            func()

    if args.run is not None:
        for task_key in args.run:
            tasks[task_key]()  # Call the corresponding function

