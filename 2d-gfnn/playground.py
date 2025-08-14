from data_generation_utils import multiple_f_meshes_generate_points

multiple_f_meshes_generate_points(
    domain=(0, 1, 0, 1),
    save_dir="data/",
    file_name="data_train.pt",
    log_file_name="train_params.json",
    num_f_terms=100,
    u_mesh_sizes=[(8, 8), (12, 12), (20, 20)],
    f_mesh_sizes=[(4,4), (6,6), (9,9), (13,13), (16,16), (18,18), (21,21)],
    u_mesh_type="chebyshev",
    f_mesh_type="chebyshev",
)