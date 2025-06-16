from chebyshev_utils import cheb_2d_impl, sample_chebyshev_points_2
from data_generation_utils import sample_random_mesh_points, sample_chebyshev_points_3


# domain = (-50,50,-50,50) 
domain = (0,1,0,1)
eval_points = sample_random_mesh_points(domain, 400)

#-----------------------------------------------------------------------

cheby_points = sample_chebyshev_points_3(domain, (10,10))
print("Chebyshev Points:\n", cheby_points)
cheby_points = cheby_points.reshape((10,10,2))
print("Chebyshev Points:\n", cheby_points)
