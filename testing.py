from chebyshev import cheb_2d_impl, plot_points, sample_chebyshev_points_2
from data_generation import sample_mesh_points
from loss import CustomLoss
import torch
from pde_utils import get_u_evaluation_func, greens_function_poisson_eq_2d, test_source_term


domain = (-50,50,-50,50) 
# domain = (0,1,0,1)
eval_points = sample_mesh_points(domain[0], domain[1], domain[2], domain[3], 400)

#-----------------------------------------------------------------------

##Evaluating Chebyshev points implementation and testing loss functions
# loss_fn = CustomLoss()
# domain = (-5,5,-5,5)
# eval_points = sample_mesh_points(domain[0], domain[1], domain[2], domain[3], 400)
# cheb_points = sample_chebyshev_points_2(domain, (20, 20))
# values = test_source_term(cheb_points[:, :, 0],cheb_points[:, :, 1], cheb_points.shape[0:-1])
# eval_values = cheb_2d_impl(eval_points=eval_points, values=values, domain=domain)
# plot_points(cheb_points.view(400, 2), values.view(400))
# plot_points(eval_points, eval_values)
# greens_function = greens_function_poisson_eq_2d
# loss = loss_fn(greens_function_approx=greens_function, f_source_term=test_source_term, coordinates=eval_points, domain=domain, u=eval_values)
# print(loss)

#-----------------------------------------------------------------------

##Investigating the effect of psi and phi on the structure of u
eval_points = sample_mesh_points(domain[0], domain[1], domain[2], domain[3], 1000)
def custom_greens_function(x, y):
    def phi(x, y):
        val = torch.sum(x+y, -1)
        val = torch.abs((x*y).sum(-1))
        return val
    
    def psi(x, y):
        val =  torch.sum(x-y, -1)
        val = (x*y).sum(-1)
        return val
    
    return phi(x,y) * torch.log(torch.abs(x-y).sum(-1)) + psi(x,y)

values = torch.zeros(len(eval_points))
eval_fn = get_u_evaluation_func(custom_greens_function, False)
for i, point in enumerate(eval_points):
    values[i] = eval_fn(point, domain=domain)

plot_points(eval_points, values=values)