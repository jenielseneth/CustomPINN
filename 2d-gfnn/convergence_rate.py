import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9, 9)


h_values = np.array([5, 10, 15, 20, 25, 27, 30])
# h_values = np.array([7, 9, 11, 13, 17, 21, 23, 27, 31, 33])
error_values_w_log_sing = np.array([0.16366271674633000,0.09297028183937070,0.06240420415997510,0.05639199540019040,0.05289212986826900,0.04619072005152700,0.038309987634420400])  #18 x 18
# error_values = np.array([0.8738434910774230,0.6467483043670650,0.617905855178833,0.4389096796512600,0.3173914849758150,0.20660443603992500,0.18931114673614500,0.2702561914920810,0.16382165253162400,0.29833027720451400])  / (20 * 20 * 10) #20 x 20
error_values = np.sqrt(np.array([0.243151949097713, 0.0888631236739457, 0.06100432590271030, 0.04741670299942290, 0.03954762592911720, 0.05657136680868760, 0.0321920953380565]))

print(np.mean(np.abs(error_values-error_values_w_log_sing)) / (18*18*10))
log_h = np.log(h_values)
log_E = np.log(error_values)
slope, intercept, r_value, _, _ = linregress(log_h, log_E)
p = slope

t = np.linspace(1, 100, 100)
func = t**slope * np.exp(intercept)
plt.suptitle("PINN without a singularity term")
plt.title("(18, 18) Chebyshev Evaluation Mesh: Convergence Rate p = " + str(p))
plt.plot(t, func)
plt.scatter(h_values, error_values)
plt.xlabel('Discretization Parameter h: Chebyshev node count per axis')
plt.ylabel('Error Rate E(h)')
plt.xscale("log")
plt.yscale("log")
plt.grid(which='major', color="#4D4D4D", linewidth=0.8)
plt.grid(which='minor', color="#BCBABA", linestyle=':', linewidth=1)
plt.show()