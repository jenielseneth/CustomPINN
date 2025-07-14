import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (9, 9)

df = pd.read_csv("./csv/100fPINN_fem.csv")
grouped_means = pd.DataFrame({
    f'group_{i}': df.iloc[:, i]
    for i in range(1, df.shape[1], 3)
})
h_values = np.array([int(df.columns.values[i].split(":")[1].split("-")[0].strip().split(",")[0])*int(df.columns.values[i].split(":")[1].split("-")[0].strip().split(",")[1]) for i in range(1, df.shape[1], 3)])
error_values = np.array(grouped_means)[-1]
sort_inds = np.argsort(h_values)
h_values, error_values = h_values[sort_inds], error_values[sort_inds]


# df = pd.read_csv("./csv/100fpinn_besttrain2.csv")
# grouped_means = pd.DataFrame({
#     f'group_{i}': df.iloc[:, i]
#     for i in range(1, df.shape[1], 3)
# })
# pinn_h_values = np.array([int(df.columns.values[i].split(":")[1].split("-")[0].strip().split(",")[0])*int(df.columns.values[i].split(":")[1].split("-")[0].strip().split(",")[1]) for i in range(1, df.shape[1], 3)])
# pinn_error_values = np.array(grouped_means)[0] / 32400
# sort_inds = np.argsort(pinn_h_values)[1:-1]
# pinn_h_values, pinn_error_values = pinn_h_values[sort_inds], pinn_h_values[sort_inds]

# print(pinn_h_values, h_values)

# print(np.abs(pinn_error_values-error_values).sum()/(len(pinn_h_values)))


log_h = np.log(h_values)
log_E = np.log(error_values)

indices = slice(0, -1, 1)

print(h_values)
# Algebraic Convergence
slope, intercept, r_value, _, _ = linregress(log_h[indices], log_E[indices])
p = slope
t = np.linspace(1, 1000, 100)
func = t**slope * np.exp(intercept)
# plt.suptitle("PINN without a singularity term")
plt.title("Chebyshev Evaluation Mesh: Algebraic convergence p: " + str(p))
plt.plot(t, func)
plt.scatter(h_values, error_values)
plt.xlabel('Discretization Parameter h: Chebyshev node count per axis')
plt.ylabel('Error Rate E(h)')
plt.xscale("log")
plt.yscale("log")
plt.grid(which='major', color="#4D4D4D", linewidth=0.8)
plt.grid(which='minor', color="#BCBABA", linestyle=':', linewidth=1)
plt.show()

# Exponential Convergence | E_h = C * p ^ h
# log_h = np.log(h_values)
# log_E = np.log(error_values)
# slope, intercept, r_value, _, _ = linregress(h_values, log_E)
# p = np.exp(slope)

# t = np.linspace(1, 100, 100)
# func = np.exp(intercept) * p ** (-t) 

# func = - t * p + intercept
# print(p, func)
# plt.title("(18, 18) Chebyshev Evaluation Mesh: Exponential Convergence")
# plt.plot(t, func)
# plt.scatter(h_values, log_E)
# plt.xlabel('Discretization Parameter h: Chebyshev node count per axis')
# plt.ylabel('Error Rate E(h)')
# # plt.xscale("log")
# # plt.yscale("log")
# plt.grid(which='major', color="#4D4D4D", linewidth=0.8)
# plt.grid(which='minor', color="#BCBABA", linestyle=':', linewidth=1)
# plt.show()