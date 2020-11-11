import GPyOpt
import numpy as np
from matplotlib import pyplot as plt

# --- Objective function
objective_true  = GPyOpt.objective_examples.experiments2d.branin()                 # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = 0.1)         # noisy version
bounds = objective_noisy.bounds
objective_true.plot()

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]

def mycost(x):
    cost_f  = np.atleast_2d(.1*x[:,0]**2 +.1*x[:,1]**2).T
    cost_df = np.array([0.2*x[:,0],0.2*x[:,1]]).T
    return cost_f, cost_df

# plot the cost fucntion
grid = 400
bounds = objective_true.bounds
X1 = np.linspace(bounds[0][0], bounds[0][1], grid)
X2 = np.linspace(bounds[1][0], bounds[1][1], grid)
x1, x2 = np.meshgrid(X1, X2)
X = np.hstack((x1.reshape(grid*grid,1),x2.reshape(grid*grid,1)))

cost_X, _ = mycost(X)

# Feasible region
plt.contourf(X1, X2, cost_X.reshape(grid,grid), 10)
plt.title('Cost function')
plt.colorbar()

from numpy.random import seed
seed(123)
BO = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,
                                            domain = domain,
                                            initial_design_numdata = 5,
                                            acquisition_type = 'EI',
                                            normalize_Y = True,
                                            exact_feval = False,
                                            acquisition_jitter = 0.05)

BO_cost = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,
                                            cost_withGradients = mycost,
                                            initial_design_numdata =5,
                                            domain = domain,
                                            acquisition_type = 'EI',
                                            normalize_Y = True,
                                            exact_feval = False,
                                            acquisition_jitter = 0.05)

# BO.plot_acquisition()

# BO_cost.run_optimization(15)
# BO_cost.plot_acquisition()
#
# BO.run_optimization(15)
# BO.plot_acquisition()

plt.show()