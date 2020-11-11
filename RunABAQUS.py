import os
import GPyOpt
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import seed  # fixed seed
seed(123456)

space = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 0.2)},
         {'name': 'var_2', 'type': 'continuous', 'domain': (0.1, 0.3)}]
constraints = [{'name': 'constr_1', 'constraint': '-(np.pi * x[:, 0] ** 2 + 0.8 * x[:, 1] - 0.25)'}]
feasible_region = GPyOpt.Design_space(space=space, constraints=constraints)
initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 3)

# Grid of points to make the plots
grid = 400
bounds = feasible_region.get_continuous_bounds()
X1 = np.linspace(bounds[0][0], bounds[0][1], grid)
X2 = np.linspace(bounds[1][0], bounds[1][1], grid)
x1, x2 = np.meshgrid(X1, X2)
X = np.hstack((x1.reshape(grid*grid,1),x2.reshape(grid*grid,1)))

# Check the points in the feasible region.
masked_ind = feasible_region.indicator_constraints(X).reshape(grid,grid)
masked_ind = np.ma.masked_where(masked_ind > 0.5, masked_ind)
masked_ind[1,1]=1

# Make the plots
plt.figure()

# Feasible region
plt.contourf(X1, X2, masked_ind ,100, cmap= plt.cm.bone, alpha=1,origin ='lower')

# These files must exist; provide your own paths
Rtxtpath = "C:\\temp2\\Rvalue.txt"
htxtpath = "C:\\temp2\\hvalue.txt"
outputpath = "C:\\temp2\\Displacement_of_tip.txt"

# Command line to run ABAQUS; provide your own file name
cmdl = 'cmd /c "cd c:/temp2 && abaqus cae noGUI=ABAQUS_GPYOPT_Test_Problem.py"'

def tipdisplacement(x):  # Function to optimize; x = (radius, height)
    open(Rtxtpath, "w").write(str(x[:, 0][0]))
    open(htxtpath, "w").write(str(x[:, 1][0]))
    os.system(cmdl)
    return float(open(outputpath, "r").read().strip())

# --- CHOOSE the objective
objective = GPyOpt.core.task.SingleObjective(tipdisplacement)

# --- CHOOSE the model type
model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

# --- CHOOSE the acquisition optimizer
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

# --- CHOOSE the type of acquisition
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

# --- CHOOSE a collection method
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

myBopt2D = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition,
                                                      evaluator, initial_design)

max_iter = 20  # Max number of iterations
max_time = 60  # Max seconds of algo runtime

myBopt2D.run_optimization(max_iter, max_time, verbosity=True)  # Running the BO algo

# Outputs
print(myBopt2D.x_opt)
print(myBopt2D.Y_best)

myBopt2D.plot_acquisition()

plt.show()