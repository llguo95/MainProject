import os
import GPyOpt
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import seed  # fixed seed
# =============================================================================
seed(123456)
# =============================================================================

space = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.05, 0.2)},
         {'name': 'var_2', 'type': 'continuous', 'domain': (0.2, 0.6)}]
constraints = [{'name': 'constr_1', 'constraint': '-(np.pi * x[:, 0] ** 2 + 0.4 * x[:, 1] - 0.2)'}]
feasible_region = GPyOpt.Design_space(space=space, constraints=constraints)
initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 3)

# Grid of points to make the plots
grid = 400
bounds = feasible_region.get_continuous_bounds()
X1 = np.linspace(bounds[0][0], bounds[0][1], grid)
X2 = np.linspace(bounds[1][0], bounds[1][1], grid)
x1, x2 = np.meshgrid(X1, X2)
X = np.hstack((x1.reshape(grid * grid, 1), x2.reshape(grid * grid, 1)))

# Check the points in the feasible region.
masked_ind = feasible_region.indicator_constraints(X).reshape(grid, grid)
masked_ind = np.ma.masked_where(masked_ind > 0.5, masked_ind)
masked_ind[1, 1] = 1

# Make the plots
plt.figure()

# Feasible region
plt.contourf(X1, X2, masked_ind, 100, cmap=plt.cm.bone, alpha=1, origin='lower')

# These files must exist; provide your own paths
Rtxtpath = "C:/temp/temptemp/Rvalue.txt"
htxtpath = "C:/temp/temptemp/hvalue.txt"
outputpath = "C:/temp/temptemp/Displacement_of_tip.txt"

# Command line to run ABAQUS; provide your own file name & check problem file itself
cmdl = 'cmd /c "cd c:/temp/temptemp && abaqus cae noGUI=ABAQUS_GPYOPT_Test_Problem2.py"'

def tipdisplacement(x):  # Function to optimize; x = (radius, height)
    open(Rtxtpath, "w").write(str(x[:, 0][0]))
    open(htxtpath, "w").write(str(x[:, 1][0]))
    os.system(cmdl)
    return float(open(outputpath, "r").read().strip())

max_iter = 10  # Max number of iterations
max_time = 120  # Max seconds of algo runtime

current_iter = 0

files = []  # File name string array initialization

bo_step = GPyOpt.methods.BayesianOptimization(f=tipdisplacement, domain=space, constraints=constraints,
                                              initial_design_numdata=3)  # Initial DOE
X_step = bo_step.X  # Defining input of iterative step
Y_step = bo_step.Y  # Defining output of iterative step

while current_iter < max_iter:
    print("iteration ", current_iter)
    bo_step = GPyOpt.methods.BayesianOptimization(f=tipdisplacement, domain=space, constraints=constraints,
                                                  X=X_step, Y=Y_step)  # Optimization
    print(bo_step.X)

    x_next = bo_step.suggest_next_locations()  # Querying new optimization input point
    y_next = tipdisplacement(x_next)

    X_step = np.vstack((X_step, x_next))  # Combining older steps with new step
    Y_step = np.vstack((Y_step, y_next))

    # Save images
    if current_iter < 10:
        strxFile = 'opt_0%d' % current_iter
    else:
        strxFile = 'opt_%d' % current_iter
    files.append(strxFile + ".png")
    bo_step.plot_acquisition(filename=strxFile)

    current_iter += 1

print("Your file names:", files)

bo_step.plot_acquisition()

plt.show()
