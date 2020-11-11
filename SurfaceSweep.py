import os
import numpy as np

# These files must exist; provide your own paths
Rtxtpath = "C:/temp/temptemp/Rvalue.txt"
htxtpath = "C:/temp/temptemp/hvalue.txt"
outputpath = "C:/temp/temptemp/Displacement_of_tip.txt"
responsesurfacepath = "C:/Users/leoli/Documents/testwrite.txt"

# Command line to run ABAQUS; provide your own file name
cmdl = 'cmd /c "cd c:/temp/temptemp && abaqus cae noGUI=ABAQUS_GPYOPT_Test_Problem2.py"'

def tipdisplacement(r, h):  # Function to optimize; x = (radius, height)
    open(Rtxtpath, "w").write(str(r))
    open(htxtpath, "w").write(str(h))
    os.system(cmdl)
    return float(open(outputpath, "r").read().strip())

n = 2  # Number of response points per parameter
rSweep = np.linspace(0.05, 0.2, n)
hSweep = np.linspace(0.2, 0.6, n)

responseSurface = np.zeros((n, n))  # Memory initialization
for i in range(n):
    for j in range(n):
        responseSurface[i, j] = tipdisplacement(rSweep[i], hSweep[j])

# Writing the results into a numpy array format
open(responsesurfacepath, "w").write(str(responseSurface))