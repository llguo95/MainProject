import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import GPyOpt

filename = 'target_day_20140422.dat'
f = open(filename, 'r')
contents = f.readlines()

## Create a dictionary for the forecasted
forecast_dict = {}
for line in range(1, len(contents)):
    line_split = contents[line].split(' ')
    try:
        forecast_dict[line_split[0], line_split[1]][line_split[2]] = {'MaxT': float(line_split[3]),
                                                                      'MinT': float(line_split[4][:-1])}
    except:
        forecast_dict[line_split[0], line_split[1]] = {}
        forecast_dict[line_split[0], line_split[1]][line_split[2]] = {'MaxT': float(line_split[3]),
                                                                      'MinT': float(line_split[4][:-1])}
keys = forecast_dict.keys()
day_out = '0'  # 0-7
temp = 'MaxT'  # MaxT or MinT
temperature = []
lat = []
lon = []
for key in keys:
    temperature.append(float(forecast_dict[key][day_out][temp]))
    lat.append(float(key[0]))
    lon.append(float(key[1]))

## Create numpy arrays for the analyisis and remove Alaska and the islands
lon = np.array(lon)
lat = np.array(lat)
sel = np.logical_and(np.logical_and(lat > 24, lat < 51), np.logical_and(lon > -130, lon < -65))
stations_coordinates_all = np.array([lon, lat]).T
stations_maxT_all = np.array([temperature]).T
stations_coordinates = stations_coordinates_all[sel, :]
stations_maxT = stations_maxT_all[sel, :]

#  Class that defines the function to optimize given the available locations
class max_Temp(object):
    def __init__(self, stations_coordinates, stations_maxT):
        self.stations_coordinates = stations_coordinates
        self.stations_maxT = stations_maxT

    def f(self, x):
        return np.dot(0.5 * (self.stations_coordinates == x).sum(axis=1), self.stations_maxT)[:, None]


# Objective function given the current inputs
func = max_Temp(stations_coordinates, stations_maxT)

domain = [{'name': 'stations', 'type': 'bandit', 'domain': stations_coordinates}]  # armed bandit with the locations

from numpy.random import seed

seed(123)
myBopt = GPyOpt.methods.BayesianOptimization(f=func.f,  # function to optimize
                                             domain=domain,
                                             initial_design_numdata=1,
                                             acquisition_type='EI',
                                             exact_feval=True,
                                             normalize_Y=False,
                                             optimize_restarts=10,
                                             acquisition_weight=2,
                                             de_duplication=True)

# Run the optimization
max_iter = 25  # evaluation budget
myBopt.run_optimization(max_iter)

plt.figure(figsize=(15, 7))
jet = plt.cm.get_cmap('jet')
sc = plt.scatter(stations_coordinates[:, 0], stations_coordinates[:, 1], c=np.ravel(stations_maxT), s=1, vmin=0,
                 vmax=35)
cbar = plt.colorbar(sc, shrink=1)
cbar.set_label(temp)
plt.plot(myBopt.X[:, 0], myBopt.X[:, 1], 'k.', markersize=8, label='Observed stations')
plt.plot(myBopt.x_opt[0], myBopt.x_opt[1], 'ro', markersize=10, label='Best found')
plt.plot(stations_coordinates[np.argmin(stations_maxT), 0], stations_coordinates[np.argmin(stations_maxT), 1], 'r*',
         markersize=15, label='Coldest station')
plt.legend(loc='lower right')
plt.ylim((25, 50))
plt.xlim((-128, -65))

plt.title('Max. temperature on 22 April 2014', size=25)
plt.xlabel('Longitude', size=15)
plt.ylabel('Latitude', size=15)
plt.text(-125, 28, 'Total stations = ' + str(stations_maxT.shape[0]), size=20)
plt.text(-125, 26.5, 'Sampled stations = ' + str(myBopt.X.shape[0]), size=20)

plt.figure(figsize=(8, 5))
xx = plt.hist(stations_maxT, bins=50)
plt.title('Distribution of max. temperatures', size=25)
plt.vlines(min(stations_maxT), 0, 1000, lw=3, label='Coldest station')
plt.vlines(myBopt.fx_opt, 0, 1000, lw=3, linestyles=u'dotted', label='Best found')
plt.legend()
plt.xlabel('Max. temperature', size=15)
plt.xlabel('Frequency', size=15)

plt.show()
