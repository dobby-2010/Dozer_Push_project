import numpy as np

data = np.load('jacksboro_fault_dem.npz')
for key, value in data.items():
    np.savetxt("jacksboro" + key + ".csv", value)