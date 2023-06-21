import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy
from numpy.random import randn
from scipy import array, newaxisort numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as goimp




data_1 = pd.read_csv("CE_36_B03-202304281615.csv")
data_1.columns = ["easting", "northing", "elevation"]
nrows, ncols = 32785, 3
# x = np.linspace(0, 12.5, ncols)
# y = np.linspace(-6.2, 6.2, nrows)



x = data_1["easting"]
y = data_1["northing"]
z = data_1["elevation"]

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'),
                       layout='constrained')
# surf = ax.plot_surface(x, y , z, cmap='Blues_r', ec='gray', lw=0.2)
surf = ax.plot_trisurf(x, y, z , antialiased=False)
plt.xlabel('x') ; plt.ylabel('y')
#plt.colorbar(surf)
fig.colorbar(surf)
plt.show()