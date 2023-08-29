import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go




data_1 = pd.read_csv("CE_36_B03-202304281615.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202304281630.csv")
data_2.columns = ["easting", "northing", "elevation"]

nrows, ncols = 32785, 3
# x = np.linspace(0, 12.5, ncols)
# y = np.linspace(-6.2, 6.2, nrows)
from pandas.testing import assert_frame_equal

assert_frame_equal(data_1[["easting", "northing"]], data_2[["easting", "northing"]])
# assert_frame_equal(data_1[["elevation"]], data_2[["elevation"]]) throws error. Therefore, they are perfectly aligned.

# create diff surface
data_diff = data_1[["easting", "northing"]]
data_diff["elevation"] = data_1["elevation"] - data_2["elevation"]




x = data_diff["easting"]
y = data_diff["northing"]
z = data_diff["elevation"]

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'),
                       layout='constrained')
# surf = ax.plot_surface(x, y , z, cmap='Blues_r', ec='gray', lw=0.2)
surf = ax.plot_trisurf(x, y, z, linewidth=0 , antialiased=False)
plt.xlabel('x') ; plt.ylabel('y')
#plt.colorbar(surf)
fig.colorbar(surf, ax=ax, shrink=0.5)
plt.show()