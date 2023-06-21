import plotly.graph_objects as go
import pandas as pd
import numpy as np

#read data from a csv
data_1 = pd.read_csv("CE_36_B03-202304281615.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202304281630.csv")
data_2.columns = ["easting", "northing", "elevation"]

#check that 
from pandas.testing import assert_frame_equal
assert_frame_equal(data_1[["easting", "northing"]], data_2[["easting", "northing"]])
x = data_1["easting"]
y = data_1["northing"]
# print(x,y)

#  get diff surface including elevations
# get min E, N and max E, N then extend 1m x 1m grid filling in spaces with 0m elevation
# then get big array of elevations along each easting axis for all northings then np.trap again for those areas against northing (essentially line 37)
# see how this goes.

grid_e_min = data_1["easting"].min()
grid_n_min = data_1["northing"].min()
grid_e_max = data_1["easting"].max()
grid_n_max = data_1["northing"].max()

# make grid of points for easting and northing
x, y = np.meshgrid(
    np.arange(grid_e_min, grid_e_max + 0.1, 1),
    np.arange(grid_n_min, grid_n_max + 0.1, 1),
)

#plot the data 
# fig = go.Figure(data=[go.Surface(x=data_1["easting"], y=data_1["northing"], z=data_1["elevation"])])
# fig.update_layout(title='Surface Plot', autosize=False)

# fig.show()
