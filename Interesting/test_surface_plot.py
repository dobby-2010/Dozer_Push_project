# import dependencies
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm

# read in the data
data_1 = pd.read_csv("CE_36_B03-202304281615.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202304281630.csv")
data_2.columns = ["easting", "northing", "elevation"]

# check that the easting and northing coordinates are aligned between the rwo datasets
from pandas.testing import assert_frame_equal

assert_frame_equal(data_1[["easting", "northing"]], data_2[["easting", "northing"]])
# assert_frame_equal(data_1[["elevation"]], data_2[["elevation"]]) throws error. Therefore, they are perfectly aligned.

# create diff surface
data_diff = data_1[["easting", "northing"]]
data_diff["elevation"] = data_1["elevation"] - data_2["elevation"]


# plot the data with plotly.express
fig = go.Figure()
fig = px.scatter_3d(x=data_1["easting"], y=data_1["northing"], z=data_1["elevation"])
#fig.show()

fig = go.Figure()
fig = px.scatter_3d(x=data_2["easting"], y=data_2["northing"], z=data_2["elevation"])
#fig.show()

fig = go.Figure()
fig = px.scatter_3d(x=data_diff["easting"], y=data_diff["northing"], z=data_diff["elevation"])
fig.show()


# get diff surface including elevations
# get min E, N and max E, N then extend 1m x 1m grid filling in spaces with 0m elevation
# then get big array of elevations along each easting axis for all northings then np.trap again for those areas against northing (essentially line 37)
# see how this goes.

grid_e_min = data_diff["easting"].min()
grid_n_min = data_diff["northing"].min()
grid_e_max = data_diff["easting"].max()
grid_n_max = data_diff["northing"].max()

# make grid of points for easting and northing
x, y = np.meshgrid(
    np.arange(grid_e_min, grid_e_max + 0.1, 1),
    np.arange(grid_n_min, grid_n_max + 0.1, 1),
)

# create df of grid points for join
data_unjoined = pd.DataFrame([])
for i in range(x.shape[0]):
    data_unjoined = pd.concat([data_unjoined, pd.concat([pd.DataFrame(x[i]), pd.DataFrame(y[i])], axis=1)])
data_unjoined.columns = ["easting", "northing"]

# make join between grid df and data_diff df which should leave some null values in elevation column (fill these with 0)
result = pd.merge(data_unjoined, data_diff, how="left", on=["easting", "northing"])
result["elevation"] = result["elevation"].fillna(0)

# Test step to remove huge and impossible distances.
result.loc[result["elevation"] < -6, "elevation"] = 0

# there are some issues with the surfaces. This is mainly because the surface made by the differential of the two survey surfaces results in some wild numbers e.g. -6m or even -12m
# this would then create very large volumes moved and so the productivity would end up being quite large.

# let's just compare the two surveys
# combine the two datasets
data_1 = pd.read_csv("CE_36_B03-202304281615.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202304281630.csv")
data_2.columns = ["easting", "northing", "elevation"]
# data_1 = pd.read_csv("CE_36_B03-202304292100.csv")
# data_1.columns = ["easting", "northing", "elevation"]
# data_2 = pd.read_csv("CE_36_B03-202304292115.csv")
# data_2.columns = ["easting", "northing", "elevation"]
# data_3 = pd.read_csv("CE_36_B03-202304292045.csv")
# data_3.columns = ["easting", "northing", "elevation"]

data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
# data_3["dataset"] = "t0"

data_full = pd.concat([data_1, data_2], ignore_index=True)

# plot result
data_full = data_full[data_full["northing"] < 7516750]
data_full = data_full[data_full["northing"] > 7516700]
fig = go.Figure()
fig = px.scatter_3d(
    x=data_full["easting"], y=data_full["northing"], z=data_full["elevation"], color=data_full["dataset"]
)
fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode="data"), )
fig.show()

# the issue is probably that dozer is running over an area that has not been updated in a while and the dirt that has been moved (added or subtracted) is causing large discrepancies.
# could non-autonomous be working there too on the boundary of the working area of autonomous dozers and by being so close to that boundary it is hopping between an old surface and an updated one?

# on this date DZ2174 which is not an autonomous DZ was working. Check if was working in the same general area
# DZ2174 is working in between the autonomous DZs and is not the cause of having an imopssible grade between two lanes and is most likely caused by a surface not being updated but having been worked on and not recorded until the DZ passes near this area.
# Makes it difficult to determine what the dozer moved and what was already add/moved by other equipment and over how long?

norths = [7516720.99, 7516721.99, 7516722.99, 7516723.99]
for i in norths:
    data_full_1 = data_full[data_full["northing"] == i]
    fig = go.Figure()
    fig = px.scatter_3d(
        x=data_full_1["easting"], y=data_full_1["northing"], z=data_full_1["elevation"], color=data_full_1["dataset"]
    )
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode="data"))
    fig.show()