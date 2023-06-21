# import dependencies
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Attempt to fit plane of best fit and see if there are any of the points that i want to be removed can be removed using this method.

from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d

threshold_distance = 4
data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]
data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
# break data into 3 parts using np.percentile
data_1 = data_full[data_full["easting"] < np.percentile(data_full.easting, 33.333333)]
data_2 = data_full[
    (data_full["easting"] >= np.percentile(data_full.easting, 33.333333))
    & (data_full["easting"] <= np.percentile(data_full.easting, 66.666666))
]
data_3 = data_full[data_full["easting"] > np.percentile(data_full.easting, 66.666666)]
data_1_values = data_1.loc[:, ["easting", "northing", "elevation"]].values
points_1 = Points(data_1_values)
plane_1 = Plane.best_fit(points_1)
x1, y1, z1 = plane_1.to_mesh()
x1 = x1.flatten()
y1 = y1.flatten()
z1 = z1.flatten()
plane_1_points = np.vstack([x1, y1, z1]).T
# fig = go.Figure()
# fig = px.scatter_3d(x=data_1["easting"], y=data_1["northing"], z=data_1["elevation"], color=data_1["dataset"])
# fig.show()
# replace_section_1 = []
# for p in range(len(data_1)):
#     if plane_1.distance_point(points_1[p]) > threshold_distance:
#         replace_section_1.append(p)
# replace_section_1 = list(data_1.index[replace_section_1])
# data_1.loc[replace_section_1, "elevation"] = np.nan
# fig = go.Figure()
# fig = px.scatter_3d(x=data_1["easting"], y=data_1["northing"], z=data_1["elevation"], color=data_1["dataset"])
# fig.show()


# From this we can see that there are some points being removed that are not meant to be removed
# Let's try breaking the surface up further into 9 sections. so now we partition the data on the easting axis

# break data_1 into 3
data_1_1 = data_1[data_1["northing"] < np.percentile(data_1.northing, 33.333333)]
data_1_2 = data_1[
    (data_1["northing"] >= np.percentile(data_1.northing, 33.333333))
    & (data_1["northing"] <= np.percentile(data_1.northing, 66.666666))
]
data_1_3 = data_1[data_1["northing"] > np.percentile(data_1.northing, 66.666666)]

# plane 1
data_1_1_values = data_1_1.loc[:, ["easting", "northing", "elevation"]].values
points_1_1 = Points(data_1_1_values)
plane_1_1 = Plane.best_fit(points_1_1)
x11, y11, z11 = plane_1_1.to_mesh()
x11 = x11.flatten()
y11 = y11.flatten()
z11 = z11.flatten()
plane_1_1_points = np.vstack([x11, y11, z11]).T

# Lets check plane 1
fig = go.Figure()
fig = px.scatter_3d(x=data_1_1["easting"], y=data_1_1["northing"], z=data_1_1["elevation"], color=data_1_1["dataset"])
fig.show()
replace_section_1 = []
for p in range(len(data_1_1)):
    if plane_1_1.distance_point(points_1_1[p]) > threshold_distance:
        replace_section_1.append(p)
replace_section_1 = list(data_1_1.index[replace_section_1])
data_1_1.loc[replace_section_1, "elevation"] = np.nan
fig = go.Figure()
fig = px.scatter_3d(x=data_1_1["easting"], y=data_1_1["northing"], z=data_1_1["elevation"], color=data_1_1["dataset"])
fig.show()

# plane 2
data_1_2_values = data_1_2.loc[:, ["easting", "northing", "elevation"]].values
points_1_2 = Points(data_1_2_values)
plane_1_2 = Plane.best_fit(points_1_2)
x12, y12, z12 = plane_1_2.to_mesh()
x12 = x12.flatten()
y12 = y12.flatten()
z12 = z12.flatten()
plane_1_2_points = np.vstack([x12, y12, z12]).T

# Lets check plane 2
fig = go.Figure()
fig = px.scatter_3d(x=data_1_2["easting"], y=data_1_2["northing"], z=data_1_2["elevation"], color=data_1_2["dataset"])
fig.show()
replace_section_2 = []
for p in range(len(data_1_2)):
    if plane_1_2.distance_point(points_1_2[p]) > threshold_distance:
        replace_section_2.append(p)
replace_section_2 = list(data_1_2.index[replace_section_2])
data_1_2.loc[replace_section_2, "elevation"] = np.nan
fig = go.Figure()
fig = px.scatter_3d(x=data_1_2["easting"], y=data_1_2["northing"], z=data_1_2["elevation"], color=data_1_2["dataset"])
fig.show()

# plane 3
data_1_3_values = data_1_3.loc[:, ["easting", "northing", "elevation"]].values
points_1_3 = Points(data_1_3_values)
plane_1_3 = Plane.best_fit(points_1_3)
x13, y13, z13 = plane_1_3.to_mesh()
x13 = x13.flatten()
y13 = y13.flatten()
z13 = z13.flatten()
plane_1_3_points = np.vstack([x13, y13, z13]).T

# Lets check plane 3
fig = go.Figure()
fig = px.scatter_3d(x=data_1_3["easting"], y=data_1_3["northing"], z=data_1_3["elevation"], color=data_1_3["dataset"])
fig.show()
replace_section_3 = []
for p in range(len(data_1_3)):
    if plane_1_3.distance_point(points_1_3[p]) > threshold_distance:
        replace_section_3.append(p)
replace_section_3 = list(data_1_3.index[replace_section_3])
data_1_3.loc[replace_section_3, "elevation"] = np.nan
fig = go.Figure()
fig = px.scatter_3d(x=data_1_3["easting"], y=data_1_3["northing"], z=data_1_3["elevation"], color=data_1_3["dataset"])
fig.show()


data_2_values = data_2.loc[:, ["easting", "northing", "elevation"]].values
points_2 = Points(data_1_values)
plane_2 = Plane.best_fit(points_1)
x2, y2, z2 = plane_2.to_mesh()

data_3_values = data_3.loc[:, ["easting", "northing", "elevation"]].values
points_3 = Points(data_1_values)
plane_3 = Plane.best_fit(points_1)
x3, y3, z3 = plane_3.to_mesh()
# plot_3d(
#     plane.plotter(alpha=0.2)
# )
# plot_3d(
#     points.plotter(c="k", s=50, depthshade=False),
# )

# Plane of best fit did not seem to work out or needs finer tuning.
