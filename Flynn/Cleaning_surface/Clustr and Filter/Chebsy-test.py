# Test straight filter on cross section path
from scipy.signal import filtfilt, cheby1, butter, savgol_filter
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from scipy.signal import filtfilt, cheby1, butter, savgol_filter
from tqdm import tqdm
import statistics as stat
from numpy import unique
from numpy import where
from pandas.testing import assert_frame_equal
from sklearn import metrics


data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]
data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]
fig = go.Figure()
fig = px.scatter_3d(x=data_full.easting, y=data_full.northing, z=data_full.elevation, color=data_full.dataset)
fig.show()
norths = [7516846.99, 7516921.99, 7516805.99]
data_full["cluster"] = np.nan
data_full_1 = data_full[data_full["northing"] == 7516921.99]
cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"].reset_index(drop=True)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
data_diff = cross_section_t1[["easting", "northing"]].reset_index(drop=True)
data_diff["delta_elevation"] = cross_section_t1["elevation"].reset_index(drop=True) - cross_section_t2[
    "elevation"
].reset_index(drop=True)
cross_section_worthy = data_full_1[
    data_full_1["easting"].isin(list(data_diff[data_diff["delta_elevation"] != 0].easting))
]
cross_section_worthy_1 = cross_section_worthy[cross_section_worthy["dataset"] == "t1"].reset_index(drop=True)
cross_section_worthy_2 = cross_section_worthy[cross_section_worthy["dataset"] != "t1"].reset_index(drop=True)
fig = go.Figure()
fig = px.scatter(x=cross_section_worthy_1.easting, y=cross_section_worthy_1.elevation)
fig.show()
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
fig.show()


# perform hyperparameter grid search of multiple filters trying to minimize sum of difference
from tqdm import tqdm

cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True) # this is a single cross section of a single surface
cross_section_t1_new = cross_section_t1.copy()
N = np.arange(2, 11, 1) # possible values of first parameter
Wn = np.linspace(0.01, 0.99, 50) # possible values of second parameter
grid = np.ones((len(N), len(Wn))) # this is where I will store the values from my loss function
for i, row in enumerate(N): # loop through all possible permutations of the 2 parameters
    for j, item in enumerate(Wn):
        b, a = cheby1(N=row, Wn=item, rp=21) # parameters define a & b
        cross_section_t1_new.loc[:, "elevation"] = pd.DataFrame(
            filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
        ).loc[:, 0] # a and b determine the smoothing of elevations
        delta_elevation = cross_section_t1_new["elevation"] - cross_section_t2["elevation"] # compare new elevations to the original
        loss_fn = abs(delta_elevation).sum() # my loss function here is just the sum of the difference
        grid[i, j] = loss_fn # store this combination of the hyperparameter's loss value in my grid

grid_search = np.unravel_index(grid.argmin(), grid.shape) # find the index of the minimum loss (want to minimize loss)
optimised_N = N[grid_search[0]] # this is the optimized param N
optimised_Wn = Wn[grid_search[1]] # this is the optimized param Wn

b, a = cheby1(N=optimised_N, Wn=optimised_Wn, rp=21) # Let's recompute the elevations using these optimised a & b
cross_section_t1.loc[:, "elevation"] = filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
fig.update_layout(title="Cheby1") # display the resulting smoothed elevations
fig.show()