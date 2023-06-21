# import dependencies
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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering


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
print(data_diff)
cross_section_worthy_1 = cross_section_worthy[cross_section_worthy["dataset"] == "t1"].reset_index(drop=True)
cross_section_worthy_2 = cross_section_worthy[cross_section_worthy["dataset"] != "t1"].reset_index(drop=True)
fig = go.Figure()
fig = px.scatter(x=cross_section_worthy_1.easting, y=cross_section_worthy_1.elevation)
# fig.show()
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
# fig.show()


from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN

from matplotlib import pyplot

features = cross_section_worthy_1[["elevation", "easting"]].to_numpy()
# define the model
model = DBSCAN(eps=3, min_samples=4)
# fit model and predict clusters
yhat = model.fit_predict(cross_section_worthy_1[["elevation", "easting"]])
# retrieve unique clusters
clusters = unique(yhat)


distribution_limit = 4
for cluster in clusters:
    # get row indexes for samples with this cluster
    cross_section_worthy_1["cluster"] = yhat
    row_ix = where(yhat == cluster)

    # create scatter of these samples
    pyplot.scatter(features[row_ix, 1], features[row_ix, 0])
    pyplot.scatter(features[row_ix, 1].mean(), features[row_ix, 0].mean())
    # show the plot
    pyplot.show()

    # find the centroid of this cluster
    centroid_elevation = features[row_ix, 0].mean()
    other_surface_elevation = cross_section_worthy_2.elevation.mean()
    if (centroid_elevation < other_surface_elevation - distribution_limit) or (
        centroid_elevation > other_surface_elevation + distribution_limit
    ):
        row_ix = list(cross_section_worthy_1.iloc[row_ix]["elevation"].index)
        cross_section_worthy_1.loc[row_ix, "elevation"] = np.nan

if cross_section_worthy_1.elevation.isna().all():
    # just make this the same so that we don't count this cross section
    cross_section_worthy_1.loc[:, "elevation"] = cross_section_worthy_2["elevation"].values
    update_values = list(cross_section_worthy_1.index)
    data_full_1.loc[update_values, "elevation"] = cross_section_worthy_1["elevation"].values
else:
    # In this area begin interpolation
    cross_section_worthy_1.loc[:, "elevation"] = cross_section_worthy_1.loc[:, "elevation"].interpolate(method="linear")
    b, a = cheby1(N=3, Wn=0.94, rp=21)
    cross_section_worthy_1.loc[:, "elevation"] = filtfilt(b, a, cross_section_worthy_1.loc[:, "elevation"].values)
    update_values = list(cross_section_worthy_1.index)

    data_full_1.loc[data_full_1.index[update_values], :] = cross_section_worthy_1
fig = go.Figure()
fig = px.scatter(x=data_full_1.easting, y=data_full_1.elevation, color=data_full_1.dataset)
fig.show()

# This works exceptionally well. Makes more clusters though, not too big of an issue. Doesn't give cluster centre but that can be worked out with
# features[row_ix, 1].mean(), features[row_ix, 0].mean()

# But it does make much more believable or realistic cross sections


# perform hyperparameter grid search of multiple filters trying to minimize sum of difference
# from tqdm import tqdm

# cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
# cross_section_t1_new = cross_section_t1.copy()
# N = np.arange(2, 11, 1)
# Wn = np.linspace(0.01, 0.99, 50)
# grid = np.ones((len(N), len(Wn)))
# for i, row in enumerate(N):
#     for j, item in enumerate(Wn):
#         b, a = cheby1(N=row, Wn=item, rp=21)
#         cross_section_t1_new.loc[:, "elevation"] = pd.DataFrame(
#             filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
#         ).loc[:, 0]
#         delta_elevation = cross_section_t1_new["elevation"] - cross_section_t2["elevation"]
#         loss_fn = abs(delta_elevation).sum()
#         grid[i, j] = loss_fn

# grid_search = np.unravel_index(grid.argmin(), grid.shape)
# optimised_N = N[grid_search[0]]
# optimised_Wn = Wn[grid_search[1]]

# b, a = cheby1(N=optimised_N, Wn=optimised_Wn, rp=21)
# cross_section_t1.loc[:, "elevation"] = filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
# data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
# fig = go.Figure()
# fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
# fig.update_layout(title="Cheby1")
# fig.show()


# try for butter
# cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
# cross_section_t1_new = cross_section_t1.copy()
# N = np.arange(2, 11, 1)
# Wn = np.linspace(0.01, 0.99, 50)
# grid = np.ones((len(N), len(Wn)))
# for i, row in enumerate(N):
#     for j, item in enumerate(Wn):
#         b, a = butter(N=row, Wn=item)
#         cross_section_t1_new.loc[:, "elevation"] = pd.DataFrame(
#             filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
#         ).loc[:, 0]
#         delta_elevation = cross_section_t1_new["elevation"] - cross_section_t2["elevation"]
#         loss_fn = abs(delta_elevation).sum()
#         grid[i, j] = loss_fn

# grid_search = np.unravel_index(grid.argmin(), grid.shape)
# optimised_N = N[grid_search[0]]
# optimised_Wn = Wn[grid_search[1]]

# b, a = butter(N=optimised_N, Wn=optimised_Wn)
# cross_section_t1.loc[:, "elevation"] = filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
# data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
# fig = go.Figure()
# fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
# fig.update_layout(title="Butter")
# fig.show()


# # try for convolve
# cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
# cross_section_t1_new = cross_section_t1.copy()

# x = np.linspace(0, 2 * np.pi, 100)
# y = np.sin(x) + np.random.random(100) * 0.8


# def smooth(y, box_pts):
#     box = np.ones(box_pts) / box_pts
#     y_smooth = np.convolve(y, box, mode="same")
#     return y_smooth


# plt.plot(x, y)
# plt.plot(x, smooth(y, 3))
# plt.plot(x, smooth(y, 19))

# try for savgol_filter
# cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
# cross_section_t1_new = cross_section_t1.copy()

# window = np.linspace(8, 70, 63)
# poly_order = np.arange(2, 6, 1)
# grid = np.ones((len(window), len(poly_order)))
# for i, row in enumerate(window):
#     for j, item in enumerate(poly_order):
#         cross_section_t1_new.loc[:, "elevation"] = pd.DataFrame(
#             savgol_filter(cross_section_t1.loc[:, "elevation"].values, int(window[i]), poly_order[j])
#         ).loc[:, 0]
#         delta_elevation = cross_section_t1_new["elevation"] - cross_section_t2["elevation"]
#         loss_fn = abs(delta_elevation).sum()
#         grid[i, j] = loss_fn

# grid_search = np.unravel_index(grid.argmin(), grid.shape)
# optimised_window = int(window[grid_search[0]])
# optimised_poly_order = poly_order[grid_search[1]]

# cross_section_t1.loc[:, "elevation"] = savgol_filter(
#     cross_section_t1.loc[:, "elevation"].values, optimised_window, optimised_poly_order
# )
# data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
# fig = go.Figure()
# fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
# fig.update_layout(title="Savgol")
# fig.show()
'''
data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

norths_to_inspect = []
for i in data_diff_full["northing"].unique():
    x = data_diff_full.loc[
        (data_diff_full["northing"] == i)
        & ((data_diff_full["elevation"] < -6) | ((data_diff_full["elevation"] > 6))),
        :,
    ]
# 7516846.99 is an immediate success
df1 = data_full[data_full["northing"] == 7516917.99]
cs1 = df1[df1["dataset"] == "t1"].reset_index(drop=True)
cs2 = df1[df1["dataset"] == "t2"].reset_index(drop=True)
df = pd.concat([cs1, cs2], ignore_index=True)
dfdiff = cs1[["easting", "northing"]].reset_index(drop=True)
dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

# Let us try agglomerative clustering with single linkage nearest neighbours
silhouette_coefficients = []

for k in tqdm(range(2, 10), desc="Clustering"):
    ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
    score = silhouette_score(dfdiff[["easting", "delta_elevation"]], ac.labels_)
    silhouette_coefficients.append(score)
fig = go.Figure()
fig = px.scatter(x=range(2, 10), y=silhouette_coefficients)
fig.show()
clusters = range(2, 10)[silhouette_coefficients.index(max(silhouette_coefficients))]

ac = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
dfdiff["cluster"] = ac.labels_
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.show()


# Let us try gaussian mixture models clustering
# let us attempt clustering with variable number of clusters and see what number to use
silhouette_coefficients = []

for k in tqdm(range(2, 10), desc="Clustering"):
    gm = GaussianMixture(n_components=k, random_state=0).fit(dfdiff[["easting", "delta_elevation"]])
    score = silhouette_score(dfdiff[["easting", "delta_elevation"]], gm.predict(dfdiff[["easting", "delta_elevation"]]))
    silhouette_coefficients.append(score)
fig = go.Figure()
fig = px.scatter(x=range(2, 10), y=silhouette_coefficients)
fig.show()
clusters = range(2, 10)[silhouette_coefficients.index(max(silhouette_coefficients))]

gmm = GaussianMixture(n_components=clusters, random_state=0).fit(dfdiff[["easting", "delta_elevation"]])
dfdiff["cluster"] = gmm.predict(dfdiff[["easting", "delta_elevation"]])
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.show()


# try clustering purely both the datasets
data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]
data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

norths_to_inspect = []
for i in data_diff_full["northing"].unique():
    x = data_diff_full.loc[
        (data_diff_full["northing"] == i) & ((data_diff_full["elevation"] < -6) | ((data_diff_full["elevation"] > 6))),
        :,
    ]
    if not x.empty:
        norths_to_inspect.append(i)

data_full_1 = data_full[data_full["northing"] == norths_to_inspect[17]]
cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"].reset_index(drop=True)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)


ac = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(data[["easting", "elevation"]])
data["cluster"] = ac.labels_
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.cluster)
fig.show()

gmm = GaussianMixture(n_components=2, random_state=0).fit(data[["easting", "elevation"]])
data["cluster"] = gmm.predict(data[["easting", "elevation"]])
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.cluster)
fig.show()

'''