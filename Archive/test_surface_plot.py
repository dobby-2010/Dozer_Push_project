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
from scipy.signal import filtfilt, cheby1

# read in the data
# data_1 = pd.read_csv("CE_36_B03-202304292100.csv")
# data_1.columns = ["easting", "northing", "elevation"]
# data_2 = pd.read_csv("CE_36_B03-202304292115.csv")
# data_2.columns = ["easting", "northing", "elevation"]
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

# # Triangulate surface
# from scipy.spatial import Delaunay

# triangs = Delaunay(data_diff[["easting", "northing"]])
# # add the elevations back to triangs
# # loop through all the triangulations
# for i in range(len(triangs.points[triangs.simplices])):
#     for j in range(len()) then get big array of elevations along each easting axis for all northings then np.trap again for those areas against northing (essentially line 37)
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
# result.loc[result["elevation"] < -6, "elevation"] = 0
# use new function here

# Method for getting volume of the delta surface is to have an equally sppaced grid then perform bilinear
# interpolation on the surface between four grid points (assuming that the surface is a 1m by 1m grid beginning at the origin)
#  and integrating over x and y for an analytical solution to the volume and then substituting in the 4 delta elevations.

# Integral solution is:
# https://www.wolframalpha.com/input?i=integrate+%28%281-x%29a+%2Bbx%29*%281-y%29+%2B%28%281-x%29c%2Bfx%29*%28y%29+dxdy with x=0 and y=0 substituted
# z00 - (z00)/2 + (z10)/2 - (z00)/2 + (z01)/2 + (z00)/4 - (z10)/4 - (z01)/4 + (z11)/4
# (z00)/4 + (z10)/4 + (z01)/4 + (z11)/4
# mean(z00 + z01 + z10 + z11)

# Then we have the ability to get volume for each grid square and just need to loop through them all.
# This is a grid of 344 x 112 or x/y.shape therefore:

import statistics as stat
from tqdm import tqdm

cuboid_volumes_pos = []  # not really cuboid volumes
cuboid_volumes_neg = []
# for i in range(len(result)):
#     stat.mean([result.loc[i,"elevation"],result.loc[i+1,"elevation"],result.loc[i,"elevation"],result.loc[i,"elevation"]])

# !!! Uncomment the following for more information !!!
# for index, row in tqdm(result.iterrows(), total=((x.shape[0] * x.shape[1]) - 1 - x.shape[1])):
#     if index >= (x.shape[0] * x.shape[1]) - 1 - x.shape[1]:
#         continue
#     else:
#         mean = stat.mean(
#             [
#                 result.loc[index, "elevation"],
#                 result.loc[index + 1, "elevation"],
#                 result.loc[index + x.shape[1], "elevation"],
#                 result.loc[index + x.shape[1] + 1, "elevation"],
#             ]
#         )
#         if mean > 0:
#             cuboid_volumes_pos.append(mean)
#         else:
#             cuboid_volumes_neg.append(mean)


# # # sum all the volumes in cuboid_volumes
# print(
#     f"This is how much total volume was moved in 15 minutes {sum(cuboid_volumes_pos) + (-1*sum(cuboid_volumes_neg))} BCM"
# )

# there are some issues with the volumes moved. This is mainly because the surface made by the differential of the two survey surfaces results in some wild numbers e.g. -6m or even -12m
# this then creates very large volumes moved and so the productivity ends up being quite large.

# Question the method for making the delta surface? This shouldn't be an issue. Need to check the actual values. Let's plot the values and elevations against the PAMS numbers.

# # read in the PAMS data
# from pyproj import Transformer

# pams_data = pd.read_csv("pams_positions_4-29-2100-2115-lcl.csv")
# trans = Transformer.from_crs(
#     "EPSG:4326",
#     f"EPSG:{20355}",
#     always_xy=True,
# )
# easting, northing = trans.transform(pams_data["longitude"].values, pams_data["latitude"].values)
# pams_data.loc[:, "easting"] = easting
# pams_data.loc[:, "northing"] = northing
# pams_data.loc[:, "elevation"] = pams_data["altitude"]
# pams_data.drop(columns=["latitude", "longitude", "altitude"], inplace=True)

# # plot the positions
# fig = go.Figure()
# fig = px.scatter_3d(x=pams_data["easting"], y=pams_data["northing"], z=pams_data["elevation"], color=pams_data["name"])
# fig.show()

# # plot the positions from pams and the final surface together and see how they compare.
# # elevations of positions are different for each dataset therefore subrtact min elevation from all points to make sure both are comparable
# data_1["elevation"] = data_1["elevation"] - data_1["elevation"].min()
# data_1["dataset"] = "MINESTAR t0"
# data_2["elevation"] = data_2["elevation"] - data_2["elevation"].min()
# data_2["dataset"] = "MINESTAR t1"
# pams_data["elevation"] = pams_data["elevation"] - pams_data["elevation"].min()
# pams_data["dataset"] = "PAMS"
# combined = pd.concat([pams_data, data_1, data_2], ignore_index=True)

# fig = go.Figure()
# fig = px.scatter_3d(x=combined["easting"], y=combined["northing"], z=combined["elevation"], color=combined["dataset"])
# fig.show()

# somehow the coordinates from the two sources are wildly different. Not sure how to reconcile the massive differences in location.


# let's just compare the two surveys
# combnie the two datasets
# data_1 = pd.read_csv("CE_36_B03-202304281615.csv")
# data_1.columns = ["easting", "northing", "elevation"]
# data_2 = pd.read_csv("CE_36_B03-202304281630.csv")
# data_2.columns = ["easting", "northing", "elevation"]
data_1 = pd.read_csv("CE_36_B03-202304292100.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202304292115.csv")
data_2.columns = ["easting", "northing", "elevation"]
# data_3 = pd.read_csv("CE_36_B03-202304292045.csv")
# data_3.columns = ["easting", "northing", "elevation"]

data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
# data_3["dataset"] = "t0"

data_full = pd.concat([data_1, data_2], ignore_index=True)

# plot result
# data_full = data_full[data_full["northing"] < 7516750]
# data_full = data_full[data_full["northing"] > 7516700]
fig = go.Figure()
fig = px.scatter_3d(
    x=data_full["easting"], y=data_full["northing"], z=data_full["elevation"], color=data_full["dataset"]
)
fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode="data"))
fig.show()


# the issue is probably that dozer is running over an area that has not been updated in a while and the dirt that has been moved (added or subtracted) is causing large discrepancies.
# could non-autonomous be working there too on the boundary of the working area of autonomous dozers and by being so close to that boundary it is hopping between an old surface and an updated one?

# on this date DZ2174 which is not an autonomous DZ was working. Check if was working in the same general area
# pams_data = pd.read_csv("pams_positions_4-28-1615-1630-lcl.csv")
# trans = Transformer.from_crs(
#     "EPSG:4326",
#     f"EPSG:{20355}",
#     always_xy=True,
# )
# easting, northing = trans.transform(pams_data["longitude"].values, pams_data["latitude"].values)
# pams_data.loc[:, "easting"] = easting
# pams_data.loc[:, "northing"] = northing
# pams_data.loc[:, "elevation"] = pams_data["altitude"]
# pams_data.drop(columns=["latitude", "longitude", "altitude"], inplace=True)

# # plot the positions
# fig = go.Figure()
# fig = px.scatter_3d(x=pams_data["easting"], y=pams_data["northing"], z=pams_data["elevation"], color=pams_data["name"])
# fig.show()

# DZ2174 is working in between the autonomous DZs and is not the cause of having an imopssible grade between two lanes and is most likely caused by a surface not being updated but having been worked on and not recorded until the DZ passes near this area.
# Makes it difficult to determine what the dozer moved and what was already add/moved by other equipment and over how long?

norths = [7516897.99, 7516921.99, 7516805.99]  # [7516720.99, 7516721.99, 7516722.99, 7516723.99]
for i in norths:
    data_full_1 = data_full[data_full["northing"] == i]
    fig = go.Figure()
    fig = px.scatter(x=data_full_1["easting"], y=data_full_1["elevation"], color=data_full_1["dataset"])
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode="data"))
    fig.show()

# procedure for fixing the great differences between two 15min surveys will be to look along each cross section for each metre of northing in the delta surface grid
# then have a procedure to check for these great differences and then apply some process for fixing these to prevent innacurate reporting of volumes.
# the process will be to break the elevations along this cross section into clusters using kmeans clustering or some other method. Next, the clusters that are too low for either the new or old survey will be
# discarded. The rest of the points left will be used in a linear interpolation step to fill in the discarded clusters.
# This is the process for fixing these strips of the survey surface; then the volume calculation can be completed with greater confidence.


# Attempt to cluster using K-Means
from sklearn.cluster import KMeans

data_broken = data_full[
    (data_full["northing"] == norths[0])
    | (data_full["northing"] == norths[1])
    | (data_full["northing"] == norths[2])
    # | (data_full["northing"] == norths[3])
]


# imagine passing data_full df to this function
def clean_cross_sections(
    delta_surface, max_clusters=5, end_weight=5.0, dataset_weight=1e6, max_stddev=2, max_dist_tolerance=1.3
):
    delta_surface["cluster"] = np.nan
    norths = delta_surface.northing.unique()
    features = []
    for k in tqdm(norths, total=len(delta_surface.northing.unique()), desc="Filter Cross Sections"):
        # Create parameters

        # Break into cross sections
        cross_section = delta_surface[delta_surface["northing"] == k]
        cross_section_t1 = cross_section[cross_section["dataset"] == "t1"]
        cross_section_t2 = cross_section[cross_section["dataset"] == "t2"]
        data_diff = cross_section_t1[["easting", "northing"]].reset_index(drop=True)
        data_diff["delta_elevation"] = cross_section_t1["elevation"].reset_index(drop=True) - cross_section_t2[
            "elevation"
        ].reset_index(drop=True)
        # only select points in cross section where there is a dffference in elevation
        cross_section_worthy = cross_section[
            cross_section["easting"].isin(list(data_diff[data_diff["delta_elevation"] != 0].easting))
        ]
        if cross_section_worthy.empty or (len(cross_section_worthy) < 4):
            continue

        cluster_centres = []
        elevation_centres = []
        # process to decide if even worth interpolating for each dataset
        for i in cross_section_worthy["dataset"].unique():
            df_temp = cross_section_worthy[cross_section_worthy["dataset"] == i].reset_index(drop=True)
            df_index = cross_section_worthy[cross_section_worthy["dataset"] == i].copy()
            elevation_centres.append(
                (i, stat.median(cross_section_worthy.loc[cross_section_worthy["dataset"] == i, "elevation"]))
            )
            # write process to find distance from point to point in this dataset
            import math

            dist = []
            for index, row in df_temp.iterrows():
                if index < len(df_temp) - 2:
                    dist.append(
                        math.dist(
                            df_temp.loc[index][["easting", "elevation"]],
                            df_temp.loc[index + 1][["easting", "elevation"]],
                        )
                    )
                else:
                    continue
            max_dist = max_dist_tolerance * math.dist(
                df_temp.iloc[0][["easting", "elevation"]], df_temp.iloc[-1][["easting", "elevation"]]
            )

            if (
                stat.stdev(df_temp.elevation) < max_stddev or sum(dist) < max_dist
            ):  # max_dist needs to be relative to range from start to finish maybe 20%
                continue
            else:
                df_temp.loc[:, "easting"] = df_temp.loc[:, "easting"] / df_temp.loc[:, "easting"].min()
                df_temp["elevation_weighted"] = df_temp["elevation"] * end_weight
                df_temp.loc[df_temp["dataset"] == "t1", "dataset"] = 1 * dataset_weight
                df_temp.loc[df_temp["dataset"] == "t2", "dataset"] = 2 * dataset_weight
                features = df_temp[["elevation", "easting", "dataset"]].to_numpy()
                sse = []
                silhouette_coefficients = []
                for j in tqdm(range(1, max_clusters + 1), desc="Clustering"):
                    clusters = KMeans(n_clusters=j, n_init=10).fit(features)
                    sse.append(clusters.inertia_)
                    if j >= 2:
                        score = silhouette_score(features, clusters.labels_)
                        silhouette_coefficients.append(score)

                # dist = features[clusters.labels_ == k, :] - clusters.cluster_centers_[k][:]
                # dist[:, [0, num_feature_points - 1, num_feature_points, -1]] /= end_weight
                # stderr.append(np.std(dist))
                # if np.max(stderr) > end_stderr:
                #     continue
                # break
                # Trying elbow method

            # update cross_section_worthy with cluster details
            kl = KneeLocator(range(1, max_clusters + 1), sse, curve="convex", direction="decreasing")
            n_clusters = kl.elbow

            clusters = KMeans(n_clusters=n_clusters, n_init=10).fit(features)
            cluster_centres.append(
                (
                    i,
                    clusters.cluster_centers_,
                )
            )
            df_index["cluster"] = clusters.labels_

            cross_section_worthy.loc[cross_section_worthy["dataset"] == i, "cluster"] = df_index.loc[:, "cluster"]

            fig = go.Figure()
            fig = px.scatter(
                x=cross_section_worthy.easting,
                y=cross_section_worthy.elevation,
                color=cross_section_worthy.cluster,
                color_continuous_scale=px.colors.qualitative.Plotly,
            )
            fig.show()

        # now need to interpolate each cluster

        # drop cluster with cluster centre that is farthest away mean elevation of the other dataset
        # Go through cluster_centres
        for dataset_clusters in cluster_centres:
            dataset = dataset_clusters[0]
            index_elev = [y[0] for y in elevation_centres].index(dataset)
            dataset_elev_median = elevation_centres[index_elev][1]

            centres = dataset_clusters[1]
            clusters_elevations = []
            for i in range(len(centres)):
                clusters_elevations.append(centres[i][0])

            # find cluster centre closest to the other dataset median elevation
            diff = [i - dataset_elev_median for i in clusters_elevations]
            diff_index = diff.index(min(diff))

            # diff_index in order of clusters.labels_
            # keep closest cluster to the other dataset & drop the rest
            cross_section_worthy.loc[
                (cross_section_worthy["dataset"] == dataset)
                & (cross_section_worthy["cluster"] != clusters.labels_[diff_index]),
                "elevation",
            ] = np.nan

        # dropping prematurely is always dangerous because of changing length of df
        # maybe just make the elevations null for now and overwrite next

        # come to fix later
        for s in cross_section_worthy["dataset"].unique():
            cross_section_worthy.loc[cross_section_worthy["dataset"] == s, "elevation"] = cross_section_worthy.loc[
                cross_section_worthy["dataset"] == s, "elevation"
            ].interpolate(method="linear")
            b, a = cheby1(N=3, Wn=0.94, rp=21)  # Wn = 0.9
            cross_section_worthy.loc[cross_section_worthy["dataset"] == s, "cluster"]
            if cross_section_worthy.loc[cross_section_worthy["dataset"] == s, "cluster"].isna().all():
                continue
            else:
                cross_section_worthy.loc[cross_section_worthy["dataset"] == s, "elevation"] = filtfilt(
                    b, a, cross_section_worthy.loc[cross_section_worthy["dataset"] == s, "elevation"].values
                )
                # overwrite these new elevations with those in the dataframe
                delta_surface.loc[
                    list(cross_section_worthy.loc[cross_section_worthy["dataset"] == s, "elevation"].index), "elevation"
                ] = cross_section_worthy.loc[cross_section_worthy["dataset"] == s, "elevation"]

            # ## or
            # run_interpolated = np.linspace(
            #     df_copy.loc[:, "easting"].iloc[0],
            #     df_copy.loc[:, "easting"].iloc[-1],
            #     len(df_copy.loc[:, "easting"]),
            # )
            # cross_interpolated = interp1d(
            #     df_copy.loc[:, "easting"],
            #     df_copy.loc[:, "elevation"],
            #     "linear",
            # )
            # updated_elevations = cross_interpolated(run_interpolated)
            # df_copy.loc[df_copy["cluster"] == cluster, "elevation"] = updated_elevations
            # cross_section_worthy.loc[
            #     (cross_section_worthy["dataset"] == i) & (cross_section_worthy["cluster"] == cluster), :
            # ] = df_copy.loc[df_copy["cluster"] == cluster, :]

        fig = go.Figure()
        fig = px.scatter(
            x=cross_section_worthy.easting, y=cross_section_worthy.elevation, color=cross_section_worthy.dataset
        )
        fig.show()

    return delta_surface


# clean_cross_sections(data_full)

# code before this would determine whether to apply this process

# Test DBSCAN
norths = [7516897.99, 7516921.99, 7516805.99]
data_full["cluster"] = np.nan
data_full_1 = data_full[data_full["northing"] == 7516897.99]
cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"]
cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"]
data_diff = cross_section_t1[["easting", "northing"]].reset_index(drop=True)
data_diff["delta_elevation"] = cross_section_t1["elevation"].reset_index(drop=True) - cross_section_t2[
    "elevation"
].reset_index(drop=True)
cross_section_worthy = data_full_1[
    data_full_1["easting"].isin(list(data_diff[data_diff["delta_elevation"] != 0].easting))
]
cross_section_worthy_1 = cross_section_worthy[cross_section_worthy["dataset"] == "t1"]
cross_section_worthy_2 = cross_section_worthy[cross_section_worthy["dataset"] != "t1"]
fig = go.Figure()
fig = px.scatter(x=cross_section_worthy_1.easting, y=cross_section_worthy_1.elevation)
fig.show()

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
    # pyplot.show()

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

    data_full_1.loc[update_values, :] = cross_section_worthy_1
fig = go.Figure()
fig = px.scatter(x=data_full_1.easting, y=data_full_1.elevation, color=data_full_1.dataset)
fig.show()

# This works exceptionally well. Makes more clusters though, not too big of an issue. Doesn't give cluster centre but that can be worked out with
# features[row_ix, 1].mean(), features[row_ix, 0].mean()

# But it does make much more believable or realistic cross sections


# put this into the cleaning function


# Test straight filter on cross section path
from scipy.signal import filtfilt, cheby1, butter, savgol_filter

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

cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
cross_section_t1_new = cross_section_t1.copy()
N = np.arange(2, 11, 1)
Wn = np.linspace(0.01, 0.99, 50)
grid = np.ones((len(N), len(Wn)))
for i, row in enumerate(N):
    for j, item in enumerate(Wn):
        b, a = cheby1(N=row, Wn=item, rp=21)
        cross_section_t1_new.loc[:, "elevation"] = pd.DataFrame(
            filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
        ).loc[:, 0]
        delta_elevation = cross_section_t1_new["elevation"] - cross_section_t2["elevation"]
        loss_fn = abs(delta_elevation).sum()
        grid[i, j] = loss_fn

grid_search = np.unravel_index(grid.argmin(), grid.shape)
optimised_N = N[grid_search[0]]
optimised_Wn = Wn[grid_search[1]]

b, a = cheby1(N=optimised_N, Wn=optimised_Wn, rp=21)
cross_section_t1.loc[:, "elevation"] = filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
fig.update_layout(title="Cheby1")
fig.show()


# try for butter
cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
cross_section_t1_new = cross_section_t1.copy()
N = np.arange(2, 11, 1)
Wn = np.linspace(0.01, 0.99, 50)
grid = np.ones((len(N), len(Wn)))
for i, row in enumerate(N):
    for j, item in enumerate(Wn):
        b, a = butter(N=row, Wn=item)
        cross_section_t1_new.loc[:, "elevation"] = pd.DataFrame(
            filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
        ).loc[:, 0]
        delta_elevation = cross_section_t1_new["elevation"] - cross_section_t2["elevation"]
        loss_fn = abs(delta_elevation).sum()
        grid[i, j] = loss_fn

grid_search = np.unravel_index(grid.argmin(), grid.shape)
optimised_N = N[grid_search[0]]
optimised_Wn = Wn[grid_search[1]]

b, a = butter(N=optimised_N, Wn=optimised_Wn)
cross_section_t1.loc[:, "elevation"] = filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
fig.update_layout(title="Butter")
fig.show()


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
cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
cross_section_t1_new = cross_section_t1.copy()

window = np.linspace(8, 70, 63)
poly_order = np.arange(2, 6, 1)
grid = np.ones((len(window), len(poly_order)))
for i, row in enumerate(window):
    for j, item in enumerate(poly_order):
        cross_section_t1_new.loc[:, "elevation"] = pd.DataFrame(
            savgol_filter(cross_section_t1.loc[:, "elevation"].values, int(window[i]), poly_order[j])
        ).loc[:, 0]
        delta_elevation = cross_section_t1_new["elevation"] - cross_section_t2["elevation"]
        loss_fn = abs(delta_elevation).sum()
        grid[i, j] = loss_fn

grid_search = np.unravel_index(grid.argmin(), grid.shape)
optimised_window = int(window[grid_search[0]])
optimised_poly_order = poly_order[grid_search[1]]

cross_section_t1.loc[:, "elevation"] = savgol_filter(
    cross_section_t1.loc[:, "elevation"].values, optimised_window, optimised_poly_order
)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
fig.update_layout(title="Savgol")
fig.show()


# Plots all the cross sections I think are worth looking at
norths_to_inspect = []
for i in data_diff_full["northing"].unique():
    x = data_diff_full.loc[
        (data_diff_full["northing"] == i) & ((data_diff_full["elevation"] < -6) | ((data_diff_full["elevation"] > 6))),
        :,
    ]
    if not x.empty:
        fig = go.Figure()
        fig = px.scatter(
            x=data_diff_full.loc[(data_diff_full["northing"] == i), "easting"],
            y=data_diff_full.loc[(data_diff_full["northing"] == i), "elevation"],
        )
        fig.show()
        norths_to_inspect.append(i)
for i in norths_to_inspect:
    data_full_1 = data_full[data_full["northing"] == i]
    cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
    cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"].reset_index(drop=True)
    data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
    fig = go.Figure()
    fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
    fig.show()


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
fig.show()
# replace_section_1 = []
# for p in range(len(data_1)):
#     if plane_1.distance_point(points_1[p]) > threshold_distance:
#         replace_section_1.append(p)
# replace_section_1 = list(data_1.index[replace_section_1])
# data_1.loc[replace_section_1, "elevation"] = np.nan
# fig = go.Figure()
# fig = px.scatter_3d(x=data_1["easting"], y=data_1["northing"], z=data_1["elevation"], color=data_1["dataset"])
fig.show()


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
# Now attempt to fit parabolic/quadratic model to cross sections of concern and remove the outliers that are too far from the line of best fit.
# Choosing quadratic model to begin with because it is simple and the shape of the cross section generally represents a quadratic or some other model with ccurvature.


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

data_full_1 = data_full[data_full["northing"] == norths_to_inspect[0]]
cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"].reset_index(drop=True)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)

# agglomerative
ac = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(data[["easting", "elevation"]])
data["cluster"] = ac.labels_
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.delta_elevation, color=data.cluster)
fig.show()
