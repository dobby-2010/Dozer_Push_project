import pandas as pd
import numpy as np
import statistics as stat
from sklearn.metrics import silhouette_score
from scipy.signal import filtfilt, cheby1
from tqdm import tqdm
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN
from pandas.testing import assert_frame_equal


##################################################################################################################################
############## Define functions ##################################################################################################
def calculate_volume(rows, columns, result):
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

    cuboid_volumes_pos = []  # not really cuboid volumes
    cuboid_volumes_neg = []

    for index, row in tqdm(result.iterrows(), total=((rows * columns) - 1 - columns)):
        if index >= (rows * columns) - 1 - columns:
            continue
        else:
            mean = stat.mean(
                [
                    result.loc[index, "elevation"],
                    result.loc[index + 1, "elevation"],
                    result.loc[index + columns, "elevation"],
                    result.loc[index + columns + 1, "elevation"],
                ]
            )
            if mean > 0:
                cuboid_volumes_pos.append(mean)
            else:
                cuboid_volumes_neg.append(mean)

    return sum(cuboid_volumes_pos) + (-1 * sum(cuboid_volumes_neg))


# ----------------------------------------------------------------
def clean_cross_sections(
    data_1,
    data_2,
    max_clusters=5,
    end_weight=5.0,
    dataset_weight=1e6,
    max_stddev=2,
    max_dist_tolerance=1.3,
    distribution_limit=4,
    eps=3,
    min_samples=4,
):
    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"
    data_full = pd.concat([data_1, data_2], ignore_index=True)

    data_full["cluster"] = np.nan
    norths = data_full.northing.unique()
    features = []
    for k in tqdm(norths, total=len(data_full.northing.unique()), desc="Filter Cross Sections"):
        # Create parameters

        # Break into cross sections
        cross_section = data_full[data_full["northing"] == k]
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

            if stat.stdev(df_temp.elevation) < max_stddev or sum(dist) < max_dist:
                continue
            else:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                yhat = model.fit_predict(df_index[["elevation", "easting"]])
                clusters = unique(yhat)
                for cluster in clusters:
                    # get row indexes for samples with this cluster
                    df_index["cluster"] = yhat
                    row_ix = where(yhat == cluster)

                    # find the centroid of this cluster
                    centroid_elevation = df_index[["elevation", "easting"]].to_numpy()[row_ix, 0].mean()
                    other_surface_elevation = cross_section_worthy[
                        cross_section_worthy["dataset"] != i
                    ].elevation.mean()
                    if (centroid_elevation < other_surface_elevation - distribution_limit) or (
                        centroid_elevation > other_surface_elevation + distribution_limit
                    ):
                        row_ix = list(df_index.iloc[row_ix]["elevation"].index)
                        df_index.loc[row_ix, "elevation"] = np.nan

                if df_index.elevation.isna().all():
                    # just make this the same so that we don't count this cross section
                    df_index.loc[:, "elevation"] = cross_section_worthy[cross_section_worthy["dataset"] != i].loc[
                        :, "elevation"
                    ]
                    update_values = list(df_index.index)
                    data_full.loc[update_values, "elevation"] = df_index["elevation"].values
                else:
                    # In this area begin interpolation
                    df_index.loc[:, "elevation"] = df_index.loc[:, "elevation"].interpolate(method="linear")
                    b, a = cheby1(N=3, Wn=0.94, rp=21)
                    df_index.loc[:, "elevation"] = filtfilt(b, a, df_index.loc[:, "elevation"].values)
                    update_values = list(df_index.index)
                    data_full.loc[update_values, :] = df_index

    # don't need this column anymore
    data_full = data_full.drop(["cluster"], axis=1)

    dataset_1 = data_full[data_full["dataset"] == "t1"].reset_index(drop=True)
    dataset_2 = data_full[data_full["dataset"] == "t2"].reset_index(drop=True)

    return dataset_1, dataset_2


# ----------------------------------------------------------------
def clean_cross_sections_long(
    data_1,
    data_2,
):
    """
    This function cleans the cross sections that are abnormal for comparing 2 surfaces fa apart in time.
    """

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
        if not x.empty:
            norths_to_inspect.append(i)
    for j in range(len(norths_to_inspect)):
        data_full_1 = data_full[data_full["northing"] == norths_to_inspect[j]]
        cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
        cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"].reset_index(drop=True)
        data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)

        # reshape and make new df with easting and absolute delta elevation
        data_reshaped = data[data["dataset"] == "t1"].reset_index(drop=True)
        data_reshaped["absolute_delta_elevation"] = abs(
            data_reshaped["elevation"] - data[data["dataset"] == "t2"]["elevation"].reset_index(drop=True)
        )
        data_reshaped = data_reshaped[["easting", "absolute_delta_elevation"]]
        # find when abs delta > 6
        rows_of_interest = data_reshaped[data_reshaped["absolute_delta_elevation"] > 6]
        if not np.isnan(rows_of_interest[rows_of_interest.index < (data_reshaped.index.max() // 3)].index.max()):
            # data_reshaped[data_reshaped["easting"]== data_reshaped[data_reshaped["absolute_delta_elevation"] > 6].max()["easting"]].index[0] # returns index of max elev diff
            data_small_1 = data_full_1.loc[
                data_full_1["easting"] < data_reshaped[data_reshaped["absolute_delta_elevation"] > 6].max()["easting"],
                :,
            ]  # returns all rows where
            data_small_1.loc[data_small_1["dataset"] == "t1", "elevation"] = np.array(
                data_small_1.loc[data_small_1["dataset"] == "t2", "elevation"]
            )
            data_1.loc[data_small_1.loc[data_small_1["dataset"] == "t1", :].index.tolist(), "elevation"] = np.array(
                data_small_1.loc[data_small_1["dataset"] == "t1", "elevation"]
            )

    return data_1, data_2


##################################################################################################################################

# Read in the dataframes
# data_1 = pd.read_csv("CE_36_B03-202304281615.csv")
# data_1.columns = ["easting", "northing", "elevation"]
# data_2 = pd.read_csv("CE_36_B03-202304281630.csv")
# data_2.columns = ["easting", "northing", "elevation"]
# 1716.458969372341 BCM compared to 2450.1249999999995 BCM without cleaning step

data_1 = pd.read_csv("CE_36_B03-202304292100.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202304292115.csv")
data_2.columns = ["easting", "northing", "elevation"]  # No difference


# check that the easting and northing coordinates are aligned between the rwo datasets
assert_frame_equal(data_1[["easting", "northing"]], data_2[["easting", "northing"]])
# assert_frame_equal(data_1[["elevation"]], data_2[["elevation"]]) throws error. Therefore, they are perfectly aligned.

# Clean up surfaces
data_1, data_2 = clean_cross_sections_long(
    data_1, data_2
)  # this is for long diff like 24hrs # clean_cross_sections(data_1, data_2) # this is for 15min diff

# create diff surface
data_diff = data_1[["easting", "northing"]]
data_diff["elevation"] = data_1["elevation"] - data_2["elevation"]
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

# Calculate volume
rows = x.shape[0]
columns = x.shape[1]
volume = calculate_volume(rows, columns, result)

print(f"This is how much total volume was moved in 15 minutes {volume} BCM")
