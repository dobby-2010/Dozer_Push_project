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
from sklearn import metrics




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
    elevation_limit=6.0
):
    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"
   
    #combine data
    data_full = pd.concat([data_1, data_2], ignore_index=True)
    data_diff_full = data_1[["easting", "northing"]]
    data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

    #create a copy of data_1 this will be able to be used to update and then compare with data_1 
    data_1_updated = data_1.copy()

    # Find all initial locations for where we need to make changes
    norths_to_inspect = []
    for i in data_diff_full["northing"].unique():
        x = data_diff_full.loc[
            (data_diff_full["northing"] == i)
            & ((data_diff_full["elevation"] < -elevation_limit) | (data_diff_full["elevation"] > elevation_limit)),
            :,
        ]
        if not x.empty:
            norths_to_inspect.append(i)

    #loop through all clusters that are considered to be inaccurate
    for x in norths_to_inspect:

        df1 = data_full.loc[data_full["northing"] == x, :]  # Identify all points at specific northing
        cs1 = df1.loc[df1["dataset"] == "t1"]  # cross section 1
        cs2 = df1.loc[df1["dataset"] == "t2"]  # cross section 2
        df = pd.concat([cs1, cs2], ignore_index=True)  # data
        dfdiff = cs1[["easting", "northing"]].reset_index(drop=True)  # data_diff
        dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)

        # use cluster analysis to split line into sections 
        # using DBSCAN as a cluster technique
        X = dfdiff

        # Optiizing the parameters
        # Defining the list of hyperparameters to try
        eps_list = np.arange(start=0.1, stop=10, step=0.1)
        min_sample_list = np.arange(start=2, stop=5, step=1)

        # setup the silhouette list
        silhouette_coefficients = []
        eps_coefficients = []
        min_samp_list = []

    #####################################################################################################################
    # This section is for the dbscan parameters


        # create dataframe to store the silhouette parameters for each trial"
        silhouette_scores_data = pd.DataFrame()
        sil_score = 0  # sets the first sil score to zero
        for eps_trial in eps_list:
            for min_sample_trial in min_sample_list:
            
                clustering = DBSCAN(eps=eps_trial, min_samples=min_sample_trial).fit(X)
                # storing the labels formed by the DBSCAN
                labels = clustering.labels_

                # measure the performance of DBSCAN algo
                # identifying which points make up our 'core points'
                core_samples = np.zeros_like(labels, dtype=bool)
                core_samples[clustering.core_sample_indices_] = True

                # Calculating "the number of clusters"
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters_ > 0:

                    if len(np.unique(labels)) > 1:  # check if labels is greater than 1 which it has to be. If not, then likely all zeros and not useful anyway
                        sil_score_new = metrics.silhouette_score(X, labels)

                    else:
                        continue

                    if sil_score_new > sil_score:  # checks if new silhouette score is greater than previous, if so make it the greatest score. This is to find the greatest silhouette score possible and its corresponding values
                        sil_score = sil_score_new
                        eps_best = eps_trial
                        min_sample_best = min_sample_trial
                        silhouette_scores_data = silhouette_scores_data.append(
                            pd.DataFrame(data=[[sil_score, eps_best, min_sample_best]], columns=['Best Silhouette Score', 'Optimal EPS', 'Optimal Minimal Sample Score']))

                    else:
                        continue

        db = clustering = DBSCAN(eps=eps_best, min_samples=min_sample_best).fit(X)  # use min samples = 4
        # add the cluster labels to the dfdiff dataframe
        dfdiff["cluster"] = db.labels_

        # Irterate through each cluster if any part of the cluster is outside the limit then add the northing to a list
        # set a limit for which the elevation is too great for it not to be an error
        elevation_limit = 3
        index_list = []
        data_update = cs1.reset_index(drop=True)  # T1 is updating the old surface, drop=True means that we reset the index, if we wanna use the original indexes remove this

    
        cs1_updated = cs1.copy()
        # Iterate through each cluster and check if the average elevation is too great
        for cluster_number in dfdiff["cluster"].unique():
            cl1 = dfdiff[dfdiff["cluster"] == cluster_number]
            elevation_avg = cl1['delta_elevation'].mean()

            if abs(elevation_avg) >= elevation_limit:

                # Find the corresponding cluster in data_2
                cl2 = cs2[(cs2["dataset"] == "t2")]
                eastings_to_update = cl1[abs(cl1['delta_elevation']) >= elevation_limit]['easting'].values
            
                # Update the corresponding points in data_1_updated with values from data_2
                for easting in eastings_to_update:
                    print(easting)
                    data_1_updated.loc[(data_1_updated["northing"] == x) & (data_1_updated["easting"] == easting), "elevation"] = cl2[cl2['easting'] == easting]['elevation'].values[0]

    dataset_1 = data_1_updated
    dataset_2 = data_2

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
data_1, data_2 = clean_cross_sections(
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
