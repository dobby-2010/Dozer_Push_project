#import dependencies
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

inaccurate_northings = []

# setup function
data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]

elevation_limit = 6.0

data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

# find all initial locations for where we need to make changes
norths_to_inspect = []
for i in data_diff_full["northing"].unique():
    x = data_diff_full.loc[
        (data_diff_full["northing"] == i)
        & ((data_diff_full["elevation"] < -elevation_limit) | (data_diff_full["elevation"] > elevation_limit)),
        :,
    ]
    if not x.empty:
        norths_to_inspect.append(i)
northing_number = 0
for x in norths_to_inspect:
    location = norths_to_inspect[northing_number]

    df1 = data_full.loc[data_full["northing"] == location, :]  # Identify all points at specific northing
    cs1 = df1.loc[df1["dataset"] == "t1"]  # cross section 1
    cs2 = df1.loc[df1["dataset"] == "t2"]  # cross section 2
    df = pd.concat([cs1, cs2], ignore_index=True)  # data
    dfdiff = cs1[["easting", "northing"]].reset_index(drop=True)  # data_diff
    dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)

    # use cluster analysis to determine if any sections are
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
####################################################################################################################

    db = clustering = DBSCAN(eps=eps_best, min_samples=4).fit(X)  # use min samples = 4
    # add the cluster labels to the dfdiff dataframe
    dfdiff["cluster"] = db.labels_

    # Irterate through each cluster if any part of the cluster is outside the limit then add the northing to a list
    # set a limit for which the elevation is too great for it not to be an error
    elevation_limit = 4
    index_list = []
    data_update = cs1.reset_index(drop=True)  # T1 is updating the old surface, drop=True means that we reset the index, if we wanna use the original indexes remove this
    for cluster_number in dfdiff["cluster"].unique():
        cl1 = dfdiff[dfdiff["cluster"] == cluster_number]  # isolate all points connected to this specific cluster
        # remove previous values from index list
        index_list = []
        # find the average elevation for this specific cluster
        elevation_avg = cl1['delta_elevation'].mean()

        # check to see if this elevation is too great
        if elevation_avg * elevation_avg >= elevation_limit * elevation_limit:
            inaccurate_northings.append(location)
            #need to add the average elevations to all elevations with the corresponding cluster number 
    northing_number += 1
data_1_updated = data_1.copy()    
for northing_remove in inaccurate_northings:
    data_1_updated = data_1_updated.loc[data_1_updated['northing'] != northing_remove]

    
data_1['version'] = 'before changes'
data_1_updated['version'] = 'after changes'
data_full_updated = pd.concat([data_1, data_1_updated], ignore_index=True)
fig = go.Figure()
fig = px.scatter_3d( x=data_full_updated["easting"], y=data_full_updated["northing"], z=data_full_updated["elevation"], color=data_full_updated["version"])
fig.show()

...

