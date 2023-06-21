############################################################################################################################################
"""
Agglomerative Clustering

The agglomerativeClustering object performs a hierarchical clustering using a bottom up apporoach:each observation starts in its own cluster,
and clusters are successivley merged together. The linkage criterai determiens the metrix used for the merge strategy

USECASE: Many clusters, possibly connectivity consraints, non Euclidean distances
"""
#############################################################################################################################################
 #import dependencies
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
from scipy.spatial import ConvexHull
from kneed import KneeLocator
from tqdm import tqdm
from numpy import unique
from numpy import where
from pandas.testing import assert_frame_equal


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
        (data_diff_full["northing"] == i)
        & ((data_diff_full["elevation"] < -6) | ((data_diff_full["elevation"] > 6))),
        :,
    ]
    if not x.empty:
        norths_to_inspect.append(i)

# 7516846.99 is an immediate success
df1 = data_full[data_full["northing"] == 7516917.99]
cs1 = df1[df1["dataset"] == "t1"].reset_index(drop=True)    #cross section 1
cs2 = df1[df1["dataset"] == "t2"].reset_index(drop=True)    #cross section 2
df = pd.concat([cs1, cs2], ignore_index=True)               #data
dfdiff = cs1[["easting", "northing"]].reset_index(drop=True) #data_diff
dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)


#import cluster methods
from sklearn.cluster import AgglomerativeClustering 


silhouette_coefficients = []

for k in tqdm(range(2, 10), desc="Clustering"):
    ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
    print(ac)
    score = silhouette_score(dfdiff[["easting", "delta_elevation"]], ac.labels_)
    silhouette_coefficients.append(score)
  
fig = go.Figure()
fig = px.scatter(x=range(2, 10), y=silhouette_coefficients)
fig.update_layout(title="Optimal Silhouette Coefficients-Agglomerative Clustering")
fig.show()
clusters = range(2, 10)[silhouette_coefficients.index(max(silhouette_coefficients))]


ac = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
dfdiff["cluster"] = ac.labels_
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.update_layout(title="Agglomerative Clustering")
fig.show()