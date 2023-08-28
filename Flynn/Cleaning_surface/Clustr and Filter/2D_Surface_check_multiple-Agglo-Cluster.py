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
from numpy import unique
from numpy import where
from pandas.testing import assert_frame_equal
from sklearn import metrics

from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import DBSCAN


####################################################################################################################################################################################
### 
### This will be able to idenfity areas which are unreasonible in terms of depth/height of cut/dig
###
################################################################################################################################################################################################


#Locaiton 2
data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]

#location 1
data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]

#find elevation change
data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]         #elevation change dataset

#find all locations where elevation change is greater than 6m
data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

norths_to_inspect = []
for i in data_diff_full["northing"].unique():
    x = data_diff_full.loc[
        (data_diff_full["northing"] == i)
        & ((data_diff_full["elevation"] < -8) | ((data_diff_full["elevation"] > 8))),
        :,
    ]
    if not x.empty:
        norths_to_inspect.append(i)

################################################################################################################################
#BEFORE SMOOTHING THE SURFACE
##############################################################################################################
#for testing purposes 7516918.99
 
i = 0
# for iterating through the locaitons
for x in norths_to_inspect:
    if i > 6:
        break
      
    location = norths_to_inspect[i]
        
    df1 = data_full[data_full["northing"] == x]
    cs1 = df1[df1["dataset"] == "t1"].reset_index(drop=True)    #cross section 1
    cs2 = df1[df1["dataset"] == "t2"].reset_index(drop=True)    #cross section 2
    df = pd.concat([cs1, cs2], ignore_index=True)               #data
    dfdiff = cs1[["easting", "northing"]].reset_index(drop=True) #data_diff
    dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)

  
    #show the two surfaces against each other
    fig = go.Figure()
    fig = px.scatter(
        x=df1["easting"],  y=df1["elevation"], color=df1["dataset"]
    )
   
    fig.update_layout(title=location)
    fig.show()
      
    fig = go.Figure()
    fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation)
    fig.update_layout(title=location)
    fig.show()
    
    
    #using Angglomerative Cluster as a cluster technique

    X = dfdiff

    from sklearn.cluster import AgglomerativeClustering     

    silhouette_coefficients = []
    
    for k in tqdm(range(2, 10), desc="Clustering"):
        ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
         #print(ac)
        score = silhouette_score(dfdiff[["easting", "delta_elevation"]], ac.labels_)
        silhouette_coefficients.append(score)
  
    clusters = range(2, 10)[silhouette_coefficients.index(max(silhouette_coefficients))]


    ac = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
    dfdiff["cluster"] = ac.labels_
    fig = go.Figure()
    fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
    fig.update_layout(title="Agglomerative Clustering")
    fig.show()
    
    
    print(i)
    i = 1 + i




#############################################################################################################################
# AFTER CLEANING UP THE SURFACE
################################################################################################################################################################
#Import the cluster analysis
