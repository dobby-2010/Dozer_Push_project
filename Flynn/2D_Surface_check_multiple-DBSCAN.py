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
    
    
    #using DBSCAN as a cluster technique

    X = dfdiff



    #Optiizing the parameters
    # Defining the list of hyperparameters to try
    eps_list=np.arange(start=0.1, stop=10, step=0.1)   #EPS is the maximum distance between two samples for one to be considered as in the neighborhood
    min_sample_list=np.arange(start=2, stop=7, step=1) #min_samples is the number of samples to be considered as in the neighborhood for a point to be considered as a core ppoint
    #setup the silhouette list
    silhouette_coefficients = []
    eps_coefficients = []
    min_samp_list = []
    #create dataframe to store the silhouette parameters for each trial"
    silhouette_scores_data=pd.DataFrame()
    sil_score= 0  #sets the first sil score to zero
    for eps_trial in eps_list:
        for min_sample_trial in min_sample_list:
        

            clustering = DBSCAN(eps=eps_trial, min_samples=min_sample_trial).fit(X)
            #storing the labels formed by the DBSCAN
            labels = clustering.labels_ 
       
            #measure the peformace of dbscan algo
            #idenfitying which points make up our 'core points'
            core_samples = np.zeros_like(labels, dtype=bool)
            core_samples[clustering.core_sample_indices_] = True
        

            #Calculating "the number of clusters"
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_ > 0:
                #print("the number of clusters are",n_clusters_)
                if len(np.unique(labels)) > 1: #check is lebels if greater than 1 which is has to be. IF not then likely all zeros and not useful anyway
                    sil_score_new = metrics.silhouette_score(X, labels)
                else:
                    continue
                if sil_score_new > sil_score:       #checks if new silhouette score is greater than previous, if so make it the greatest score. This is to find the greatest silhouse score possible and its corresponding values
                    silscore = sil_score_new
                    eps_best = eps_trial
                    min_sample_best  = min_sample_trial
                    silhouette_scores_data = silhouette_scores_data.append(pd.DataFrame(data=[[silscore, eps_best, min_sample_best]], columns=['Best Silhouette Score', 'Optimal EPS', 'Optimal Minimal Sample Score']))
                    
            else:
                continue
        

    db = clustering = DBSCAN(eps=eps_best, min_samples=min_sample_trial).fit(X)  #use min samples = 4
        #storing the labels formed by the DBSCAN
    dfdiff["cluster"] = db.labels_
    fig = go.Figure()
    fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
    fig.update_layout(title="DBSCAN")
    fig.show()
    
    
    print(i)
    i = 1 + i




#############################################################################################################################
# AFTER CLEANING UP THE SURFACE
################################################################################################################################################################
#Import the cluster analysis
