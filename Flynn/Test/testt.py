#import dependencies
# Import dependencies
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px

# Load data
data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]

# Setup parameters
elevation_limit = 6.0

# Add dataset labels
data_1["dataset"] = "t1"
data_2["dataset"] = "t2"

# Concatenate dataframes
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

# Create a copy of data_1 for updates
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

# Loop through all northings to inspect
for location in norths_to_inspect:
    df1 = data_full.loc[data_full["northing"] == location, :]
    cs1 = df1.loc[df1["dataset"] == "t1"]
    cs2 = df1.loc[df1["dataset"] == "t2"]
    dfdiff = cs1[["easting", "northing"]].reset_index(drop=True)
    dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)
    
    X = dfdiff
   
    
    #Optiizing the parameters
    # Defining the list of hyperparameters to try
    eps_list=np.arange(start=0.1, stop=10, step=0.1)
    min_sample_list=np.arange(start=2, stop=5, step=1)

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
            
                if len(np.unique(labels)) > 1: #check is lebels if greater than 1 which is has to be. IF not then likely all zeros and not useful anyway
                    sil_score_new = metrics.silhouette_score(X, labels)
                    #print("sil score new is " + str(sil_score_new))
                else:
                    continue
                if sil_score_new > sil_score:       #checks if new silhouette score is greater than previous, if so make it the greatest score. This is to find the greatest silhouse score possible and its corresponding values
                    sil_score = sil_score_new

                    eps_best = eps_trial
                    min_sample_best  = min_sample_trial
                    silhouette_scores_data = silhouette_scores_data.append(pd.DataFrame(data=[[sil_score, eps_best, min_sample_best]], columns=['Best Silhouette Score', 'Optimal EPS', 'Optimal Minimal Sample Score']))
                    #print(silhouette_coefficients)
               
                else:
                    continue
        

    db = clustering = DBSCAN(eps=eps_best, min_samples=min_sample_best).fit(X)
    dfdiff["cluster"] = db.labels_
    
    elevation_limit = 3
    data_update = cs1.reset_index(drop=True)
    
    cs1_updated = cs1.copy()
    for cluster_number in dfdiff["cluster"].unique():
        cl1 = dfdiff[dfdiff["cluster"] == cluster_number]
        elevation_avg = cl1['delta_elevation'].mean()

        if abs(elevation_avg) >= elevation_limit:
            cl2 = cs2[(cs2["dataset"] == "t2")]
            eastings_to_update = cl1[abs(cl1['delta_elevation']) >= elevation_limit]['easting'].values
            
            # Update the corresponding points in data_1_updated with values from data_2
            for easting in eastings_to_update:
                data_1_updated.loc[
                    (data_1_updated["northing"] == location) & (data_1_updated["easting"] == easting),
                    "elevation"
                ] = cl2[cl2['easting'] == easting]['elevation'].values[0]

# Add version labels
data_1['version'] = 'data_1 before changes'
data_1_updated['version'] = 'data_1 after changes'
data_2['version'] = 'data_2'

# Concatenate and visualize updated data
data_full_updated = pd.concat([data_1, data_1_updated, data_2], ignore_index=True)
fig = go.Figure()
fig = px.scatter_3d(x=data_full_updated["easting"], y=data_full_updated["northing"], z=data_full_updated["elevation"], color=data_full_updated["version"])
fig.show()
