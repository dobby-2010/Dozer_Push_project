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
elevation_limit_org = 5
# Read and preprocess the data
data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]

data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

# Identify northings to inspect
norths_to_inspect = []
for i in data_diff_full["northing"].unique():
    x = data_diff_full.loc[
        (data_diff_full["northing"] == i)
        & ((data_diff_full["elevation"] < - elevation_limit_org) | ((data_diff_full["elevation"] > elevation_limit_org))),
        :,
    ]
    if not x.empty:
        norths_to_inspect.append(i)

# Specify a location of interest

l2 = 7516917.99
l3 = 7516918.99
l4 = 7516919.99
l5 = 7516920.99
l6 = 7516921.99
l7 = 7516922.99
l8 = 7516923.99
l9 = 7516924.99
l10 = 7516925.99
l11 = 7516926.99
l12 = 7516927.99
l13 = 7516909.99
l14 = 7516910.99
l15 = 7516911.99
l16 = 7516915.99
l17 = 7516930.99
l18 = 7516931.99
l19 = 7516932.99
l20 = 7516933.99
l21 = 7516934.99
l22 = 7516747.99
l23 = 7516738.99

location = l5

df1 = data_full[data_full["northing"] == location]
cs1 = df1[df1["dataset"] == "t1"].reset_index(drop=True)
cs2 = df1[df1["dataset"] == "t2"].reset_index(drop=True)
df = pd.concat([cs1, cs2], ignore_index=True)
dfdiff = cs1[["easting", "northing"]].reset_index(drop=True)
dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)

# Visualize the data
fig = go.Figure()
fig = px.scatter(x=df1.easting, y=df1.elevation, color=df1['dataset'])
fig.update_layout(title=str(location),
                  xaxis_title='Easting (m)',
                  yaxis_title='Elevation (m)')
fig.show()

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
                #print("silscore is " + str(sil_score))
                eps_best = eps_trial
                min_sample_best  = min_sample_trial
                silhouette_scores_data = silhouette_scores_data.append(pd.DataFrame(data=[[sil_score, eps_best, min_sample_best]], columns=['Best Silhouette Score', 'Optimal EPS', 'Optimal Minimal Sample Score']))
                #print(silhouette_coefficients)
               
        else:
            continue
        

db = clustering = DBSCAN(eps=1.2, min_samples=3).fit(X)  #use min samples = 4
        #storing the labels formed by the DBSCAN

dfdiff["cluster"] = db.labels_


fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.update_layout(title="DBSCAN")
fig.show()


    # Irterate through each cluster if any part of the cluster is outside the limit then add the northing to a list
    # set a limit for which the elevation is too great for it not to be an error
elevation_limit = 4
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
        
        # Update the corresponding points in cs1_updated with values from cs2
        cs1_updated.loc[cs1_updated["easting"].isin(eastings_to_update), "elevation"] = cl2[cl2['easting'].isin(eastings_to_update)]['elevation'].values


cs1['version'] = 'before changes'
cs1_updated['version'] = 'after changes'
cs2['version'] = 'data 2'
data_full_updated = pd.concat([cs1, cs1_updated, cs2], ignore_index=True)

fig = go.Figure()
fig = px.scatter(x=data_full_updated["easting"],  y=data_full_updated["elevation"], color=data_full_updated["version"])

fig.update_layout(title="updated at"+str(location),
                  xaxis_title='Easting (m)',
                  yaxis_title='Elevation (m)')
fig.show()

...
   
