############################################################################################################################################
"""
DBSCAN
For DBscan a s

USECASE: Non-flat geometry, uneven cluster sizes, outlier removal, transductive
"""
############################################################################################################################################
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
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering 
from sklearn import metrics


data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]


data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]


#to find locations which have uncertaintys
norths_to_inspect = []
for i in data_diff_full["northing"].unique():
    x = data_diff_full.loc[
        (data_diff_full["northing"] == i)
        & ((data_diff_full["elevation"] < -6) | ((data_diff_full["elevation"] > 6))),
        :,
    ]
    if not x.empty:
        norths_to_inspect.append(i)
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


l1 = 7516846.99         #several different locations which should work
l2 = 7516847.99
l3 = 7516918.99
l4 = 7516921.99

# 7516846.99 is an immediate success
df1 = data_full[data_full["northing"] == l4]
cs1 = df1[df1["dataset"] == "t1"].reset_index(drop=True)    #cross section 1
cs2 = df1[df1["dataset"] == "t2"].reset_index(drop=True)    #cross section 2
df = pd.concat([cs1, cs2], ignore_index=True)               #data
dfdiff = cs1[["easting", "northing"]].reset_index(drop=True) #data_diff
dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)
cross_section_worthy = df1[df1["easting"].isin(list(dfdiff[dfdiff["delta_elevation"] != 0].easting))]
cross_section_worthy_1 = cross_section_worthy[cross_section_worthy["dataset"] == "t1"].reset_index(drop=True)

####################################################################################################################################################################################################################################################
X = dfdiff

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
                #print(silscore)
                #print("eps value is",eps_best)
                #print("min sample trial is",min_sample_best)
        else:
            continue

#computing "the silhouette score"
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels))
print(silhouette_scores_data)

#fig = go.Figure()
#fig = px.scatter(x=silhouette_scores_data.loc[:,"Optimal Minimal Sample Score"], y=silhouette_scores_data.loc[:,"Best Silhouette Score"])
#fig = px.scatter(x=silhouette_scores_data.loc[:,"Optimal EPS"], y=silhouette_scores_data.loc[:,"Best Silhouette Score"])
#fig.update_layout(title="Sample Score vs Silhouette Score")
#fig.show()

#test

db = clustering = DBSCAN(eps=5, min_samples=3).fit(X)
        #storing the labels formed by the DBSCAN
dfdiff["cluster"] = db.labels_
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.update_layout(title="DBSCAN")
fig.show()






'''

for k in tqdm(range(2, 10), desc="Clustering"):
    ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
    db = DBSCAN(eps=3, min_samples=4).fit(dfdiff[["easting", "delta_elevation"]])
    acscore = silhouette_score(dfdiff[["easting", "delta_elevation"]], ac.labels_)
    dbscore =  db.labels_
    print(acscore)
    print(dbscore)

features = cross_section_worthy_1[["elevation", "easting"]].to_numpy()
# define the model
model = DBSCAN(eps=3, min_samples=4)
# fit model and predict clusters
yhat = model.fit_predict(cross_section_worthy_1[["elevation", "easting"]])
# retrieve unique clusters
clusters = unique(yhat)


for k in tqdm(range(2, 10), desc="Clustering"):
    gm = DBSCAN(eps=3, min_samples=4).fit(dfdiff[["easting", "delta_elevation"]])
    #score = silhouette_score(dfdiff[['easting', 'delta_elevation']], gm.fit_predict(dfdiff[["elevation", "delta_easting"]]))
    #silhouette_coefficients.append(score)   
    
#fig = go.Figure()
fig = px.scatter(x=range(2, 10), y=silhouette_coefficients)
fig.update_layout(title="Optimal Silhouette Coefficients-DBSCAN")
#fig.show()
clusters = range(2, 10)[silhouette_coefficients.index(max(silhouette_coefficients))]

gmm = DBSCAN(eps=3, min_samples=4).fit(dfdiff[["easting", "delta_elevation"]])
dfdiff["cluster"] = gmm.fit_predict(dfdiff[["easting", "delta_elevation"]])
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.update_layout(title="DBSCAN")
#fig.show()

'''