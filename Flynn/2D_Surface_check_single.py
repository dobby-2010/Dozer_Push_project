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



####################################################################################################################################################################################
### 
### This will be able to idenfity areas which are unreasonible in terms of depth/height of cut/dig
###
################################################################################################################################################################################################


#import and read csv file
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
#some good locations
l1 = 7516856.99
l2 = 7516918.99
l3 = 7516919.99


#select a location
location = l3



df1 = data_full[data_full["northing"] == location]
cs1 = df1[df1["dataset"] == "t1"]  #cross section 1
cs2 = df1[df1["dataset"] == "t2"]    #cross section 2
df = pd.concat([cs1, cs2], ignore_index=True)               #data
dfdiff = cs1[["easting", "northing"]].reset_index(drop=True) #data_diff
dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)

fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation)

fig.update_layout(title=location)
fig.show()
    
fig = go.Figure()
fig = px.scatter(
        x=df1["easting"],  y=df1["elevation"], color=df1["dataset"])
   
fig.update_layout(title=7516918.99)
fig.show()
   







#############################################################################################################################
# AFTER CLEANING UP THE SURFACE
################################################################################################################################################################
#Import the cluster analysis
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import DBSCAN

#using DBSCAN as a cluster technique

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


########################################################################################################################################################################################################################################################################### #
#Irterate through each cluster
###################################################################################################################################################################################################################################################################
#set a limit for which the elevation is too great for it not to be an error
elevation_limit = 6
index_list = []
data_update = cs2.reset_index(drop=True) #T2 is updating the older surface, drop=True means that we reset the index, if we wanna use the original indexs remove this 
for cluster_number in dfdiff["cluster"].unique():
    cl1 = dfdiff[dfdiff["cluster"] == cluster_number]   #isolate all points connected to this specific cluster
    #find the average elevation for this specific cluster
    elevation_avg = cl1['delta_elevation'].mean()
    #check to see if this elevtion is too great
    if elevation_avg*elevation_avg >= elevation_limit*elevation_limit:
        locations_1 = cl1.iloc[:,[0,1]]
        #dataframe of locations for which need there elevation wiped
        for easting_change in locations_1['easting']:
            for northing_change in locations_1["northing"]:
           #  #this will find the index locaions for each easing and nothing found above
                index_change = data_update[data_update["easting"] == easting_change]
                index_change_1 = index_change[index_change['northing'] == northing_change]
                index_list.append(index_change_1.index.tolist())
        unique_index_list = []
        #print(index_list)
        for index_value in index_list:
            data_update.loc[index_value, "elevation"] = 0
            
#get rid of any duplicates     
index_list = sorted(list(set([item for sublist in index_list for item in sublist])))

#We need the values either side of the points which have been removed.#lets say were looking for 3 points either side
varaince = 3 #how many points either side
min_update_value = min(index_list) - varaince
max_update_value = max(index_list) + varaince


#Need to use the cheby filter remove the 'incorrect' cluster 
from tqdm import tqdm

cross_section_t1 = data_update[data_update["dataset"] == "t2"].reset_index(drop=True) # this is a single cross section of a single surface
cross_section_t1_new = cross_section_t1.copy()
N = np.arange(2, 11, 1) # possible values of first parameter
Wn = np.linspace(0.01, 0.99, 50) # possible values of second parameter
grid = np.ones((len(N), len(Wn))) # this is where I will store the values from my loss function
for i, row in enumerate(N): # loop through all possible permutations of the 2 parameters
    for j, item in enumerate(Wn):
        b, a = cheby1(N=row, Wn=item, rp=21) # parameters define a & b
        cross_section_t1_new.loc[:, "elevation"] = pd.DataFrame(
            filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
        ).loc[:, 0] # a and b determine the smoothing of elevations
        delta_elevation = cross_section_t1_new["elevation"] - cross_section_t2["elevation"] # compare new elevations to the original
        loss_fn = abs(delta_elevation).sum() # my loss function here is just the sum of the difference
        grid[i, j] = loss_fn # store this combination of the hyperparameter's loss value in my grid

grid_search = np.unravel_index(grid.argmin(), grid.shape) # find the index of the minimum loss (want to minimize loss)
optimised_N = N[grid_search[0]] # this is the optimized param N
optimised_Wn = Wn[grid_search[1]] # this is the optimized param Wn

b, a = cheby1(N=optimised_N, Wn=optimised_Wn, rp=21) # Let's recompute the elevations using these optimised a & b
cross_section_t1.loc[:, "elevation"] = filtfilt(b, a, cross_section_t1.loc[:, "elevation"].values)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
fig = go.Figure()
fig = px.scatter(x=data.easting, y=data.elevation, color=data.dataset)
fig.update_layout(title="Cheby1") # display the resulting smoothed elevations
fig.show()





fig = go.Figure()
fig = px.scatter(x=data_update.easting, y=data_update.elevation)

fig.update_layout(title="update")
fig.show()
    


      
       
