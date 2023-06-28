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
        
l0 = 7516846.99         #several different locations which should work
l1 = 7516847.99
l2 = 7516918.99
l3 = 7516921.99

# 7516846.99 is an immediate success
df1 = data_full[data_full["northing"] == l2]
cs1 = df1[df1["dataset"] == "t1"].reset_index(drop=True)    #cross section 1
cs2 = df1[df1["dataset"] == "t2"].reset_index(drop=True)    #cross section 2
df = pd.concat([cs1, cs2], ignore_index=True)               #data
dfdiff = cs1[["easting", "northing"]].reset_index(drop=True) #data_diff
dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)


#import cluster methods
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation


############################################################################################################################################
"""
Agglomerative Clustering

The agglomerativeClustering object performs a hierarchical clustering using a bottom up apporoach:each observation starts in its own cluster,
and clusters are successivley merged together. The linkage criterai determiens the metrix used for the merge strategy

USECASE: Many clusters, possibly connectivity consraints, non Euclidean distances

https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
"""
#############################################################################################################################################

silhouette_coefficients = []

for k in tqdm(range(2, 10), desc="Clustering"):
    ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
    score = silhouette_score(dfdiff[["easting", "delta_elevation"]], ac.labels_)
    silhouette_coefficients.append(score)
    
fig = go.Figure()
fig = px.scatter(x=range(2, 10), y=silhouette_coefficients)
fig.update_layout(title="Optimal Silhouette Coefficients-Agglomerative Clustering")
#fig.show()
clusters = range(2, 10)[silhouette_coefficients.index(max(silhouette_coefficients))]


ac = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit(dfdiff[["easting", "delta_elevation"]])
dfdiff["cluster"] = ac.labels_
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.update_layout(title="Agglomerative Clustering")
#fig.show()


############################################################################################################################################
"""
Gaussian Mixtures

A Gaussian mixture model is a probabilisitic model that assumes all the data points are generated from a mixture of finitie number of gaussian
distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorperate information about the
covariance stucuture of the data as well as the centres of latent gaussians,

USECASE: Flat Geometry, good for density estimation, Inductive

GuassianMixture(n_components).fit

https://scikit-learn.org/stable/modules/mixture.html#mixture

"""
############################################################################################################################################

silhouette_coefficients = []

for k in tqdm(range(2, 10), desc="Clustering"):
    gm = GaussianMixture(n_components=k, random_state=0).fit(dfdiff[["easting", "delta_elevation"]])
    score = silhouette_score(dfdiff[["easting", "delta_elevation"]], gm.predict(dfdiff[["easting", "delta_elevation"]]))
    silhouette_coefficients.append(score)

   
fig = go.Figure()
fig = px.scatter(x=range(2, 10), y=silhouette_coefficients)
fig.update_layout(title="Optimal Silhouette Coefficients-Guassian Mixture")
#fig.show()
clusters = range(2, 10)[silhouette_coefficients.index(max(silhouette_coefficients))]

gmm = GaussianMixture(n_components=clusters, random_state=0).fit(dfdiff[["easting", "delta_elevation"]])
dfdiff["cluster"] = gmm.predict(dfdiff[["easting", "delta_elevation"]])
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.update_layout(title="Gaussian Mixture")
#fig.show()



############################################################################################################################################
"""
DBSCAN

The DBSCAN algoithm views clusters as areas of high density seperated by areas of low density.

USECASE: Non-flat geometry, uneven cluster sizes, outlier removal, transductive

https://scikit-learn.org/stable/modules/clustering.html#dbscan

"""
############################################################################################################################################
# Let us try gaussian mixture models clustering
# let us attempt clustering with variable number of clusters and see what number to use

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
#(silhouette_scores_data)

#fig = go.Figure()
#fig = px.scatter(x=silhouette_scores_data.loc[:,"Optimal Minimal Sample Score"], y=silhouette_scores_data.loc[:,"Best Silhouette Score"])
#fig = px.scatter(x=silhouette_scores_data.loc[:,"Optimal EPS"], y=silhouette_scores_data.loc[:,"Best Silhouette Score"])
#fig.update_layout(title="Sample Score vs Silhouette Score")
#fig.show()

#test

#db = clustering = DBSCAN(eps=eps_best, min_samples=min_sample_best).fit(X)  
db = clustering = DBSCAN(eps=eps_best, min_samples=4).fit(X)  #use min samples = 4
        #storing the labels formed by the DBSCAN
dfdiff["cluster"] = db.labels_
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.update_layout(title="DBSCAN")
#fig.show()

###########################################################################################################################################
"""
Affinity Propagation

Perform affinity propagation clustering of data

USECASE: Many Clusters, uneven cluster size, non-flat geometry, inductive

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#

"""
############################################################################################################################################



silhouette_coefficients = []
'''
for k in tqdm(range(2, 200), desc="Clustering"):
    afprop = AffinityPropagation(preference=-50, max_iter=k).fit(dfdiff[["easting", "delta_elevation"]])
    score = silhouette_score(dfdiff[["easting", "delta_elevation"]], afprop.predict(dfdiff[["easting", "delta_elevation"]]))
    silhouette_coefficients.append(score)

   
fig = go.Figure()
fig = px.scatter(x=range(2, 10), y=silhouette_coefficients)
fig.update_layout(title="Optimal Silhouette Coefficients-Guassian Mixture")
fig.show()

#clusters = range(2, 10)[silhouette_coefficients.index(max(silhouette_coefficients))]
    
afprop = AffinityPropagation().fit(dfdiff[["easting", "delta_elevation"]])
dfdiff["cluster"] = afprop.predict(dfdiff[["easting", "delta_elevation"]])
score = silhouette_score(dfdiff[["easting", "delta_elevation"]], afprop.predict(dfdiff[["easting", "delta_elevation"]]))
silhouette_coefficients.append(score)
print(silhouette_score)
fig = go.Figure()
fig = px.scatter(x=dfdiff.easting, y=dfdiff.delta_elevation, color=dfdiff.cluster)
fig.update_layout(title="Affinity Propagation")
fig.show()

'''
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Standardize the data
print(dfdiff[["easting", "delta_elevation"]])
scaler = StandardScaler()
data_scaled = dfdiff[["easting", "delta_elevation"]]

# Define the parameter grid to search
param_grid = {
    'damping': np.linspace(0.5, 0.1, 1.0),
    'preference': np.linspace(-10, 1, 21)
}

# Calculate the silhouette scores for each combination of parameters
best_score = -1
best_params = None
for damping in param_grid['damping']:
    for preference in param_grid['preference']:
        # Create an instance of the AffinityPropagation algorithm with current parameters
        affinity_propagation = AffinityPropagation(damping=damping, preference=preference)
        
        # Fit the data to the algorithm
        affinity_propagation.fit(data_scaled)
        
        # Get the cluster labels
        labels = affinity_propagation.labels_
        
        print(data_scaled)
        print(labels)
        
        # Calculate the silhouette score
        score = silhouette_score(data_scaled, labels)
        
        # Update the best score and parameters if the current score is higher
        if score > best_score:
            best_score = score
            best_params = {'damping': damping, 'preference': preference}

# Print the best parameters and silhouette score
print("Best Parameters: ", best_params)
print("Best Silhouette Score: ", best_score)

# Find the outlier sections based on the z-axis using the optimal parameters
affinity_propagation = AffinityPropagation(**best_params)
affinity_propagation.fit(data_scaled)
labels = affinity_propagation.labels_
unique_labels = np.unique(labels)
outlier_sections = []
for label in unique_labels:
    z_values = data[:, 2][labels == label]
    if np.std(z_values) > 2:  # Modify the threshold as needed
        outlier_sections.append(label)

# Print the outlier sections
print("Outlier Sections:", outlier_sections)





