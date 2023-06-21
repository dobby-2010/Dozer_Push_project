# import dependencies
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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN

from matplotlib import pyplot

from sklearn import metrics



l1 = 7516846.99         #several different locations which should work
l2 = 7516847.99
l3 = 7516918.99
l4 = 7516921.99


data_1 = pd.read_csv("CE_36_B03-202305030845.csv")
data_1.columns = ["easting", "northing", "elevation"]
data_2 = pd.read_csv("CE_36_B03-202305040800.csv")
data_2.columns = ["easting", "northing", "elevation"]
data_1["dataset"] = "t1"
data_2["dataset"] = "t2"
data_full = pd.concat([data_1, data_2], ignore_index=True)
data_diff_full = data_1[["easting", "northing"]]
data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]
fig = go.Figure()
fig = px.scatter_3d(x=data_full.easting, y=data_full.northing, z=data_full.elevation, color=data_full.dataset)
#fig.show()

#where are some good exampels
norths_to_inspect = []
for i in data_diff_full["northing"].unique():
    x = data_diff_full.loc[
        (data_diff_full["northing"] == i)
        & ((data_diff_full["elevation"] < -6) | ((data_diff_full["elevation"] > 6))),
        :,
    ]
    if not x.empty:
        norths_to_inspect.append(i)

data_full["cluster"] = np.nan
data_full_1 = data_full[data_full["northing"] == l4]
cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"].reset_index(drop=True)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)
data_diff = cross_section_t1[["easting", "northing"]].reset_index(drop=True)
data_diff["delta_elevation"] = cross_section_t1["elevation"].reset_index(drop=True) - cross_section_t2[
    "elevation"
].reset_index(drop=True)
cross_section_worthy = data_full_1[
    data_full_1["easting"].isin(list(data_diff[data_diff["delta_elevation"] != 0].easting))
]
cross_section_worthy_1 = cross_section_worthy[cross_section_worthy["dataset"] == "t1"].reset_index(drop=True)
cross_section_worthy_2 = cross_section_worthy[cross_section_worthy["dataset"] != "t1"].reset_index(drop=True)



##################################################################################################################################################
data = data_diff
#data = cross_section_worthy_1[[]]
#pca = PCA()
#spca_data = pca.fit_transform(data)
#pca_data = pd.DataFrame(pca_data, columns=["pc"+str(i+1) for i in range(len(data.columns))])
#print("pca.explained variance ratio:\n ", " ".join(map("{:.3f}".format, pca.explained_variance_ratio_)))


epsilon = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.25,1.5,1.75, 2,2.25,2.5,2.75, 3,3.25,3.5,3.75, 4]
min_samples = [2,3,4,5]


sil_avg = []
max_value = [0,0,0,0]

for i in epsilon:
    for j in min_samples:
        print(i,j)
        db = DBSCAN(eps =3, min_samples=j).fit(data)
        #cluster_labels=dbscan.fit_predict(data) 
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(labels)

        silhouette_avg = metrics.silhouette_score(data, labels)
        if silhouette_avg > max_value[3]:
            max_value=(i, j, n_clusters_, silhouette_avg)
        sil_avg.append(silhouette_avg)

print("epsilon=", max_value[0], 
      "\nmin_sample=", max_value[1],
      "\nnumber of clusters=", max_value[2],
      "\naverage silhouette score= %.4f" % max_value[3])

