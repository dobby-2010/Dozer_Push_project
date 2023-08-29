import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

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
 
#several different locations which should work
       
l0 = 7516846.99       #7516846.99 is an immediate success     
l1 = 7516847.99
l2 = 7516918.99
l3 = 7516921.99


location = l2


df1 = data_full[data_full["northing"] == l2]
cs1 = df1[df1["dataset"] == "t1"].reset_index(drop=True)        #cross section 1
cs2 = df1[df1["dataset"] == "t2"].reset_index(drop=True)        #cross section 2
df = pd.concat([cs1, cs2], ignore_index=True)                   #data
dfdiff = cs1[["easting", "northing"]].reset_index(drop=True)    #data_diff
dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)

#checkout what the data looks like 
fig = go.Figure()
fig = px.scatter(x=df1.easting, y=df1.elevation, color=df1['dataset'])
fig.update_layout(title=str(location),
                  xaxis_title='Easting (m)',
                  yaxis_title='Elevation (m)')
fig.show() 

#search for all values in this specfic easting for which the delta_elevation is greater than a specfied value: elevation_limit
elevation_limit = 4   
q_data = dfdiff[(dfdiff["delta_elevation"] > elevation_limit) | (dfdiff["delta_elevation"] < -elevation_limit)]

################################################################################################################
#Need to make the eastings which have been deemeed to be inaccurate, need to make data_2 (old) be 
#moved to data_1 (new). This will make the elevation be 0 and not show up in the elevation data results.
###############################################################################################################

#find greatest easting value within this range of data
greatest_east = q_data['easting'].max()
   
criteria = data_2['easting'] < greatest_east

#make the elevation between these points be 0. This will be done by updatig the previous surface to the values of the new surface
data_2_updated = data_2[~criteria]
data_1_updated = data_1[~criteria]

#update the data with the easting range values omiited   
data_full = pd.concat([data_1_updated, data_2_updated], ignore_index=True)
df1_updated = data_full[data_full["northing"] == l2]
cs1_updated = df1_updated[df1_updated["dataset"] == "t1"].reset_index(drop=True)    #cross section 1
cs2_updated = df1_updated[df1_updated["dataset"] == "t2"].reset_index(drop=True)    #cross section 2
df1_updated = pd.concat([cs1_updated, cs2_updated], ignore_index=True)               #data    
fig = go.Figure()
fig = px.scatter(x=df1_updated.easting, y=df1_updated.elevation, color=df1_updated["dataset"])
fig.update_layout(title=str(location),
                  xaxis_title='Easting (m)',
                  yaxis_title='Elevation (m)')
fig.show()     

  
...