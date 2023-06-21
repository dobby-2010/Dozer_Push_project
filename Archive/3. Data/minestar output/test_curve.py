# import dependencies
import pandas as pd
import numpy as np


# Now attempt to fit parabolic/quadratic model to cross sections of concern and remove the outliers that are too far from the line of best fit.
# Choosing quadratic model to begin with because it is simple and the shape of the cross section generally represents a quadratic or some other model with ccurvature.
def point_to_line_distance(point: np.ndarray, model):
    """
    This function returns the distance between point and a single line segment
    """
    point_easting = point[0]
    point_elevation = point[1]
    model_intercept = model.intercept_
    model_coef = model.coef_[0]

    # shortest distance betweem point and line is perpendiculr to regression and must pass through
    normal_coef = -1 * (1 / model_coef)
    normal_intercept = point_elevation - (normal_coef * point_easting)

    norm_model_intercept_east = (normal_intercept - model_intercept) / (model_coef - normal_coef)
    norm_model_intercept_elev = normal_coef * (norm_model_intercept_east) + normal_intercept

    distance = np.sqrt(
        (point_easting - norm_model_intercept_east) ** 2 + (point_elevation - norm_model_intercept_elev) ** 2
    )

    return distance


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
    blah = data_diff_full.loc[
        (data_diff_full["northing"] == i) & ((data_diff_full["elevation"] < -6) | ((data_diff_full["elevation"] > 6))),
        :,
    ]
    if not blah.empty:
        norths_to_inspect.append(i)

# Let's test the first case
data_full_1 = data_full[data_full["northing"] == norths_to_inspect[0]]
cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"].reset_index(drop=True)
data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)


# Build K-lines custom algorithm
# Initialise: Generate lines
import numpy as np
from sklearn.linear_model import LinearRegression

# Initial split of data (in future iteration have a random split and let scipy minimize work it out with each iteration) (and have many iterations with different initialisations and see the result)
arrays = np.array_split(np.array(data[["easting", "elevation"]]), 2)
array_1 = arrays[0]
array_2 = arrays[1]
x1 = np.array([[i] for i in array_1[:, 0]])
y1 = np.array([[i] for i in array_1[:, 1]])
x2 = np.array([[i] for i in array_2[:, 0]])
y2 = np.array([[i] for i in array_2[:, 1]])
model = LinearRegression()
reg1 = model.fit(x1, y1)
reg2 = model.fit(x2, y2)

# EM
# Step 1 calculate the distance from points to each line and associate with closest line
distances = []
for i in range(len(data)):
    # need to calculate distance between point and variable number of lines
    distance = point_to_line_distance(np.array(data.loc[i, ["easting", "elevation"]]), model)
    distances.append(distance)
# argmin or index smallest distance and label as the line to be grouped too.
distances.index(min(distances))

# Step 2 refit the linear regression with an additional line
# take full range of eastings and split by an iterator into equal parts
# break if the error of the fit is below a threshold
print
