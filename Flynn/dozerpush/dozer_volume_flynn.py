import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statistics as stat
from datetime import datetime, timedelta
import math
from sklearn.metrics import silhouette_score
from scipy.signal import filtfilt, cheby1
from tqdm import tqdm
from numpy import unique
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from numpy import where
from sklearn.cluster import DBSCAN
from pandas.testing import assert_frame_equal
from sklearn import metrics


##################################################################################################################################
############## Define functions ##################################################################################################
def calculate_volume(rows, columns, result):
    # Method for getting volume of the delta surface is to have an equally sppaced grid then perform bilinear
    # interpolation on the surface between four grid points (assuming that the surface is a 1m by 1m grid beginning at the origin)
    #  and integrating over x and y for an analytical solution to the volume and then substituting in the 4 delta elevations.

    # Integral solution is:
    # https://www.wolframalpha.com/input?i=integrate+%28%281-x%29a+%2Bbx%29*%281-y%29+%2B%28%281-x%29c%2Bfx%29*%28y%29+dxdy with x=0 and y=0 substituted
    # z00 - (z00)/2 + (z10)/2 - (z00)/2 + (z01)/2 + (z00)/4 - (z10)/4 - (z01)/4 + (z11)/4
    # (z00)/4 + (z10)/4 + (z01)/4 + (z11)/4
    # mean(z00 + z01 + z10 + z11)

    # Then we have the ability to get volume for each grid square and just need to loop through them all.
    # This is a grid of 344 x 112 or x/y.shape therefore:

    cuboid_volumes_pos = []  # not really cuboid volumes
    cuboid_volumes_neg = []

    for index, row in tqdm(result.iterrows(), total=((rows * columns) - 1 - columns)):
        if index >= (rows * columns) - 1 - columns:
            continue
        else:
            mean = stat.mean(
                [
                    result.loc[index, "elevation"],
                    result.loc[index + 1, "elevation"],
                    result.loc[index + columns, "elevation"],
                    result.loc[index + columns + 1, "elevation"],
                ]
            )
            if mean > 0:
                cuboid_volumes_pos.append(mean)
            else:
                cuboid_volumes_neg.append(mean)

    return sum(cuboid_volumes_pos) + (-1 * sum(cuboid_volumes_neg))


# ---------------------------------------------------------------------
def remove_errors(data_1, data_2, elevation_limit=4.0):
    """This function removes any points which are considered invalid and anomolies:

    Args:
        data_1 (_type_): _description_
        data_2 (_type_): _description_
        elevation_limit (float): This limit is the distance for which the elevation between surface 1 and surface
        2 is considered unreasonible.

    Raises:
        Exception: _description_

    Returns:
        data with points which have been deemed to be incorrect/inaccurate removed
    """
    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"
    data_full = pd.concat([data_1, data_2], ignore_index=True)
    data_diff_full = data_1[["easting", "northing"]]
    data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

    # find all initial locaitons for where we
    norths_to_inspect = []
    for i in data_diff_full["northing"].unique():
        x = data_diff_full.loc[
            (data_diff_full["northing"] == i)
            & ((data_diff_full["elevation"] < -elevation_limit) | ((data_diff_full["elevation"] > elevation_limit))),
            :,
        ]
        if not x.empty:
            norths_to_inspect.append(i)

    for location in norths_to_inspect:
        df1 = data_full[data_full["northing"]] == location  # Identify all points at specfic northing
        cs1 = df1[df1["dataset"] == "t1"]  # cross section 1
        cs2 = df1[df1["dataset"] == "t2"]  # cross section 2
        df = pd.concat([cs1, cs2], ignore_index=True)  # data
        dfdiff = cs1[["easting", "northing"]].reset_index(drop=True)  # data_diff
        dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)

        # use cluster analysis to determine if any sections are
        # using DBSCAN as a cluster technique
        X = dfdiff

        # Optiizing the parameters
        # Defining the list of hyperparameters to try
        eps_list = np.arange(start=0.1, stop=10, step=0.1)
        min_sample_list = np.arange(start=2, stop=5, step=1)

        # setup the silhouette list
        silhouette_coefficients = []
        eps_coefficients = []
        min_samp_list = []

        # create dataframe to store the silhouette parameters for each trial"
        silhouette_scores_data = pd.DataFrame()
        sil_score = 0  # sets the first sil score to zero
        for eps_trial in eps_list:
            for min_sample_trial in min_sample_list:
                clustering = DBSCAN(eps=eps_trial, min_samples=min_sample_trial).fit(X)
                # storing the labels formed by the DBSCAN
                labels = clustering.labels_

                # measure the peformace of dbscan algo
                # idenfitying which points make up our 'core points'
                core_samples = np.zeros_like(labels, dtype=bool)
                core_samples[clustering.core_sample_indices_] = True

                # Calculating "the number of clusters"
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters_ > 0:
                    if (
                        len(np.unique(labels)) > 1
                    ):  # check is lebels if greater than 1 which is has to be. IF not then likely all zeros and not useful anyway
                        sil_score_new = metrics.silhouette_score(X, labels)

                    else:
                        continue

                    if (
                        sil_score_new > sil_score
                    ):  # checks if new silhouette score is greater than previous, if so make it the greatest score. This is to find the greatest silhouse score possible and its corresponding values
                        sil_score = sil_score_new
                        eps_best = eps_trial
                        min_sample_best = min_sample_trial
                        silhouette_scores_data = silhouette_scores_data.append(
                            pd.DataFrame(
                                data=[[sil_score, eps_best, min_sample_best]],
                                columns=[
                                    "Best Silhouette Score",
                                    "Optimal EPS",
                                    "Optimal Minimal Sample Score",
                                ],
                            )
                        )

                else:
                    continue

        db = clustering = DBSCAN(eps=eps_best, min_samples=min_sample_best).fit(X)  # use min samples = 4
        # add the cluster labels to the dfdiff dataframe
        dfdiff["cluster"] = db.labels_
        ########################################################################################################################################################################################################################################################################### #
        # Irterate through each cluster if any part of the cluster is outside the limit then add the northing to a list
        ###################################################################################################################################################################################################################################################################
        # set a limit for which the elevation is too great for it not to be an error
        elevation_limit = 3.5
        index_list = []
        data_update = cs1.reset_index(
            drop=True
        )  # T1 is updating the old surface, drop=True means that we reset the index, if we wanna use the original indexs remove this
        for cluster_number in dfdiff["cluster"].unique():
            cl1 = dfdiff[dfdiff["cluster"] == cluster_number]  # isolate all points connected to this specific cluster
            # remote previous values from index list
            index_list = []
            # find the average elevation for this specific cluster
            elevation_avg = cl1["delta_elevation"].mean()

            # check to see if this elevtion is too great
            if elevation_avg * elevation_avg >= elevation_limit * elevation_limit:
                locations_1 = cl1.iloc[:, [0, 1]]

                # dataframe of locations for which need there elevation wiped
                for easting_change in locations_1["easting"]:
                    for northing_change in locations_1["northing"]:
                        # this will find the index locaions for each easing and nothing found above
                        index_change = data_update[data_update["easting"] == easting_change]
                        index_change_1 = index_change[index_change["northing"] == northing_change]
                        index_list.append(index_change_1.index.tolist())


# ----------------------------------------------------------------
def clean_cross_sections(
    data_1,
    data_2,
    max_clusters=5,
    end_weight=5.0,
    dataset_weight=1e6,
    max_stddev=2,
    max_dist_tolerance=1.3,
    distribution_limit=3,
    eps=3,
    min_samples=4,
    elevation_limit=6.0,
):
    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"

    # combine data
    data_full = pd.concat([data_1, data_2], ignore_index=True)
    data_diff_full = data_1[["easting", "northing"]]
    data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

    # create a copy of data_1 this will be able to be used to update and then compare with data_1
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

    # loop through all clusters that are considered to be inaccurate
    for x in norths_to_inspect:
        df1 = data_full.loc[data_full["northing"] == x, :]  # Identify all points at specific northing
        cs1 = df1.loc[df1["dataset"] == "t1"]  # cross section 1
        cs2 = df1.loc[df1["dataset"] == "t2"]  # cross section 2
        df = pd.concat([cs1, cs2], ignore_index=True)  # data
        dfdiff = cs1[["easting", "northing"]].reset_index(drop=True)  # data_diff
        dfdiff["delta_elevation"] = cs1["elevation"].reset_index(drop=True) - cs2["elevation"].reset_index(drop=True)

        # use cluster analysis to split line into sections
        # using DBSCAN as a cluster technique
        X = dfdiff

        # Optiizing the parameters
        # Defining the list of hyperparameters to try
        eps_list = np.arange(start=0.1, stop=10, step=0.1)
        min_sample_list = np.arange(start=2, stop=5, step=1)

        # setup the silhouette list
        silhouette_coefficients = []
        eps_coefficients = []
        min_samp_list = []

        #####################################################################################################################
        # This section is for the dbscan parameters

        # create dataframe to store the silhouette parameters for each trial"
        silhouette_scores_data = pd.DataFrame()

        sil_score = 0  # sets the first sil score to zero
        for eps_trial in eps_list:
            for min_sample_trial in min_sample_list:
                clustering = DBSCAN(eps=eps_trial, min_samples=min_sample_trial).fit(X)
                # storing the labels formed by the DBSCAN
                labels = clustering.labels_

                # measure the performance of DBSCAN algo
                # identifying which points make up our 'core points'
                core_samples = np.zeros_like(labels, dtype=bool)
                core_samples[clustering.core_sample_indices_] = True

                # Calculating "the number of clusters"
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters_ > 0:
                    if (
                        len(np.unique(labels)) > 1
                    ):  # check if labels is greater than 1 which it has to be. If not, then likely all zeros and not useful anyway
                        sil_score_new = metrics.silhouette_score(X, labels)

                    else:
                        continue

                    if (
                        sil_score_new > sil_score
                    ):  # checks if new silhouette score is greater than previous, if so make it the greatest score. This is to find the greatest silhouette score possible and its corresponding values
                        sil_score = sil_score_new
                        eps_best = eps_trial
                        min_sample_best = min_sample_trial
                        new_data = pd.DataFrame(
                            data=[[sil_score, eps_best, min_sample_best]],
                            columns=[
                                "Best Silhouette Score",
                                "Optimal EPS",
                                "Optimal Minimal Sample Score",
                            ],
                        )
                        silhouette_scores_data = pd.concat([silhouette_scores_data, new_data], ignore_index=True)

                    else:
                        continue

        db = clustering = DBSCAN(eps=eps_best, min_samples=min_sample_best).fit(X)  # use min samples = 4
        # add the cluster labels to the dfdiff dataframe
        dfdiff["cluster"] = db.labels_

        # Irterate through each cluster if any part of the cluster is outside the limit then add the northing to a list
        # set a limit for which the elevation is too great for it not to be an error
        elevation_limit = 3
        index_list = []
        data_update = cs1.reset_index(
            drop=True
        )  # T1 is updating the old surface, drop=True means that we reset the index, if we wanna use the original indexes remove this

        cs1_updated = cs1.copy()
        # Iterate through each cluster and check if the average elevation is too great
        for cluster_number in dfdiff["cluster"].unique():
            cl1 = dfdiff[dfdiff["cluster"] == cluster_number]
            elevation_avg = cl1["delta_elevation"].mean()

            if abs(elevation_avg) >= elevation_limit:
                # Find the corresponding cluster in data_2
                cl2 = cs2[(cs2["dataset"] == "t2")]
                eastings_to_update = cl1[abs(cl1["delta_elevation"]) >= elevation_limit]["easting"].values

                # Update the corresponding points in data_1_updated with values from data_2
                for easting in eastings_to_update:
                    data_1_updated.loc[
                        (data_1_updated["northing"] == x) & (data_1_updated["easting"] == easting),
                        "elevation",
                    ] = cl2[cl2["easting"] == easting]["elevation"].values[0]

    dataset_1 = data_1_updated
    dataset_2 = data_2

    return dataset_1, dataset_2


# ----------------------------------------------------------------
def locate_dozer(cycles, surface, allocated_dozers):
    """This function takes the minestar cycles and the minestar surface and identifies the dozer working in the broken slots.
    Assuming that the push direction is due north."""

    dozer_codes = []
    cycles_dozers = cycles.NAME.unique()
    for i in cycles_dozers:
        cycles_north = cycles[cycles["NAME"] == i]
        norths = cycles_north.StartNorth.unique().tolist()
        # minestar most likely rounds up on norths and eastings so subtract or add to value depending on what the decimal is
        decimal_to_get = np.round(surface.northing.unique().tolist()[0] % 1, 2)
        if decimal_to_get >= 0.5:
            subtract = 1 - decimal_to_get
            norths = [i - subtract for i in norths]
        else:
            add = decimal_to_get
            norths = [i + add for i in norths]
        # norths.append(cycles_north.EndNorth.unique().tolist())
        # search surface and see if this dozer worked in this area
        for j in norths:
            if j in surface.northing.unique().tolist():
                dozer_codes.append(i)

            else:
                pass

    dozer_codes_unique = list(set(dozer_codes))

    if len(dozer_codes_unique) == 1:
        dozer_code = dozer_codes_unique[0]

    else:
        # check if any dozers in dozer_codes already assigned
        # if assigned then drop from dozer_codes
        dozers_remaining = [x for x in dozer_codes_unique if x not in allocated_dozers]
        # calculate centroid of all minestar cycles included in dozer_codes
        plot_cycles_df_start = cycles[
            [
                "StartEast",
                "StartNorth",
                "StartElv",
                "NAME",
            ]
        ]
        plot_cycles_df_stop = cycles[["EndEast", "EndNorth", "EndElv", "NAME"]]
        plot_cycles_df_start["source"] = "minestar"
        plot_cycles_df_stop["source"] = "minestar"

        plot_cycles_df_start = plot_cycles_df_start.rename(
            columns={
                "StartEast": "easting",
                "StartNorth": "northing",
                "StartElv": "elevation",
                "NAME": "dozer",
            }
        )

        plot_cycles_df_stop = plot_cycles_df_stop.rename(
            columns={
                "EndEast": "easting",
                "EndNorth": "northing",
                "EndElv": "elevation",
                "NAME": "dozer",
            }
        )

        cycles_start_end = pd.concat([plot_cycles_df_start, plot_cycles_df_stop])

        minestar_centroid_northings = []

        for i in dozers_remaining:
            minestar_centroid_northings.append(
                cycles_start_end.loc[cycles_start_end["dozer"] == i, "northing"].median()
            )

        # calculate centroid of pams delta surfaces
        pams_centroid_northing = surface.northing.median()
        # determine which is closer then assign
        distance = [abs(i - pams_centroid_northing) for i in minestar_centroid_northings]
        min_index = np.array(distance).argmin()
        # therefore closest dozer is
        dozer_code = dozers_remaining[min_index]

    return dozer_code


# ----------------------------------------------------------------
def clean_slots(
    data_1,
    data_2,
    cycles,
    threshold=1.5,
):
    """
    This function cleans the cross sections that are abnormal for comparing 2 surfaces far apart in time.
    """

    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"
    data_full = pd.concat([data_1, data_2], ignore_index=True)
    data_diff_full = data_1[["easting", "northing"]].copy()
    data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]
    # isolate strips of work
    a = sorted(data_diff_full[data_diff_full["elevation"] != 0].northing.unique().tolist())
    b = [x - a[i - 1] for i, x in enumerate(a)][1:]
    b.insert(0, 0.0)
    diff_norths = pd.DataFrame(b)
    start_new_section = diff_norths[diff_norths[0] > 1.0].index.tolist()
    start_new_section.insert(0, 0)
    start_new_section.insert(len(a) - 1, len(a) - 1)
    allocated_dozers = []
    # loop through and apply fix
    for i in range(len(start_new_section)):
        if i + 1 == len(start_new_section):
            break

        elif start_new_section == [-1, 0]:
            # no work has been done.
            break
        data_test = data_full[data_full["northing"].isin(a[start_new_section[i] : start_new_section[i + 1]])][
            ["easting", "northing", "elevation"]
        ].astype(np.float64)

        X = data_test[["easting", "northing"]].values.reshape(-1, 2)
        Y = data_test["elevation"]

        # Find out who was working using the are to be checked and the rest of the areas
        dozer_code = locate_dozer(cycles, data_test, allocated_dozers)
        allocated_dozers.append(dozer_code)
        mn = np.min(data_test, axis=0)
        mx = np.max(data_test, axis=0)
        xx_pred, yy_pred = np.meshgrid(
            np.linspace(mn[0], mx[0], int(mx[0] - mn[0] + 1)),
            np.linspace(mn[1], mx[1], int(mx[1] - mn[1] + 1)),
        )

        model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
        degree = 3

        # poly = PolynomialFeatures(degree=degree)
        # X_poly = poly.fit_transform(X)
        # lr_poly = LinearRegression()
        # lr_poly.fit(X_poly, Y)
        model = make_pipeline(SplineTransformer(n_knots=4, degree=degree), Ridge(alpha=1e-3))
        model.fit(X, Y)
        predicted = model.predict(model_viz)
        # now plot / see what points would be removed by this method
        data_spline = pd.DataFrame(
            np.array(
                (
                    xx_pred.flatten(),
                    yy_pred.flatten(),
                    predicted - threshold,
                    predicted + threshold,
                )
            ).T,
            columns=[["easting", "northing", "lower_elevation", "upper_elevation"]],
        )

        data_spline.columns = ["_".join(a) for a in data_spline.columns.to_flat_index()]
        # join this dataframe with data_test
        df_spline = pd.merge(data_spline, data_test, on=["easting", "northing"])
        # then update elevations to upper / lower limit of spline fit if outside this interval
        df_spline.loc[df_spline["elevation"] < df_spline["lower_elevation"], "elevation"] = df_spline.loc[
            df_spline["elevation"] < df_spline["lower_elevation"], "lower_elevation"
        ]

        df_spline.loc[df_spline["elevation"] > df_spline["upper_elevation"], "elevation"] = df_spline.loc[
            df_spline["elevation"] > df_spline["upper_elevation"], "upper_elevation"
        ]

        df_spline = df_spline.sort_values(by=["easting", "northing"])
        data_test = data_test.sort_values(by=["easting", "northing"])
        # check same before equating indexes

        if (
            (
                data_test[["easting", "northing"]].reset_index(drop=True)
                == df_spline[["easting", "northing"]].reset_index(drop=True)
            )
            .all()
            .all()
        ):
            df_spline.index = data_test.index

            data_full.loc[df_spline.index, "elevation"] = df_spline.loc[df_spline.index, "elevation"]

            data_full.loc[df_spline.index, "dozer"] = dozer_code

            data_1 = data_full[data_full["dataset"] == "t1"]

            data_2 = data_full[data_full["dataset"] == "t2"]

            data_full = pd.concat(
                [
                    data_full,
                ],
                axis=1,
            )

        else:
            raise Exception

    return data_1, data_2


# ----------------------------------------------------------------
def query_cycles(cycles, start_time, end_time):
    """_summary_

    Args:
        cycles (_type_): _description_
        start_time (_type_): _description_
        end_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    cycles_update = cycles.copy()
    # transform start and end times
    start = str(
        start_time[:4]
        + "-"
        + start_time[4:6]
        + "-"
        + start_time[6:8]
        + " "
        + start_time[8:10]
        + ":"
        + start_time[10:12]
        + ":00"
    )
    end = str(
        end_time[:4] + "-" + end_time[4:6] + "-" + end_time[6:8] + " " + end_time[8:10] + ":" + end_time[10:12] + ":00"
    )
    start = datetime.strftime(
        datetime.strptime(start, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=30) - timedelta(hours=10),
        "%Y-%m-%d %H:%M:%S",
    )
    end = datetime.strftime(
        datetime.strptime(end, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=15) - timedelta(hours=10),
        "%Y-%m-%d %H:%M:%S",
    )
    cycles_update = cycles_update[(cycles_update["TIME_START"] >= start) & (cycles_update["TIME_START"] < end)]

    return cycles_update


# ----------------------------------------------------------------
def calc_volume_minestar(shift, date):
    """
    This function calculates the volume between two surfaces that are recorded 15 minutes apart.
    It additionally denotes which volume moved is contributed by which dozer.
    If want to calculate volume over a longer period then use alternative function.
    """

    # Initial code will loop through the surfaces saved as CSVs
    # Next iteration will utilise SQL database queries

    # create linspace between start and end of shift
    num_linspace = (12 * 4) + 1
    if shift == "Day":
        shift_start = "06:30:00"
        shift_end = "18:30:00"
        list_times = (
            pd.DataFrame(
                pd.DatetimeIndex(
                    np.linspace(
                        pd.Timestamp(f"{date} {shift_start}").value,
                        pd.Timestamp(f"{date} {shift_end}").value,
                        num_linspace,
                    )
                ),
                columns=["datetimes"],
            )
            .datetimes.dt.strftime("%Y%m%d%H%M")
            .tolist()
        )
    else:
        date_next = datetime.strftime(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1), "%Y-%m-%d")
        shift_end = "06:30:00"
        shift_start = "18:30:00"
        list_times = (
            pd.DataFrame(
                pd.DatetimeIndex(
                    np.linspace(
                        pd.Timestamp(f"{date} {shift_start}").value,
                        pd.Timestamp(f"{date_next} {shift_end}").value,
                        num_linspace,
                    )
                ),
                columns=["datetimes"],
            )
            .datetimes.dt.strftime("%Y%m%d%H%M")
            .tolist()
        )

    vol_at_time = []
    print(vol_at_time)
    for k in range(len(list_times)):
        if (len(list_times) - 1) == list_times.index(list_times[k]):
            break
        # Read in the dataframes

        if os.path.exists(f".\\dozerpush\\test\\cycles_minestar_{list_times[k]}.csv"):
            cycles = pd.read_csv(f".\\dozerpush\\test\\cycles_minestar_{list_times[k]}.csv")
        else:
            Exception("No Push Cycles Data")

        if os.path.exists(f".\\dozerpush\\test\\CE_36_B03-{list_times[k]}.csv"):
            data_1 = pd.read_csv(f".\\dozerpush\\test\\CE_36_B03-{list_times[k]}.csv")
            data_1.columns = ["easting", "northing", "elevation"]
        else:
            continue
        if os.path.exists(f".\\dozerpush\\test\\CE_36_B03-{list_times[k+1]}.csv"):
            data_2 = pd.read_csv(f".\\dozerpush\\test\\CE_36_B03-{list_times[k+1]}.csv")
            data_2.columns = ["easting", "northing", "elevation"]
        else:
            continue

        cycles_select = query_cycles(cycles, list_times[k], list_times[k + 1])

        if data_1.empty or data_2.empty or cycles_select.empty:
            continue

        # check that the easting and northing coordinates are aligned between the rwo datasets
        assert_frame_equal(data_1[["easting", "northing"]], data_2[["easting", "northing"]])

        data_1["dataset"] = ""  # Create an empty "dataset" column
        for p in range(len(list_times)):
            if (len(list_times) - 1) == list_times.index(list_times[p]):
                break

        data_1["dataset"] = str(k)  # Assign the value to the "dataset" column

        data_2["dataset"] = str(k + 1)
        data_full = pd.concat([data_1, data_2], ignore_index=True)
        fig = go.Figure()
        time = f"CE_36_B03-{list_times[k]}.csv"
        fig = px.scatter_3d(
            x=data_full["easting"],
            y=data_full["northing"],
            z=data_full["elevation"],
            color=data_full["dataset"],
            title=" ".join(time),
        )
        fig.show()

        # Check if there is any volume to calculate and if not skip
        data_diff_full = data_1[["easting", "northing"]].copy()
        data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]
        a = sorted(data_diff_full[data_diff_full["elevation"] != 0].northing.unique().tolist())
        if a == []:
            continue

        # Clean up surfaces
        data_1, data_2 = clean_cross_sections(data_1, data_2)
        data_1, data_2 = clean_slots(data_1, data_2, cycles_select)

        for l in [x for x in data_1["dozer"].unique().tolist() if str(x) != "nan"]:
            dozer_df_1 = data_1[data_1["dozer"] == l]
            dozer_df_2 = data_2[data_2["dozer"] == l]

            print("Warning: 'dozer' column not found in data_1.")
            # create diff surface
            data_diff = dozer_df_1.reset_index(drop=True)[["easting", "northing"]]
            data_diff["elevation"] = (
                dozer_df_1.reset_index(drop=True)["elevation"] - dozer_df_2.reset_index(drop=True)["elevation"]
            )
            # plot to check
            # fig = go.Figure()
            # fig = px.scatter_3d(x=data_diff["easting"], y=data_diff["northing"], z=data_diff["elevation"])
            # fig.show()
            grid_e_min = data_diff["easting"].min()
            grid_n_min = data_diff["northing"].min()
            grid_e_max = data_diff["easting"].max()
            grid_n_max = data_diff["northing"].max()

            # make grid of points for easting and northing
            x, y = np.meshgrid(
                np.arange(grid_e_min, grid_e_max + 0.1, 1),
                np.arange(grid_n_min, grid_n_max + 0.1, 1),
            )

            # create df of grid points for join
            data_unjoined = pd.DataFrame([])
            for i in tqdm(range(x.shape[0]), desc="Build Dataframe"):
                data_unjoined = pd.concat(
                    [
                        data_unjoined,
                        pd.concat([pd.DataFrame(x[i]), pd.DataFrame(y[i])], axis=1),
                    ]
                )
            data_unjoined.columns = ["easting", "northing"]

            # make join between grid df and data_diff df which should leave some null values in elevation column (fill these with 0)
            result = pd.merge(data_unjoined, data_diff, how="left", on=["easting", "northing"])
            result["elevation"] = result["elevation"].fillna(0)

            # Calculate volume
            rows = x.shape[0]
            columns = x.shape[1]
            volume = calculate_volume(rows, columns, result)
            vol_at_time.append(
                (
                    datetime.strptime(
                        datetime.strftime(
                            datetime.strptime(list_times[k], "%Y%m%d%H%M"),
                            "%Y-%m-%d %H:%M:%S",
                        ),
                        "%Y-%m-%d %H:%M:%S",
                    ),
                    volume,
                    l,
                )
            )

    volumes = pd.DataFrame(vol_at_time, columns=["datetime", "volume_(m^3)", "dozer"])
    return volumes


# cycles = pd.read_csv(".\\dozerpush\\test\\CAT_MINESTAR_cycles_04292100-15.csv")

# cycles = pd.read_csv("Result_5.csv")
# cycles = pd.read_csv(".\\dozerpush\\test\\Result_5.csv")


# fig = go.Figure()
# fig = px.scatter_3d(x=cycles["StartEast"], y=cycles["StartNorth"], z=cycles["StartElv"], color=cycles["NAME"])
# fig.show()


calc_volume_minestar("Night", "2023-04-29")

...
