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
):
    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"
    data_full = pd.concat([data_1, data_2], ignore_index=True)

    data_full["cluster"] = np.nan
    norths = data_full.northing.unique()
    features = []
    for k in tqdm(norths, total=len(data_full.northing.unique()), desc="Filter Cross Sections"):
        # Create parameters

        # Break into cross sections
        cross_section = data_full[data_full["northing"] == k]
        cross_section_t1 = cross_section[cross_section["dataset"] == "t1"]
        cross_section_t2 = cross_section[cross_section["dataset"] == "t2"]
        data_diff = cross_section_t1[["easting", "northing"]].reset_index(drop=True)
        data_diff["delta_elevation"] = cross_section_t1["elevation"].reset_index(drop=True) - cross_section_t2[
            "elevation"
        ].reset_index(drop=True)
        # only select points in cross section where there is a dffference in elevation
        cross_section_worthy = cross_section[
            cross_section["easting"].isin(list(data_diff[data_diff["delta_elevation"] != 0].easting))
        ]
        if cross_section_worthy.empty or (len(cross_section_worthy) < 4):
            continue

        elevation_centres = []
        # process to decide if even worth interpolating for each dataset
        for i in cross_section_worthy["dataset"].unique():
            df_temp = cross_section_worthy[cross_section_worthy["dataset"] == i].reset_index(drop=True)
            df_index = cross_section_worthy[cross_section_worthy["dataset"] == i].copy()
            elevation_centres.append(
                (i, stat.median(cross_section_worthy.loc[cross_section_worthy["dataset"] == i, "elevation"]))
            )
            # write process to find distance from point to point in this dataset

            dist = []
            for index, row in df_temp.iterrows():
                if index < len(df_temp) - 2:
                    dist.append(
                        math.dist(
                            df_temp.loc[index][["easting", "elevation"]],
                            df_temp.loc[index + 1][["easting", "elevation"]],
                        )
                    )
                else:
                    continue
            max_dist = max_dist_tolerance * math.dist(
                df_temp.iloc[0][["easting", "elevation"]], df_temp.iloc[-1][["easting", "elevation"]]
            )

            if stat.stdev(df_temp.elevation) < max_stddev or sum(dist) < max_dist:
                continue
            else:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                yhat = model.fit_predict(df_index[["elevation", "easting"]])
                clusters = unique(yhat)
                for cluster in clusters:
                    # get row indexes for samples with this cluster
                    df_index["cluster"] = yhat
                    row_ix = where(yhat == cluster)

                    # find the centroid of this cluster
                    centroid_elevation = df_index[["elevation", "easting"]].to_numpy()[row_ix, 0].mean()
                    other_surface_elevation = cross_section_worthy[
                        cross_section_worthy["dataset"] != i
                    ].elevation.mean()
                    if (centroid_elevation < other_surface_elevation - distribution_limit) or (
                        centroid_elevation > other_surface_elevation + distribution_limit
                    ):
                        row_ix = list(df_index.iloc[row_ix]["elevation"].index)
                        df_index.loc[row_ix, "elevation"] = np.nan

                if df_index.elevation.isna().all():
                    # just make this the same so that we don't count this cross section
                    df_index.loc[:, "elevation"] = cross_section_worthy[cross_section_worthy["dataset"] != i].loc[
                        :, "elevation"
                    ]
                    update_values = list(df_index.index)
                    data_full.loc[update_values, "elevation"] = df_index["elevation"].values
                else:
                    # In this area begin interpolation
                    df_index.loc[:, "elevation"] = df_index.loc[:, "elevation"].interpolate(method="linear")
                    b, a = cheby1(N=3, Wn=0.94, rp=21)
                    df_index.loc[:, "elevation"] = filtfilt(b, a, df_index.loc[:, "elevation"].values)
                    update_values = list(df_index.index)
                    data_full.loc[update_values, :] = df_index

    # don't need this column anymore
    data_full = data_full.drop(["cluster"], axis=1)

    dataset_1 = data_full[data_full["dataset"] == "t1"].reset_index(drop=True)
    dataset_2 = data_full[data_full["dataset"] == "t2"].reset_index(drop=True)

    return dataset_1, dataset_2


# ----------------------------------------------------------------
def clean_cross_sections_long_confirmed(
    data_1,
    data_2,
    threshold=1,
):
    """
    This function cleans the cross sections that are abnormal for comparing 2 surfaces far apart in time.
    """

    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"
    data_full = pd.concat([data_1, data_2], ignore_index=True)
    data_diff_full = data_1[["easting", "northing"]]
    data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

    norths_to_inspect = []
    for i in data_diff_full["northing"].unique():
        x = data_diff_full.loc[
            (data_diff_full["northing"] == i)
            & ((data_diff_full["elevation"] < -1 * threshold) | ((data_diff_full["elevation"] > threshold))),
            :,
        ]
        if not x.empty:
            norths_to_inspect.append(i)
    for j in range(len(norths_to_inspect)):
        data_full_1 = data_full[data_full["northing"] == norths_to_inspect[j]]
        cross_section_t1 = data_full_1[data_full_1["dataset"] == "t1"].reset_index(drop=True)
        cross_section_t2 = data_full_1[data_full_1["dataset"] == "t2"].reset_index(drop=True)
        data = pd.concat([cross_section_t1, cross_section_t2], ignore_index=True)

        # reshape and make new df with easting and absolute delta elevation
        data_reshaped = data[data["dataset"] == "t1"].reset_index(drop=True)
        data_reshaped["absolute_delta_elevation"] = abs(
            data_reshaped["elevation"] - data[data["dataset"] == "t2"]["elevation"].reset_index(drop=True)
        )
        data_reshaped = data_reshaped[["easting", "absolute_delta_elevation"]]
        # find when abs delta > 6
        rows_of_interest = data_reshaped[data_reshaped["absolute_delta_elevation"] > threshold]
        if not np.isnan(rows_of_interest[rows_of_interest.index < (data_reshaped.index.max() // 3)].index.max()):
            # data_reshaped[data_reshaped["easting"]== data_reshaped[data_reshaped["absolute_delta_elevation"] > 6].max()["easting"]].index[0] # returns index of max elev diff
            data_small_1 = data_full_1.loc[
                data_full_1["easting"]
                < data_reshaped[data_reshaped["absolute_delta_elevation"] > threshold].max()["easting"],
                :,
            ]  # returns all rows where
            data_small_1.loc[data_small_1["dataset"] == "t1", "elevation"] = np.array(
                data_small_1.loc[data_small_1["dataset"] == "t2", "elevation"]
            )
            data_1.loc[data_small_1.loc[data_small_1["dataset"] == "t1", :].index.tolist(), "elevation"] = np.array(
                data_small_1.loc[data_small_1["dataset"] == "t1", "elevation"]
            )

    return data_1, data_2


def locate_dozer(cycles, surface):
    """This function takes the minestar cycles and the minestar surface and identifies the dozer working in the broken slots.
    Assuming that the push direction is due north."""
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
                dozer_code = i
            else:
                pass
    return dozer_code


def clean_slots(
    data_1,
    data_2,
    threshold=1.5,
):
    """
    This function cleans the cross sections that are abnormal for comparing 2 surfaces far apart in time.
    """

    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"
    data_full = pd.concat([data_1, data_2], ignore_index=True)
    data_diff_full = data_1[["easting", "northing"]]
    data_diff_full["elevation"] = data_1["elevation"] - data_2["elevation"]

    # cycles = pd.read_csv(".\\dozerpush\\test\\Result_5.csv") # 5 is for 29
    #cycles = pd.read_csv(".\\dozerpush\\test\\Result_4.csv")
    cycles = pd.read_csv("Result_5.csv")  # 4 is for 28
    # isolate strips of work
    a = sorted(data_diff_full[data_diff_full["elevation"] != 0].northing.unique().tolist())
    b = [x - a[i - 1] for i, x in enumerate(a)][1:]
    b.insert(0, 0.0)
    diff_norths = pd.DataFrame(b)
    start_new_section = diff_norths[diff_norths[0] > 1.0].index.tolist()
    start_new_section.insert(0, 0)
    start_new_section.insert(len(a) - 1, len(a) - 1)
    # loop through and apply fix
    for i in range(len(start_new_section)):
        if i + 1 == len(start_new_section):
            break

        data_test = data_full[data_full["northing"].isin(a[start_new_section[i] : start_new_section[i + 1]])][
            ["easting", "northing", "elevation"]
        ].astype(np.float64)
        X = data_test[["easting", "northing"]].values.reshape(-1, 2)
        Y = data_test["elevation"]
        # Find out who was working
        dozer_code = locate_dozer(cycles, data_test)

        mn = np.min(data_test, axis=0)
        mx = np.max(data_test, axis=0)
        xx_pred, yy_pred = np.meshgrid(
            np.linspace(mn[0], mx[0], int(mx[0] - mn[0] + 1)), np.linspace(mn[1], mx[1], int(mx[1] - mn[1] + 1))
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
            np.array((xx_pred.flatten(), yy_pred.flatten(), predicted - threshold, predicted + threshold)).T,
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
        else:
            raise Exception

    return data_1, data_2


##################################################################################################################################
def calc_volume_minestar(shift, date, data_1_path, data_2_path):
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
    for k in range(len(list_times)):
        if (len(list_times) - 1) == list_times.index(list_times[k]):
            break
        # Read in the dataframes

        if os.path.exists(f"CE_36_B03-{list_times[k]}.csv"):
            data_1 = pd.read_csv(f"CE_36_B03-{list_times[k]}.csv")
            data_1.columns = ["easting", "northing", "elevation"]
        else:
            continue
        if os.path.exists(f"CE_36_B03-{list_times[k+1]}.csv"):
            data_2 = pd.read_csv(f"CE_36_B03-{list_times[k+1]}.csv")
            data_2.columns = ["easting", "northing", "elevation"]
        else:
            continue

        if data_1.empty or data_2.empty:
            continue

        # check that the easting and northing coordinates are aligned between the rwo datasets
        assert_frame_equal(data_1[["easting", "northing"]], data_2[["easting", "northing"]])

        data_1["dataset"] = str(k)
        data_2["dataset"] = str(k + 1)
        data_full = pd.concat([data_1, data_2], ignore_index=True)
        fig = go.Figure()
        fig = px.scatter_3d(
            x=data_full["easting"], y=data_full["northing"], z=data_full["elevation"], color=data_full["dataset"]
        )
        fig.show()

        # Clean up surfaces
        data_1, data_2 = clean_cross_sections(data_1, data_2)
        data_1, data_2 = clean_slots(data_1, data_2)

        # Extract dozer name for dozer specific volumes
        for l in [x for x in data_1.dozer.unique().tolist() if str(x) != "nan"]:
            dozer_df_1 = data_1[data_1["dozer"] == l]
            dozer_df_2 = data_2[data_2["dozer"] == l]

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
                data_unjoined = pd.concat([data_unjoined, pd.concat([pd.DataFrame(x[i]), pd.DataFrame(y[i])], axis=1)])
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
                        datetime.strftime(datetime.strptime(list_times[k], "%Y%m%d%H%M"), "%Y-%m-%d %H:%M:%S"),
                        "%Y-%m-%d %H:%M:%S",
                    ),
                    volume,
                    l,
                )
            )

    volumes = pd.DataFrame(vol_at_time, columns=["datetime", "volume_(m^3)", "dozer"])
    return volumes


# Write code to calculate volume for a shift for the multitude of 15 blocks
calc_volume_minestar("Night", "2023-04-29", "", "")


def calc_volume_minestar_long(shift, date, data_1_path, data_2_path):
    """
    This function calculates the volume between two surfaces that are recorded 15 minutes apart.
    If want to calculate volume over a longer period then use alternative function.
    """

    data_1 = pd.read_csv(".\\dozerpush\\test\\.csv")
    data_1.columns = ["easting", "northing", "elevation"]

    data_2 = pd.read_csv(".\\dozerpush\\test\\.csv")
    data_2.columns = ["easting", "northing", "elevation"]

    # check that the easting and northing coordinates are aligned between the rwo datasets
    assert_frame_equal(data_1[["easting", "northing"]], data_2[["easting", "northing"]])

    data_1["dataset"] = "t1"
    data_2["dataset"] = "t2"
    data_full = pd.concat([data_1, data_2], ignore_index=True)

    # Clean up surfaces
    data_1, data_2 = clean_cross_sections(data_1, data_2)
    data_1, data_2 = clean_cross_sections_long_confirmed(data_1, data_2)

    # create diff surface
    data_diff = data_1.reset_index(drop=True)[["easting", "northing"]]
    data_diff["elevation"] = data_1.reset_index(drop=True)["elevation"] - data_2.reset_index(drop=True)["elevation"]
    # plot to check
    fig = go.Figure()
    fig = px.scatter_3d(x=data_diff["easting"], y=data_diff["northing"], z=data_diff["elevation"])
    fig.show()
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
        data_unjoined = pd.concat([data_unjoined, pd.concat([pd.DataFrame(x[i]), pd.DataFrame(y[i])], axis=1)])
    data_unjoined.columns = ["easting", "northing"]

    # make join between grid df and data_diff df which should leave some null values in elevation column (fill these with 0)
    result = pd.merge(data_unjoined, data_diff, how="left", on=["easting", "northing"])
    result["elevation"] = result["elevation"].fillna(0)

    # Calculate volume
    rows = x.shape[0]
    columns = x.shape[1]
    volume = calculate_volume(rows, columns, result)

    return volume


# cycles = pd.read_csv(".\\dozerpush\\test\\CAT_MINESTAR_cycles_04292100-15.csv")

cycles = pd.read_csv("Result_5.csv")
#cycles = pd.read_csv(".\\dozerpush\\test\\Result_5.csv")

fig = go.Figure()
fig = px.scatter_3d(x=cycles["StartEast"], y=cycles["StartNorth"], z=cycles["StartElv"], color=cycles["NAME"])
fig.show()