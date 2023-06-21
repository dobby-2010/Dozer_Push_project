"""
    Written by Dominic Cotter

---- Description --------------------------------------------------------------
    This module is for functions that generate or make statistical figures from the dozer push.

-------------------------------------------------------------------------------
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from dozerpush.dozer_backend import _delta_degrees


###############################################################################
# EXTRACT STRAIGHT STATS
##############################################################################
def extract_straight_stats(
    positions: pd.DataFrame,
) -> pd.DataFrame:

    """
        Written by James O'Connell

    ---- Description ----------------------------------------------------------
        This function assembles the cycles, a push and return pair, and for each point in the cycle
        it attributes the speed, cumulative distance, equipment (the dozer), track, whether the
        point is part of the push or return section of the cycle.
    ---------------------------------------------------------------------------

        Usage:  extract_straight_stats(positions)


        Inputs:     positions           Dataframe           Dozer positions with attributes and
                                                            labelled as push, return or
                                                            unclassified.


        Outputs:    straight_cycles     Dataframe           Grouped cycles and the statistics for
                                                            each point in the cycle.
    """

    # Identify type changes
    type_changes = np.hstack(
        (
            [True],
            (positions.iloc[:-1].straight_type.values != positions.iloc[1:].straight_type.values)
            | (positions.iloc[:-1].equipment.values != positions.iloc[1:].equipment.values),
        )
    ).nonzero()[0]

    # Assemble cycles
    straight_type_bounds = np.vstack(
        (
            type_changes,
            np.hstack((type_changes[1:], positions.shape[0])),
        )
    ).T
    straight_cycles = pd.DataFrame(straight_type_bounds, columns=["idx_start", "idx_stop"])

    # Add stats
    straight_cycles.loc[:, "track"] = straight_cycles.apply(
        lambda x: (
            90.0
            - np.degrees(
                np.arctan2(
                    positions.iloc[int(x.idx_stop) - 1].northing - positions.iloc[int(x.idx_start)].northing,
                    positions.iloc[int(x.idx_stop) - 1].easting - positions.iloc[int(x.idx_start)].easting,
                )
            )
        )
        % 360.0,
        axis=1,
    )
    straight_cycles.loc[:, "grade"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start) : int(x.idx_stop)].grade.mean(),
        axis=1,
    )
    straight_cycles.loc[:, "speed"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start) : int(x.idx_stop)].speed.median(),
        axis=1,
    )
    straight_cycles.loc[:, "distance"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start) : int(x.idx_stop)].run_point.sum(),
        axis=1,
    )
    straight_cycles.loc[:, "equipment"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start)].equipment,
        axis=1,
    )
    straight_cycles.loc[:, "pit"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start)].pit,
        axis=1,
    )
    straight_cycles.loc[:, "strip"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start)].strip,
        axis=1,
    )
    straight_cycles.loc[:, "block"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start)].block,
        axis=1,
    )
    straight_cycles.loc[:, "seam"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start)].seam,
        axis=1,
    )
    straight_cycles.loc[:, "shift_date"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start)].shift_date,
        axis=1,
    )
    straight_cycles.loc[:, "shift_description"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start)].shift_description,
        axis=1,
    )
    straight_cycles.loc[:, "status"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start)].straight_type,
        axis=1,
    )

    # Return cycle stats
    return straight_cycles


###############################################################################
# FILTER STRAIGHT STATS
##############################################################################
def filter_straight_stats(
    straight_cycles: pd.DataFrame,
    positions: pd.DataFrame,
    max_distance: float = 25.0,
    max_heading_delta: float = 45.0,
    max_cluster_std: float = 200.0,
    blade_factor: float = 32.0,
) -> tuple((pd.DataFrame, pd.DataFrame)):

    """
        Written by James O'Connell

    ---- Description ----------------------------------------------------------
        This function contains extensive logic to filter the cycles of dozer push and return where
        it looks for transitions in unclassified cycles, removes certain transitions, removes push
        and pull cycles without a neighbour, recalculates all the statistics clusters the dozer
        pushes then returns the cycle data and position data.
    ---------------------------------------------------------------------------

        Usage:    filter_straight_stats(straight_cycles,
                                        positions,
                                        max_distance,
                                        max_heading_delta,
                                        max_cluster_std
                                        )

        Inputs:     straight_cycles         Dataframe           Cycle statistice dataframe.


                    positions               Dataframe           Dozer positions.


                    max_distance            Scalar              Specifies length transitions must
                                                                be shorter than this value.


                    max_geading_delta       Scalar              Consider only entries where the
                                                                "push" and "return" track is within
                                                                max_heading_delta of each other.


                    max_cluster_std         Scalar              Continue to make clusters until
                                                                this threshold is breached.


                    blade_factor            Scalar              Level blade of dozer is filled.


        Outputs:    positions               Dataframe           Dozer positions updated.


                    straight_cycles         Dataframe           Cycle statistice dataframe updated.
    """

    # Look for transitions in "unclassified"
    for idx in range(1, straight_cycles.shape[0] - 2):

        # Only look at "unclassified" straights
        if straight_cycles.iloc[idx].status != "unclassified":
            continue

        # Transitions should be less than max_distance long
        if straight_cycles.iloc[idx].distance > max_distance:
            continue

        # Skip the ends of the record for a particular dozer (prior and next
        # dozer don't match)
        if straight_cycles.iloc[idx - 1].equipment != straight_cycles.iloc[idx + 1].equipment:
            continue

        # Consider only entries where there are "push" and "return" either
        # side of the transition
        if straight_cycles.iloc[idx - 1].status == straight_cycles.iloc[idx + 1].status:
            continue

        # Consider only entries where the"push" and "return" track is within
        # max_heading_delta of each other
        if _delta_degrees(straight_cycles.iloc[idx - 1].track, straight_cycles.iloc[idx + 1].track) < (
            180.0 - max_heading_delta
        ):
            continue

        # We have passed all checks, so remove this transition
        straight_cycles.loc[straight_cycles.index[idx], "status"] = "Remove"

        # Reallocate points to the push and return cycles either side of
        # the transition. The points are allocated to each side based on
        # heading deviations
        transition_headings = positions.iloc[
            straight_cycles.iloc[idx].idx_start : straight_cycles.iloc[idx].idx_stop
        ].track_smoothed.values

        # The central point is the one where the heading is most similar to
        # both the push and return track.
        central_point = (
            np.argmin(
                [
                    _delta_degrees(x, straight_cycles.iloc[idx - 1].track) ** 2.0
                    + _delta_degrees(x, straight_cycles.iloc[idx + 1].track) ** 2.0
                    for x in transition_headings
                ]
            )
            + straight_cycles.iloc[idx].idx_start
        )

        # Determine which side the central point is closest to
        if _delta_degrees(
            positions.iloc[central_point].track_smoothed, straight_cycles.iloc[idx - 1].track
        ) < _delta_degrees(positions.iloc[central_point].track_smoothed, straight_cycles.iloc[idx + 1].track):
            central_point += 1

        # Reallocate the transition points to either side
        straight_cycles.loc[straight_cycles.index[idx - 1], "idx_stop"] = central_point
        straight_cycles.loc[straight_cycles.index[idx + 1], "idx_start"] = central_point

    # Drop removed transitions
    straight_cycles = straight_cycles.loc[straight_cycles.status.ne("Remove")].copy()

    # Remove push and return cycles without a neighbour
    if straight_cycles.shape[0] >= 3:
        for idx in range(straight_cycles.shape[0]):

            # Check first cycle
            if idx == 0:

                # Do we have a push/return combo?
                if (
                    straight_cycles.iloc[idx].status == "push" and straight_cycles.iloc[idx + 1].status == "return"
                ) or (straight_cycles.iloc[idx].status == "return" and straight_cycles.iloc[idx + 1].status == "push"):
                    continue

                # We do not have a combo so remove this straight
                else:
                    straight_cycles.loc[straight_cycles.index[idx], "status"] = "unclassified"
                    continue

            # Check first cycle
            if idx == (straight_cycles.shape[0] - 1):

                # Do we have a push/return combo?
                if (
                    straight_cycles.iloc[idx].status == "push" and straight_cycles.iloc[idx - 1].status == "return"
                ) or (straight_cycles.iloc[idx].status == "return" and straight_cycles.iloc[idx - 1].status == "push"):
                    continue

                # We do not have a combo so remove this straight
                else:
                    straight_cycles.loc[straight_cycles.index[idx], "status"] = "unclassified"
                    continue

            # Check all other cycles for push/return combos
            if (
                straight_cycles.iloc[idx].status == "push"
                and (
                    straight_cycles.iloc[idx - 1].status == "return" or straight_cycles.iloc[idx + 1].status == "return"
                )
            ) or (
                straight_cycles.iloc[idx].status == "return"
                and (straight_cycles.iloc[idx - 1].status == "push" or straight_cycles.iloc[idx + 1].status == "push")
            ):
                continue

            # We do not have a combo so remove this straight
            else:
                straight_cycles.loc[straight_cycles.index[idx], "status"] = "unclassified"
                continue

    # Recalculate all stats
    straight_cycles.loc[:, "track"] = straight_cycles.apply(
        lambda x: (
            90.0
            - np.degrees(
                np.arctan2(
                    positions.iloc[int(x.idx_stop) - 1].northing - positions.iloc[int(x.idx_start)].northing,
                    positions.iloc[int(x.idx_stop) - 1].easting - positions.iloc[int(x.idx_start)].easting,
                )
            )
        )
        % 360.0,
        axis=1,
    )
    straight_cycles.loc[:, "speed"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start) : int(x.idx_stop)].speed.median(),
        axis=1,
    )
    straight_cycles.loc[:, "distance"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start) : int(x.idx_stop)].run_next.sum(),
        axis=1,
    )
    straight_cycles.loc[:, "time"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start) : int(x.idx_stop)].delta_T_next.sum(),
        axis=1,
    )
    straight_cycles.loc[:, "centre_E"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start) : int(x.idx_stop)].easting.mean(),
        axis=1,
    )
    straight_cycles.loc[:, "centre_N"] = straight_cycles.apply(
        lambda x: positions.iloc[int(x.idx_start) : int(x.idx_stop)].northing.mean(),
        axis=1,
    )

    # Cluster dozer push
    features = straight_cycles.loc[straight_cycles.status.eq("push"), ["centre_E", "centre_N"]].values
    for k in range(1, 100):
        clusters = KMeans(n_clusters=k).fit(features)

        # Assess clusters by looking at all points
        stderr = []
        for k in range(0, k):
            dist = (
                (features[clusters.labels_ == k, 0] - clusters.cluster_centers_[k][0]) ** 2.0
                + (features[clusters.labels_ == k, 1] - clusters.cluster_centers_[k][1]) ** 2.0
            ) ** 0.5
            stderr.append(np.std(dist))
        if np.max(stderr) < max_cluster_std:
            break

    # Add clusters to straight stats
    straight_cycles.loc[:, "cluster"] = clusters.predict(straight_cycles[["centre_E", "centre_N"]].values)

    # Apply back to positions
    for cycle in straight_cycles.itertuples():
        positions.loc[
            positions.index[cycle.idx_start] : positions.index[cycle.idx_stop - 1], "straight_start"
        ] = cycle.idx_start
        positions.loc[
            positions.index[cycle.idx_start] : positions.index[cycle.idx_stop - 1], "straight_stop"
        ] = cycle.idx_stop
        positions.loc[
            positions.index[cycle.idx_start] : positions.index[cycle.idx_stop - 1], "straight_track"
        ] = cycle.track
        positions.loc[
            positions.index[cycle.idx_start] : positions.index[cycle.idx_stop - 1], "straight_grade"
        ] = cycle.grade
        positions.loc[
            positions.index[cycle.idx_start] : positions.index[cycle.idx_stop - 1], "straight_distance"
        ] = cycle.distance
        positions.loc[
            positions.index[cycle.idx_start] : positions.index[cycle.idx_stop - 1], "straight_speed"
        ] = cycle.speed
        positions.loc[
            positions.index[cycle.idx_start] : positions.index[cycle.idx_stop - 1], "straight_type"
        ] = cycle.status
        positions.loc[
            positions.index[cycle.idx_start] : positions.index[cycle.idx_stop - 1], "straight_cluster"
        ] = cycle.cluster
    positions.drop(columns=["straight_forthiness"], inplace=True)

    # Filter out the unrealistic times for push and return
    # Remove pushes shorter than the minimum distance
    straight_cycles = straight_cycles[straight_cycles["distance"] > 50.0]
    # Remove pushes longer than the maximum time and distance
    straight_cycles = straight_cycles[(straight_cycles["time"] < 720) & (straight_cycles["time"] > 30.0)]

    ### Future filtering should fit a lm() to time and distance for both push and return and remove those not in confidence interval of lm() ###

    # Add in Productivity Estimate of Actuals
    straight_cycles_new = pd.DataFrame([])
    straight_cycles = straight_cycles.reset_index(drop=True)
    straight_cycles.total_cycle_time = np.nan
    for dozer in straight_cycles.equipment.unique():
        straight_cycles_dz = straight_cycles[straight_cycles.equipment.eq(dozer)].reset_index(drop=True)
        for i in range(len(straight_cycles_dz)):
            if i + 1 == len(straight_cycles_dz):
                continue
            if (
                (straight_cycles_dz.iloc[i].status == "push")
                & (straight_cycles_dz.iloc[i + 1].status == "return")
                & (straight_cycles_dz.iloc[i].idx_stop == straight_cycles_dz.iloc[i + 1].idx_start)
            ):
                straight_cycles_dz.loc[i, "total_cycle_time"] = (
                    straight_cycles_dz.loc[i, "time"] + straight_cycles_dz.shift(-1).loc[i, "time"]
                )
                straight_cycles_dz.loc[i, "total_cycle_distance"] = (
                    straight_cycles_dz.loc[i, "distance"] + straight_cycles_dz.shift(-1).loc[i, "distance"]
                )

        straight_cycles_new = pd.concat(
            [straight_cycles_new, straight_cycles_dz],
            axis=0,
        )

    straight_cycles = straight_cycles_new.reset_index(drop=True)
    straight_cycles["productivity"] = (blade_factor * 3600) / straight_cycles[
        "total_cycle_time"
    ]  # / straight_cycles["total_cycle_distance"]

    # Return results
    return straight_cycles, positions


def combine_positions_straight_cycle(straight_cycles: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:

    """
        Written by Dominic Cotter

    ---- Description ----------------------------------------------------------
        This function's purpose is to combine the remaining information captured in straight_cycles
        that is not captured in positions. Aim is to output single df rather than multiple.
    ---------------------------------------------------------------------------

        Usage:      combine_positions_straight_cycle(straight_cycles, positions)

        Inputs:     straight_cycles         Dataframe           Cycle statistice dataframe.


                    positions               Dataframe           Dozer positions.


        Outputs:    positions               Dataframe           Dozer positions updated with time
                                                                for push/return, centre of
                                                                positions for given status.

    """

    positions["straight_time"] = np.nan
    positions["straight_centre_E"] = np.nan
    positions["straight_centre_N"] = np.nan
    for i in range(len(straight_cycles)):
        mask = (positions["straight_start"] == straight_cycles.iloc[i]["idx_start"]) & (
            positions["straight_stop"] == straight_cycles.iloc[i]["idx_stop"]
        )
        positions.loc[mask, "straight_time",] = straight_cycles.iloc[
            i
        ]["time"]
        positions.loc[mask, "straight_centre_E",] = straight_cycles.iloc[
            i
        ]["centre_E"]
        positions.loc[mask, "straight_centre_N",] = straight_cycles.iloc[
            i
        ]["centre_N"]

    return positions


def finalise_results(straight_cycles: pd.DataFrame, positions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Written by Dominic Cotter

    ---- Description ----------------------------------------------------------
        This function renames all of the columns in the final output.

    ---------------------------------------------------------------------------

        Usage:  finalise_results(straight_cycles, positions)

        Input:  straight_cycles         Dataframe       The summary statistics dataframe of the
                                                        dozer push period.


                positions               Dataframe       The fms positions and atrributes of the
                                                        dozers in production dozing.


        Output: straight_cycles         Dataframe       The summary statistics dataframe of the
                                                        dozer push period with renamed columns.


                positions               Dataframe       The fms positions and atrributes of the
                                                        dozers in production dozing with renamed
                                                        columns.


    """

    positions = positions.drop(
        [
            "heading",
            "points_fwd",
            "points_rev",
            "back_and_forthiness",
            "run_point",
            "duration_point",
            "straight_track",
            "straight_distance",
            "straight_speed",
            "straight_grade",
            "straight_cluster",
        ],
        axis=1,
    )
    positions["datetime"] = positions["datetime"].dt.tz_localize(None)
    positions["Index"] = positions.index

    # Rename Columns
    positions.columns = [
        "Local_Datetime",
        "Track (Degrees)",
        "Speed (km/hr)",
        "Equipment_Code",
        "Site",
        "pit",
        "strip",
        "block",
        "seam",
        "Easting (m)",
        "Northing (m)",
        "Elevation (m)",
        "shift_date",
        "Shift_description",
        "Dist_Prev_Pos (m)",
        "Dist_Next_Pos (m)",
        "Delta_Elev_Previous (m)",
        "Delta_Elev_Next (m)",
        "Delta_Time_Prev (sec)",
        "Delta_Time_Next (sec)",
        "Grade (%)",
        "Track_Smoothed (Degrees)",
        "Speed_Smoothed (km/hr)",
        "Straight_Index_Start",
        "Straight_Index_Stop",
        "Straight_Type",
        "Index",
    ]
    positions.columns = (
        positions.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=True)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
    )

    # Rearrange Columns
    new_col_order = [
        "index",
        "local_datetime",
        "site",
        "shift_date",
        "shift_description",
        "equipment_code",
        "pit",
        "strip",
        "block",
        "seam",
        "easting_m",
        "northing_m",
        "elevation_m",
        "grade_%",
        "speed_km/hr",
        "speed_smoothed_km/hr",
        "track_degrees",
        "track_smoothed_degrees",
        "dist_prev_pos_m",
        "dist_next_pos_m",
        "delta_elev_previous_m",
        "delta_elev_next_m",
        "delta_time_prev_sec",
        "delta_time_next_sec",
        "straight_type",
        "straight_index_start",
        "straight_index_stop",
    ]
    positions = positions[new_col_order]

    straight_cycles.columns = [
        "Straight_Index_Start",
        "Straight_Index_Stop",
        "Straight_Track (Degrees)",
        "Straight_Grade (%)",
        "Straight_Speed (km/h)",
        "Straight_Distance (m)",
        "Equipment_Code",
        "pit",
        "strip",
        "block",
        "seam",
        "Shift_Date",
        "Shift_Description",
        "Straight_Type",
        "Straight_Time (sec)",
        "Straight_E_Centre (m)",
        "Straight_N_Centre (m)",
        "Straight_Cluster",
        "Total_Cycle_Time (sec)",
        "Total_Cycle_Distance (m)",
        "Productivity (bcm/h)",
    ]
    straight_cycles.columns = (
        straight_cycles.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=True)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
    )
    new_col_order = [
        "straight_index_start",
        "straight_index_stop",
        "equipment_code",
        "pit",
        "strip",
        "block",
        "seam",
        "shift_date",
        "shift_description",
        "straight_type",
        "straight_cluster",
        "straight_track_degrees",
        "straight_grade_%",
        "straight_distance_m",
        "straight_speed_km/h",
        "straight_time_sec",
        "total_cycle_time_sec",
        "total_cycle_distance_m",
        "productivity_bcm/h",
        "straight_e_centre_m",
        "straight_n_centre_m",
    ]

    straight_cycles = straight_cycles[new_col_order]

    return straight_cycles, positions
