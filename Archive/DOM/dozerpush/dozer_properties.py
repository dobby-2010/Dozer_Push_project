"""
    Written by Dominic Cotter

---- Description --------------------------------------------------------------
    This module relates to the section of steps that process the dozer positions that are found in
    the PWT elements of the TUM which inludes functions that recalculate the track and speed and novelly
    calculate distances and grades.

-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

###############################################################################
# HEADING, SPEED, DISTANCE, AND DURATION
##############################################################################
def calc_track_smoothed(
    positions: pd.DataFrame,
    window_smoothing: float = 20.0,
) -> np.ndarray:
    """
    Written by James O'Connell

    ---- Description ----------------------------------------------------------
        This function's purpose is to smooth the track found in the PAMSArchive for a given dozer
        on a shift.

        First the cumulative run is calculated and clip these (+/- the windo smoothing value) with
        min array value being 0 and the max being the final run value. It then gathers the easting
        and northing delta and claulctes the track as the leftover (modulo) of 90 minus the
        arctangent of ratio of the northing and easting delta.
    ---------------------------------------------------------------------------

        Usage: calc_track_smoothed(positions, window_smoothing)


        Inputs:     positions           Dataframe           The positions dataframe of a dozer.


                    window_smoothing    Scalar              The distance from the point over which
                                                            to smooth.


        Outputs:    smooth_track        Array               The angle or track for a dozer at that
                                                            position.
    """

    # Calculate cumulative run
    cumulative_run = ((positions.easting.diff() ** 2.0 + positions.northing.diff() ** 2.0) ** 0.5).cumsum().values
    cumulative_run[0] = 0.0

    # Interpolation points
    start_run = np.clip(cumulative_run - window_smoothing, 0.0, cumulative_run[-1])
    stop_run = np.clip(cumulative_run + window_smoothing, 0.0, cumulative_run[-1])

    # Calculate coordinate deltas
    delta_e = np.interp(stop_run, cumulative_run, positions.easting.values) - np.interp(
        start_run, cumulative_run, positions.easting.values
    )
    delta_n = np.interp(stop_run, cumulative_run, positions.northing.values) - np.interp(
        start_run, cumulative_run, positions.northing.values
    )

    # Track smoothed
    smooth_track = (90.0 - np.degrees(np.arctan2(delta_n, delta_e))) % 360.0

    # Return smoothed track
    return smooth_track


# ------------------------------------------------------------------------------
def calcs_grades_heading_speed_distance_duration(
    positions: pd.DataFrame,
    window_distance_smoothing: float = 30.0,
) -> pd.DataFrame:

    """
        Written by James O'Connell

    ---- Description ----------------------------------------------------------
        This function, for a given dozer, firstly calculates the distance of the previous and next
        run and the midpoint between the previous and next runs. A run is a stretch of distance
        that the dozer was either pushing or returning.

        The function then calculates the time and elevation deltas for a dozer in order to
        calculate the grades as a percentage.

        It then recalculates the track and speed because the PAMS data is unreliable using a
        smoothing window on the speed and uses calc_track_smoothed() for the track.

        It finally removes all unreliable grades (too high or too low) and replaces these with NaN.
    ---------------------------------------------------------------------------

    Usage:      calcs_grades_heading_speed_distance_duration(positions, window_distance_smoothing)


    Inputs:     positions                   DataFrame           The dozer positions (ENU) with
                                                                speed, heading and track, etc.


                window_distance_smoothing   Scalar              Factor to decide the size of the
                                                                rolling window used in smoothing
                                                                the speed of the dozer.


    Outputs:    positions                   DataFrame           Now contains distances, grade,
                                                                updated speed and track.

    """

    # Process by dozer
    for dozer in positions.groupby("equipment"):

        # Calculate distances
        positions.loc[dozer[1].index, "run_prev"] = -(
            (positions.loc[dozer[1].index].easting.diff() ** 2.0 + positions.loc[dozer[1].index].northing.diff() ** 2.0)
            ** 0.5
        )
        positions.loc[dozer[1].index, "run_next"] = (
            positions.loc[dozer[1].index].easting.diff(periods=-1) ** 2.0
            + positions.loc[dozer[1].index].northing.diff(periods=-1) ** 2.0
        ) ** 0.5
        positions.loc[dozer[1].index, "run_point"] = 0.5 * (
            positions.loc[dozer[1].index].run_next - positions.loc[dozer[1].index].run_prev
        )

        # Calculate required window sizes
        point_spacing = positions.run_point.median()
        window_smoothing = max(int(round(window_distance_smoothing / point_spacing)), 3)

        # Calculate elevation deltas
        positions.loc[dozer[1].index, "delta_Z_prev"] = -positions.loc[dozer[1].index].elevation.diff()
        positions.loc[dozer[1].index, "delta_Z_next"] = -positions.loc[dozer[1].index].elevation.diff(periods=-1)

        # Calculate time deltas
        positions.loc[dozer[1].index, "delta_T_prev"] = positions.loc[dozer[1].index].datetime.diff().dt.total_seconds()
        positions.loc[dozer[1].index, "delta_T_next"] = (
            -positions.loc[dozer[1].index].datetime.diff(periods=-1).dt.total_seconds()
        )
        positions.loc[dozer[1].index, "duration_point"] = 0.5 * (
            positions.loc[dozer[1].index].delta_T_next + positions.loc[dozer[1].index].delta_T_prev
        )

        # Calculate grade
        positions.loc[dozer[1].index, "grade"] = 100.0 * (
            (
                positions.loc[dozer[1].index].run_next ** 2.0 * positions.loc[dozer[1].index].delta_Z_prev
                - positions.loc[dozer[1].index].run_prev ** 2.0 * positions.loc[dozer[1].index].delta_Z_next
            )
            / (
                positions.loc[dozer[1].index].run_next ** 2.0 * positions.loc[dozer[1].index].run_prev
                - positions.loc[dozer[1].index].run_prev ** 2.0 * positions.loc[dozer[1].index].run_next
            )
        )

        # Calculate track from positions only. IVolve messes with heading and
        # track so their recorded values are useless.
        positions.loc[dozer[1].index, "track_smoothed"] = calc_track_smoothed(positions.loc[dozer[1].index], 20.0)

        # Smoothed speeds
        positions.loc[dozer[1].index, "speed_smoothed"] = (
            positions.loc[dozer[1].index].speed.rolling(window_smoothing, center=True, min_periods=1).median()
        )

    # Remove unreliable grades due to GPS drift
    positions.loc[
        positions.delta_T_prev.lt(0.5)
        | positions.delta_T_prev.gt(10.1)
        | positions.delta_T_next.lt(0.5)
        | positions.delta_T_next.gt(10.1)
        | positions.grade.abs().gt(25.0),
        "grade",
    ] = np.nan

    # Return dataframe
    return positions
