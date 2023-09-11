"""
    Written by Dominic Cotter

---- Description --------------------------------------------------------------
    This module is the function that combines all other modules, functions and methods to generate
    the dozer push statistics for a given shift, site and shift description.

-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from dozerpush.dozer_stats import (
    filter_straight_stats,
    extract_straight_stats,
    finalise_results,
)
from dozerpush.dozer_properties import calcs_grades_heading_speed_distance_duration
from dozerpush.dozer_back_forth import find_back_and_forth_movements
from dozerpush.dozer_straight import identify_straights, expand_straight_types

###############################################################################
# CYCLE CLASSIFICATION ENTRY FUNCTION
###############################################################################
def dozer_push(
    positions: pd.DataFrame,
    window_distance_smoothing: float = 30.0,
    grid_size: float = 10.0,
    heading_deviation_max: float = 20.0,
    push_speed_threshold: float = 4.8,
    min_distance: float = 30.0,
    max_distance: float = 25.0,
    max_heading_delta: float = 45.0,
    max_cluster_std: float = 200.0,
    blade_factor: float = 32.0,
) -> tuple((pd.DataFrame, pd.DataFrame)):
    """
        Written by James O'Connell

    ---- Description ----------------------------------------------------------
        Outputs the statistics of the dozer push. The function identifies the straights of the
        dozer push, expands upon these and extracts the statistics for all dozers on a shift at a
        given site.
    ---------------------------------------------------------------------------

        Usage:    dozer_push(positions,
                             window_distance_smoothing,
                             grid_size,
                             heading_deviation_max,
                             push_speed_threshold,
                             min_distance,
                             max_distance,
                             max_heading_delta,
                             max_cluster_std,
                             blade_factor)

        Inputs:     positions               Dataframe           Dozer positions.


                    max_distance            Scalar              Specifies length transitions must
                                                                be shorter than this value.


                    max_geading_delta       Scalar              Consider only entries where the
                                                                "push" and "return" track is within
                                                                max_heading_delta of each other.


                    max_cluster_std         Scalar              Continue to make clusters until
                                                                this threshold is breached.


                    blade_factor            Scalar              Level blade of dozer is filled.


                    min_distance            Scalar              Minimum distance for a stretch of
                                                                straight to be considered a push/
                                                                return.


                    push_speed_threshold    Scalar              The threshold to decide whether
                                                                push/return. Therefore, if greater
                                                                than the limit it is a return and
                                                                if less than it is a push.


                    heading_deviation_max   Scalar              Angle between points less than this
                                                                is added to list of points that
                                                                make a straight. Angles greater
                                                                signify turn.


        Outputs:    positions               Dataframe           Dozer positions updated.


                    straight_cycles         Dataframe           Cycle statistics dataframe.

    """

    # Calculate headings, speed, etc and back-and-forth features
    positions = calcs_grades_heading_speed_distance_duration(positions, window_distance_smoothing)
    positions = find_back_and_forth_movements(positions, grid_size, heading_deviation_max, push_speed_threshold)

    # Find push and return candidates
    positions.loc[:, "straight_start"] = np.nan
    positions.loc[:, "straight_stop"] = np.nan
    positions.loc[:, "straight_track"] = np.nan
    positions.loc[:, "straight_distance"] = np.nan
    positions.loc[:, "straight_speed"] = np.nan
    positions.loc[:, "straight_forthiness"] = np.nan
    positions.loc[:, "straight_type"] = "unclassified"
    positions = identify_straights(positions, min_distance, push_speed_threshold, heading_deviation_max)

    # Expand return and push regions
    positions = expand_straight_types(positions)

    # Straight stats
    straight_cycles = extract_straight_stats(positions)

    # Straight stats
    straight_cycles, positions = filter_straight_stats(
        straight_cycles, positions, max_distance, max_heading_delta, max_cluster_std, blade_factor
    )

    # Rename Columns and Rearrange Values
    straight_cycles, positions = finalise_results(straight_cycles, positions)

    # Return results
    return straight_cycles, positions
