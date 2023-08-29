"""
    Written by Dominic Cotter

---- Description --------------------------------------------------------------
    This module is for functions that label a stretch of straight positions as either push or
    return.

-------------------------------------------------------------------------------
"""
import numba
import numpy as np
import pandas as pd

from dozerpush.dozer_backend import _delta_degrees, _average_heading

###############################################################################
# IDENTIFY THE LONGEST POSSIBLE PUSH / RETURN AT EACH POINT
###############################################################################
@numba.guvectorize(["(f8[:,:],i8,f8,f8,f8[:,:])"], "(P,A),(),(),(),(P,B)", nopython=True, target="parallel")
def _numba_identify_straights(
    positions: np.ndarray, idx: int, heading_deviation_max: float, min_distance: float, results: np.ndarray
):

    # Columns
    col_track, col_e, col_n, col_speed, col_dev, col_feature = (0, 1, 2, 3, 4, 5)

    # Initialise running variables
    idx_start = idx
    idx_stop = idx + 1
    segment_track = positions[idx, col_track]

    # Move outwards
    move_back = True
    move_forward = True
    for idx_outwards in range(1, positions.shape[0]):

        # Try backwards
        if (
            move_back
            and (idx - idx_outwards) >= 0
            and positions[idx, col_dev] == positions[idx - idx_outwards, col_dev]
        ):

            # Check direction
            if (
                _delta_degrees(
                    segment_track,
                    positions[idx - idx_outwards, col_track],
                )
                < heading_deviation_max
            ):
                idx_start = idx - idx_outwards
                segment_track = _average_heading(positions[idx_start:idx_stop, col_track])
            else:
                move_back = False
        else:
            move_back = False

        # Try forwards
        if (
            move_forward
            and (idx + idx_outwards) < positions.shape[0]
            and positions[idx, col_dev] == positions[idx + idx_outwards, col_dev]
        ):

            # Check direction
            if (
                _delta_degrees(
                    segment_track,
                    positions[idx + idx_outwards, col_track],
                )
                < heading_deviation_max
            ):
                idx_stop = idx + idx_outwards + 1
                segment_track = _average_heading(positions[idx_start:idx_stop, col_track])
            else:
                move_forward = False
        else:
            move_forward = False

        # Have we finished?
        if not move_back and not move_forward:
            break

    # Check if we meet the minimum distance stat and feature level
    forthiness = positions[idx_start:idx_stop, col_feature].mean()
    if (
        np.sqrt(
            (positions[idx_stop, col_e] - positions[idx_start, col_e]) ** 2.0
            + (positions[idx_stop, col_n] - positions[idx_start, col_n]) ** 2.0
        )
        >= min_distance
        and forthiness > 0.5
    ):

        # Assign stats
        results[idx, 0] = idx_start
        results[idx, 1] = idx_stop
        results[idx, 2] = _average_heading(positions[idx_start:idx_stop, col_track])
        results[idx, 3] = np.sqrt(
            (positions[idx_stop - 1, col_e] - positions[idx_start, col_e]) ** 2.0
            + (positions[idx_stop - 1, col_n] - positions[idx_start, col_n]) ** 2.0
        )
        results[idx, 4] = np.median(positions[idx_start:idx_stop, col_speed])
        results[idx, 5] = forthiness


# ------------------------------------------------------------------------------
def identify_straights(
    positions: pd.DataFrame,
    min_distance: float = 30.0,
    push_speed_threshold: float = 4.4,
    heading_deviation_max: float = 20.0,
) -> pd.DataFrame:
    """
        Written by James O'Connell

    ---- Description ----------------------------------------------------------
        This function identifies whether a position is a part of a push or a return.

        After initialising a processed results variable _numba_identify_straights is the function
        that processes the points. Taking the the positions, the angle which decides whether a
        subsequent is push begins to turn, the minimum distance that a forward push must be to be
        kept we can add features like the straight start, stop, track, distance, etc.

        Given the push_speed_threshold which decides whether the straight type is a push or return.
        The return is typically faster than push. Therefore, if greater than the limit it is a
        return and if less than it is a push.
    ---------------------------------------------------------------------------

        Usage: identify_straights(positions, min_distance, push_speed_threshold, heading_deviation_max)


        Inputs:     positions               Dataframe           Dozer positions, speed, track, etc.


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


        Outputs:    positions               Dataframe           Dozer positions labelled as either
                                                                part of push or return.
    """

    # Initialise results and enumerate devices
    results = np.full((positions.shape[0], 6), np.nan)
    dev_enum = {x: i for i, x in enumerate(np.unique(positions.equipment))}
    positions.loc[:, "dev_enum"] = positions.equipment.apply(lambda x: dev_enum[x])

    # Process points
    _numba_identify_straights(
        positions[["track_smoothed", "easting", "northing", "speed", "dev_enum", "back_and_forthiness"]].values,
        np.arange(positions.shape[0]).astype(np.int64),
        heading_deviation_max,
        min_distance,
        results,
    )

    # Add straights columns and remove device enumeration
    positions.drop(columns=["dev_enum"], inplace=True)
    update_flag = ~np.isnan(results[:, 0])
    positions.loc[update_flag, "straight_start"] = results[update_flag, 0]
    positions.loc[update_flag, "straight_stop"] = results[update_flag, 1]
    positions.loc[update_flag, "straight_track"] = results[update_flag, 2]
    positions.loc[update_flag, "straight_distance"] = results[update_flag, 3]
    positions.loc[update_flag, "straight_speed"] = results[update_flag, 4]
    positions.loc[update_flag, "straight_forthiness"] = results[update_flag, 5]

    # Assign straight type
    positions.loc[
        positions.straight_speed.gt(push_speed_threshold) & ~np.isnan(positions.straight_speed), "straight_type"
    ] = "return"
    positions.loc[
        positions.straight_speed.le(push_speed_threshold) & ~np.isnan(positions.straight_speed), "straight_type"
    ] = "push"

    # Return augmented positions
    return positions


###############################################################################
# CHOOSE MOST APPROPRIATE RETURN AND PUSH CYCLES REGIONS
##############################################################################
@numba.njit()
def _numba_expand_straight_types(positions: np.ndarray) -> np.ndarray:
    # Initialise return
    position_types = np.zeros(positions.shape[0])

    # Column indices
    col_start, col_stop, col_type = (0, 1, 2)

    # Process each position
    for idx in range(positions.shape[0]):

        # Skip if this isn't a key position
        if np.isnan(positions[idx, col_start]):
            continue

        # Process range of key position
        for idx_sweep in range(
            int(positions[idx, col_start]),
            int(positions[idx, col_stop]),
        ):
            # Relabel the position if we have a better choice
            # (push > return > unclassified)
            position_types[idx_sweep] = max(positions[idx, col_type], positions[idx_sweep, col_type])

    # Return point types
    return position_types


# ------------------------------------------------------------------------------
def expand_straight_types(positions: pd.DataFrame) -> pd.DataFrame:
    """
        Written by James O'Connell

    ---- Description ----------------------------------------------------------
        This functions purpose it to attempt to relabel the position as either push or return or
        unclassified if there exists a better choice. It essentially implements
        _numba_expand_straight_types and enumerates since numba requires float.

        The numba function looks between the straight start and stop and attempts to replace
        unclassified with push or return and return with push according to the following hierarchy
        when enumerated: push > return > unclassified.
    ---------------------------------------------------------------------------

        Usage:      expand_straight_types(positions)

        Inputs:     positions           Dataframe       Dozer positions prelabelled as push or
                                                        return.


        Outputs:    positions           Dataframe       Dozer positions relabelled if possible.
    """
    # Expand markings
    positions.loc[:, "straight_type_enum"] = positions.straight_type.replace(
        {"unclassified": 0.0, "return": 1.0, "push": 2.0}
    )
    positions.loc[:, "straight_type_enum"] = _numba_expand_straight_types(
        positions[["straight_start", "straight_stop", "straight_type_enum"]].values,
    )
    positions.loc[:, "straight_type"] = positions.straight_type_enum.replace(
        {0.0: "unclassified", 1.0: "return", 2.0: "push"}
    )
    positions.drop(columns=["straight_type_enum"], inplace=True)

    # Return results
    return positions
