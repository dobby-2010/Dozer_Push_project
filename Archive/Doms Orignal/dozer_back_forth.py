"""
    Written by Dominic Cotter

---- Description --------------------------------------------------------------
    This module relates to the section of steps that discern whether the point is a forward or
    backward movement.

-------------------------------------------------------------------------------
"""

import numba
import numpy as np
import pandas as pd

from dozerpush.dozer_backend import _numba_search_stop, _numba_search_start, _delta_degrees

###############################################################################
# LOOK FOR BACK AND FORTH MOVEMENTS AT EACH POINT
###############################################################################
@numba.guvectorize(["(f8[:,:],i8,f8,f8,f8[:,:])"], "(P,A),(),(),(),(P,B)", nopython=True, target="parallel")
def _numba_find_back_and_forth_movements(
    positions: np.ndarray,
    idx: int,
    heading_deviation_max: float,
    push_speed_threshold: float,
    results: np.ndarray,
):

    # Columns
    col_e, col_n, col_v, col_track = (0, 1, 2, 3)

    # Find grid easting range
    idx_start = _numba_search_start(positions[:, col_e], positions[idx, col_e])
    idx_stop = _numba_search_stop(positions[:, col_e], positions[idx, col_e])

    # Find grid northing range
    idx_start += _numba_search_start(positions[idx_start:idx_stop, col_n], positions[idx, col_n])
    idx_stop = idx_start + _numba_search_stop(positions[idx_start:idx_stop, col_n], positions[idx, col_n])

    # Search subset
    subset = positions[idx_start:idx_stop, :].copy()
    reference_track = positions[idx, col_track]
    for number in range(subset.shape[0]):
        subset[number, col_track] = _delta_degrees(reference_track, subset[number, col_track])

    # Assign heading stats
    if positions[idx, col_v] < push_speed_threshold:
        results[idx, 0] = float(
            ((subset[:, col_v] < push_speed_threshold) & (subset[:, col_track] < heading_deviation_max)).sum()
        )
        results[idx, 1] = float(
            ((subset[:, col_v] >= push_speed_threshold) & (subset[:, col_track] > 180.0 - heading_deviation_max)).sum()
        )
    else:
        results[idx, 0] = float(
            ((subset[:, col_v] >= push_speed_threshold) & (subset[:, col_track] < heading_deviation_max)).sum()
        )
        results[idx, 1] = float(
            ((subset[:, col_v] < push_speed_threshold) & (subset[:, col_track] > 180.0 - heading_deviation_max)).sum()
        )
    if results[idx, 0:2].sum() == 0:
        results[idx, 2] = 0.0
    else:
        results[idx, 2] = 2.0 * min(results[idx, 0:2].min(), 3.0) * results[idx, 0:2].min() / results[idx, 0:2].sum()


# ------------------------------------------------------------------------------
def find_back_and_forth_movements(
    positions: pd.DataFrame,
    grid_size: float = 10.0,
    heading_deviation_max: float = 20.0,
    push_speed_threshold: float = 4.8,
) -> pd.DataFrame:
    """
        Written by James O'Connell

    ---- Description ----------------------------------------------------------
        Places the easting and northings onto an imaginary grid and passes to
        numba_find_back_and_forth_movements which does what its name suggests.

        The function then returns the points whether forward or in reverse to the appropriate
        index.
    ---------------------------------------------------------------------------
    """

    # Build grids
    grids = positions[["easting", "northing", "speed", "track_smoothed"]].copy()
    grids.loc[:, "grid_E"] = grid_size * (grids.easting / grid_size).round()
    grids.loc[:, "grid_N"] = grid_size * (grids.northing / grid_size).round()
    grids.sort_values(["grid_E", "grid_N"], inplace=True)

    # Initialise results
    results = np.full((positions.shape[0], 3), np.nan)

    # Process points
    _numba_find_back_and_forth_movements(
        grids[["grid_E", "grid_N", "speed", "track_smoothed"]].values,
        np.arange(positions.shape[0]).astype(np.int64),
        heading_deviation_max,
        push_speed_threshold,
        results,
    )

    # Add additional columns
    positions.loc[grids.index, "points_fwd"] = results[:, 0]
    positions.loc[grids.index, "points_rev"] = results[:, 1]
    positions.loc[grids.index, "back_and_forthiness"] = results[:, 2]

    # Return augmented positions
    return positions
