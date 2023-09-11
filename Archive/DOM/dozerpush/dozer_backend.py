"""
    Written by Dominic Cotter

---- Description --------------------------------------------------------------
    This module houses all the generally used numba functions that do the heavy lifting.

-------------------------------------------------------------------------------
"""

import numpy as np
import numba

###############################################################################
@numba.njit()
def _delta_degrees(alpha, beta):
    return 180.0 - abs(((alpha - beta) % 360.0) - 180.0)


###############################################################################


###############################################################################
@numba.njit()
def _numba_search_start(vector, value):
    left = 0
    right = vector.shape[0]
    while left < right:
        middle = (left + right) // 2
        if vector[middle] < value:
            left = middle + 1
        else:
            right = middle
    return left


@numba.njit()
def _numba_search_stop(vector, value):
    left = 0
    right = vector.shape[0]
    while left < right:
        middle = (left + right) // 2
        if vector[middle] > value:
            right = middle
        else:
            left = middle + 1
    return right


###############################################################################
@numba.njit()
def _average_heading(headings):
    headings = headings[~np.isnan(headings)].copy()
    return (
        90.0
        - np.degrees(
            np.arctan2(
                np.sin(np.radians(90.0 - headings)).mean(),
                np.cos(np.radians(90.0 - headings)).mean(),
            )
        )
    ) % 360.0
