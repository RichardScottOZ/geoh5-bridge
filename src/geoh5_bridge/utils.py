"""Shared utilities for geoh5-bridge."""

from __future__ import annotations

import numpy as np


def _add_data_columns(
    geoh5_object,
    data: dict[str, np.ndarray],
) -> None:
    """Add multiple named data arrays to a geoh5py object.

    Parameters
    ----------
    geoh5_object
        A geoh5py object (Grid2D, Points, Curve, BlockModel, etc.).
    data
        Mapping of data name to flat numpy array of values.
    """
    for name, values in data.items():
        arr = np.asarray(values, dtype=np.float32)
        geoh5_object.add_data({name: {"values": arr}})
