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


def _reconstruct_polylines(cells: np.ndarray) -> list[list[int]]:
    """Reconstruct connected polylines from edge cells.

    Parameters
    ----------
    cells : numpy.ndarray
        Edge array of shape ``(M, 2)`` where each row is
        ``[start_vertex, end_vertex]``.

    Returns
    -------
    list[list[int]]
        Each inner list contains ordered vertex indices forming one
        connected polyline.
    """
    if len(cells) == 0:
        return []

    lines: list[list[int]] = []
    current_line = [int(cells[0][0]), int(cells[0][1])]
    for i in range(1, len(cells)):
        if cells[i][0] == cells[i - 1][1]:
            current_line.append(int(cells[i][1]))
        else:
            lines.append(current_line)
            current_line = [int(cells[i][0]), int(cells[i][1])]
    lines.append(current_line)
    return lines
