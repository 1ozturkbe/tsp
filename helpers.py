"""Shared utilities for the TSP and VRP notebooks."""
import math
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Qualitative color palette (10 distinct colors) shared by all plots
PALETTE = list(mcolors.TABLEAU_COLORS.values())


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Symmetric Euclidean distance matrix: D[i,j] = ||coords[i] - coords[j]||."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def plot_iteration_grid(
    history: list[dict],
    plot_fn: Callable[[dict, plt.Axes], None],
    title: str,
    n_cols: int = 3,
) -> None:
    """
    Plot every iteration in `history` on a grid of subplots.

    Parameters
    ----------
    history : per-iteration records
    plot_fn : called as plot_fn(h, ax) to draw one record into one Axes
    title   : figure suptitle
    n_cols  : number of grid columns
    """
    n_iters = len(history)
    n_rows = math.ceil(n_iters / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for idx, h in enumerate(history):
        row, col = divmod(idx, n_cols)
        plot_fn(h, axes[row][col])

    # Hide any unused axes in the last row
    for empty_col in range(n_iters % n_cols or n_cols, n_cols):
        axes[n_rows - 1][empty_col].set_visible(False)

    plt.suptitle(title, fontsize=14, y=1.005)
    plt.tight_layout()
    plt.show()
