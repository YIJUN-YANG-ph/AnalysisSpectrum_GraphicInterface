#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers to save analysis figures and tabular data."""

import os
from typing import Iterable

import numpy as np
import pandas as pd


def _to_axes_list(axes):
    """Normalize a single axis or iterable of axes into a list."""
    if axes is None:
        return []
    if hasattr(axes, "figure"):
        return [axes]
    if isinstance(axes, Iterable):
        return [ax for ax in axes if hasattr(ax, "figure")]
    return []


def F_save(
    axes,
    data_items,
    isSave=False,
    save_dir=None,
    figure_prefix="group_delay",
    data_prefix="group_delay",
    logger=None,
):
    """Save selected figures and data files when isSave is enabled.

    Parameters
    ----------
    axes : axis or iterable of axis
        Matplotlib axes to locate figures for export.
    data_items : iterable
        Iterable of items in the form ((name1, name2, ...), (data1, data2, ...)).
    isSave : bool
        Save switch. When False no files are written.
    save_dir : str
        Destination directory.
    figure_prefix : str
        Prefix for saved figure names.
    data_prefix : str
        Prefix for saved CSV names.
    logger : callable
        Optional logger callback receiving a plain string.
    """
    status = {
        "saved": False,
        "save_dir": save_dir,
        "figure_files": [],
        "data_files": [],
        "message": "",
    }

    if not isSave:
        status["message"] = "isSave is False"
        return status

    if not save_dir:
        status["message"] = "save_dir is not set"
        return status

    os.makedirs(save_dir, exist_ok=True)
    status["save_dir"] = save_dir

    # Save unique figures from provided axes.
    axes_list = _to_axes_list(axes)
    unique_figures = []
    seen = set()
    for ax in axes_list:
        fig = ax.figure
        if id(fig) not in seen:
            unique_figures.append(fig)
            seen.add(id(fig))

    for idx, fig in enumerate(unique_figures, start=1):
        figure_name = f"{figure_prefix}.png" if len(unique_figures) == 1 else f"{figure_prefix}_{idx:02d}.png"
        figure_path = os.path.join(save_dir, figure_name)
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        status["figure_files"].append(figure_path)

    # Save tabular data as CSV.
    for idx, item in enumerate(data_items or [], start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        column_names, columns_data = item
        if len(column_names) != len(columns_data):
            continue
        data_dict = {}
        for name, column in zip(column_names, columns_data):
            data_dict[str(name)] = np.asarray(column)
        df = pd.DataFrame(data_dict)
        data_name = f"{data_prefix}.csv" if idx == 1 else f"{data_prefix}_{idx:02d}.csv"
        data_path = os.path.join(save_dir, data_name)
        df.to_csv(data_path, index=False)
        status["data_files"].append(data_path)

    status["saved"] = bool(status["figure_files"] or status["data_files"])
    status["message"] = "saved" if status["saved"] else "nothing to save"

    if logger is not None:
        try:
            logger(
                f"Saved {len(status['figure_files'])} figure(s) and "
                f"{len(status['data_files'])} data file(s) to {save_dir}"
            )
        except Exception:
            pass

    return status
