from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter
from scipy.interpolate import RegularGridInterpolator

from .errors import RequestError
from .geometry import Box, finite_or_none
from .statistics import finite_values, median_sigma, robust_summary, trimmed_indices


@dataclass(frozen=True)
class BackgroundSettings:
    method: str
    tile_size: int
    clip_sigma: float
    min_coverage: float
    smooth: float

    @classmethod
    def from_fields(cls, fields):
        return cls(
            fields.option("method", ("constant", "plane", "mesh"), "mesh"),
            fields.integer("tile_size", 64, 16, 400),
            fields.number("clip_sigma", 3, 1.5, 7),
            fields.number("min_coverage", .55, .1, .99),
            fields.number("smooth", .65, 0, 3),
        )

    def payload(self):
        return {
            "method": self.method,
            "tile_size": self.tile_size,
            "clip_sigma": self.clip_sigma,
            "min_coverage": self.min_coverage,
            "smooth": self.smooth,
        }


def grid_axis(length, size):
    if length <= size:
        return np.array([0, length], dtype=np.int64)
    count = max(1, int(round(length / size)))
    return np.rint(np.linspace(0, length, count + 1)).astype(np.int64)


def tile_statistics(pixels, box, settings):
    xs = grid_axis(box.width, settings.tile_size)
    ys = grid_axis(box.height, settings.tile_size)
    x_centers = (xs[:-1] + xs[1:] - 1) / 2 + box.x
    y_centers = (ys[:-1] + ys[1:] - 1) / 2 + box.y
    levels = np.full((len(ys) - 1, len(xs) - 1), np.nan)
    scatters = levels.copy()
    tile_rows = []
    for row, (top, bottom) in enumerate(zip(ys, ys[1:])):
        for col, (left, right) in enumerate(zip(xs, xs[1:])):
            section = pixels[top:bottom, left:right]
            summary = robust_summary(
                section,
                asymmetric=(settings.clip_sigma + .5, settings.clip_sigma - .3),
            )
            coverage = summary.count / section.size
            usable = coverage >= settings.min_coverage and summary.retained >= 8
            if usable:
                levels[row, col] = summary.median
                scatters[row, col] = summary.sigma
            tile_rows.append({
                "box": Box(box.x + int(left), box.y + int(top), int(right - left), int(bottom - top)).payload(),
                "x": float(x_centers[col]),
                "y": float(y_centers[row]),
                "coverage": coverage,
                "usable": usable,
                "level": finite_or_none(summary.median) if usable else None,
                "sigma": finite_or_none(summary.sigma) if usable else None,
            })
    return x_centers, y_centers, levels, scatters, tile_rows


def fit_plane(x, y, levels, scatters, box):
    xs, ys = np.meshgrid(x, y)
    valid = np.isfinite(levels)
    if np.count_nonzero(valid) < 3:
        raise RequestError("Background plane needs at least three independent usable cells.", "box")
    xc, yc = box.center
    scale = max(box.width, box.height)
    design = np.column_stack((np.ones(np.count_nonzero(valid)), (xs[valid] - xc) / scale, (ys[valid] - yc) / scale))
    targets = levels[valid]
    sigma = np.asarray(scatters[valid])
    positive = sigma[sigma > 0]
    floor = float(np.median(positive)) / 4 if len(positive) else 1
    weights = 1 / np.maximum(sigma, floor)
    coefficients = None
    for _ in range(4):
        matrix = design * weights[:, None]
        if np.linalg.matrix_rank(matrix) < 3:
            raise RequestError("Background cells do not span a two-dimensional plane.", "box")
        coefficients = np.linalg.lstsq(matrix, targets * weights, rcond=None)[0]
        residual = targets - design @ coefficients
        center, scatter = median_sigma(residual)
        if not scatter:
            break
        damping = np.maximum(1, abs(residual - center) / (2.5 * scatter))
        weights = 1 / (np.maximum(sigma, floor) * damping)
    xx, yy = box.centers()
    plane = coefficients[0] + coefficients[1] * (xx - xc) / scale + coefficients[2] * (yy - yc) / scale
    return plane, {
        "center_level": float(coefficients[0]),
        "per_x": float(coefficients[1] / scale),
        "per_y": float(coefficients[2] / scale),
    }


def fill_missing_cells(levels):
    known = np.isfinite(levels)
    if not known.any():
        raise RequestError("No background cell has enough usable original pixels.", "box")
    if known.all():
        return levels
    indices = distance_transform_edt(~known, return_distances=False, return_indices=True)
    return levels[tuple(indices)]


def mesh_plane(xs, ys, levels, box, smooth):
    mesh = fill_missing_cells(levels)
    if smooth:
        mesh = gaussian_filter(mesh, smooth, mode="nearest")
    x_range = np.clip(np.arange(box.x, box.right), xs[0], xs[-1])
    y_range = np.clip(np.arange(box.y, box.bottom), ys[0], ys[-1])
    if len(xs) == 1 and len(ys) == 1:
        return np.full((box.height, box.width), mesh[0, 0])
    if len(xs) == 1:
        return np.repeat(np.interp(y_range, ys, mesh[:, 0])[:, None], box.width, axis=1)
    if len(ys) == 1:
        return np.repeat(np.interp(x_range, xs, mesh[0])[None, :], box.height, axis=0)
    interpolate = RegularGridInterpolator((ys, xs), mesh, bounds_error=True)
    model = np.empty((box.height, box.width), dtype=np.float32)
    step = max(1, 200_000 // box.width)
    for start in range(0, len(y_range), step):
        stop = start + step
        yy, xx = np.meshgrid(y_range[start:stop], x_range, indexing="ij")
        model[start:stop] = interpolate((yy, xx))
    return model


def estimate_background(frame, box, settings: BackgroundSettings):
    pixels = box.extract(frame.pixels)
    xs, ys, levels, scatters, tiles = tile_statistics(pixels, box, settings)
    coefficients = None
    if settings.method == "constant":
        summary = robust_summary(pixels, asymmetric=(settings.clip_sigma + .5, settings.clip_sigma))
        if summary.retained < 8:
            raise RequestError("Background estimation requires at least eight usable pixels.", "box")
        model = np.full(pixels.shape, summary.median, dtype=np.float32)
        coefficients = {"center_level": summary.median, "per_x": 0, "per_y": 0}
    elif settings.method == "plane":
        model, coefficients = fit_plane(xs, ys, levels, scatters, box)
    else:
        model = mesh_plane(xs, ys, levels, box, settings.smooth)
    residual = np.asarray(pixels - model, dtype=np.float32)
    model_stats = robust_summary(model, iterations=0)
    residual_stats = robust_summary(residual)
    return model, residual, {
        "parameters": settings.payload(),
        "region": box.payload(),
        "coefficients": coefficients,
        "cells": tiles,
        "usable_cells": int(np.count_nonzero(np.isfinite(levels))),
        "result_statistics": residual_stats.payload(),
        "model_statistics": model_stats.payload(),
    }


def background_point(frame, box, model, point):
    x, y = map(int, point)
    if not box.covers((x, y)):
        return {"value": None, "background": None, "residual": None}
    original = float(frame.pixels[y, x])
    level = float(model[y - box.y, x - box.x])
    return {
        "value": finite_or_none(original),
        "background": finite_or_none(level),
        "residual": finite_or_none(original - level),
    }


def high_frequency(frame, box, window):
    pixels = box.extract(frame.pixels)
    original = finite_values(pixels)
    if len(original) < 9:
        raise RequestError("The selected region is too sparse for a local filter.", "box")
    finite = np.isfinite(pixels)
    fill = np.where(finite, pixels, np.median(original))
    trend = median_filter(fill, size=window, mode="reflect")
    residual = np.where(finite, pixels - trend, np.nan)
    return trend, residual, {
        "window": window,
        "region": box.payload(),
        "statistics": robust_summary(residual).payload(),
    }
