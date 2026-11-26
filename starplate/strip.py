from dataclasses import dataclass
from math import hypot

import numpy as np

from .errors import RequestError
from .geometry import finite_or_none, point_in_frame, sample_segment
from .sampling import interpolate, nearest
from .statistics import finite_values, robust_summary


@dataclass(frozen=True)
class StripSettings:
    width: int
    spacing: float
    method: str

    @classmethod
    def from_fields(cls, fields):
        return cls(
            fields.integer("half_width", 7, 1, 35),
            fields.number("spacing", 1, .25, 8),
            fields.option("interpolation", ("linear", "nearest"), "linear"),
        )


def cross_coordinates(start, end, distance, x, y, width):
    across = np.arange(-width, width + 1, dtype=float)
    vx, vy = (end[0] - start[0]) / distance, (end[1] - start[1]) / distance
    xx = x[None, :] - across[:, None] * vy
    yy = y[None, :] + across[:, None] * vx
    return across, xx, yy


def sample_strip(frame, box, start, end, settings):
    point_in_frame(start, frame.shape, "start")
    point_in_frame(end, frame.shape, "end")
    distance, guide_x, guide_y = sample_segment(start, end, spacing=settings.spacing, max_points=1400)
    if distance[-1] < 2:
        raise RequestError("Draw a line at least 2 pixels long.", "end")
    offset, xs, ys = cross_coordinates(start, end, distance[-1], guide_x, guide_y, settings.width)
    inside = (xs >= box.x - .5) & (xs < box.right - .5) & (ys >= box.y - .5) & (ys < box.bottom - .5)
    origin_x = xs - box.x
    origin_y = ys - box.y
    original = box.extract(frame.pixels)
    if settings.method == "linear":
        values, applied = interpolate(original, origin_x, origin_y)
    else:
        values = nearest(original, origin_x, origin_y)
        applied = np.isfinite(values).astype(float)
    valid = np.isfinite(values) & inside
    grid = np.where(valid, values, np.nan)
    return distance, offset, xs, ys, grid, np.where(inside, applied, 0)


def summarize_along_axis(grid, indexes):
    results = []
    for index, values in zip(indexes, grid):
        sample = finite_values(values)
        results.append({
            "at": float(index),
            "finite": len(sample),
            "mean": float(np.mean(sample)) if len(sample) else None,
            "median": float(np.median(sample)) if len(sample) else None,
            "max": float(np.max(sample)) if len(sample) else None,
            "coverage": len(sample) / values.size,
        })
    return results


def strip_read(frame, box, start, end, settings):
    distance, offset, source_x, source_y, grid, support = sample_strip(frame, box, start, end, settings)
    statistics = robust_summary(grid)
    if statistics.count == 0:
        raise RequestError("Strip does not contain any usable covered original pixels.", "box")
    rows = summarize_along_axis(grid, offset)
    columns = summarize_along_axis(grid.T, distance)
    return {
        "region": box.payload(),
        "start": list(start), "end": list(end),
        "unit": frame.record.unit,
        "length": float(distance[-1]),
        "offsets": offset.tolist(),
        "distance": distance.tolist(),
        "along": columns,
        "across": rows,
        "statistics": statistics.payload(),
        "reference": {
            "start": [float(source_x[settings.width, 0]), float(source_y[settings.width, 0])],
            "end": [float(source_x[settings.width, -1]), float(source_y[settings.width, -1])],
        },
        "support": int(np.count_nonzero(support >= 1 - 1e-9)),
    }


def strip_point(frame, box, start, end, settings, along, offset):
    distance, offsets, x, y, values, support = sample_strip(frame, box, start, end, settings)
    if not 0 <= along < len(distance) or not 0 <= offset < len(offsets):
        raise RequestError("Pick one of the visible sampling cells.", "sample")
    return {
        "distance": float(distance[along]),
        "offset": float(offsets[offset]),
        "x": float(x[offset, along]), "y": float(y[offset, along]),
        "value": finite_or_none(values[offset, along]),
        "coverage": float(support[offset, along]),
        "sample_type": settings.method,
    }
