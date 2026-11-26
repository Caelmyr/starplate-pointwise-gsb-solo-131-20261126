from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite, sqrt

import numpy as np

from .errors import RequestError
from .geometry import Box, finite_or_none
from .statistics import finite_values, median_sigma, percentile_summary, robust_summary
from .stellar import DetectionSettings, source_catalog


QUALITY_LIMIT = 3_400_000


@dataclass(frozen=True)
class QualitySettings:
    cell: int
    min_support: float
    clip: float
    saturation: float | None

    @classmethod
    def from_fields(cls, fields):
        saturation = (fields.number("saturation") if fields.raw("saturation") not in (None, "") else None)
        return cls(
            fields.integer("cell", 96, 20, 512),
            fields.number("min_support", .7, .1, 1),
            fields.number("clip", 3, 1.5, 8),
            saturation,
        )


def block_slices(box, step):
    nx = max(1, round(box.width / step))
    ny = max(1, round(box.height / step))
    xs = np.rint(np.linspace(box.x, box.right, nx + 1)).astype(int)
    ys = np.rint(np.linspace(box.y, box.bottom, ny + 1)).astype(int)
    for y0, y1 in zip(ys, ys[1:]):
        for x0, x1 in zip(xs, xs[1:]):
            yield Box(int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def clipped_axis_differences(array, axis):
    first = array[1:, :] if axis == 0 else array[:, 1:]
    second = array[:-1, :] if axis == 0 else array[:, :-1]
    difference = first.astype(np.float64) - second
    return finite_values(difference)


def paired_noise(array):
    readouts = {}
    for axis, name in ((0, "y"), (1, "x")):
        sample = clipped_axis_differences(array, axis)
        median, sigma = median_sigma(sample)
        readouts[name] = {
            "adjacent_pairs": len(sample),
            "typical_step": finite_or_none(median),
            "per_pixel_noise": finite_or_none(sigma / sqrt(2)),
        }
    return readouts


def signed_pair_correlation(array, axis):
    first = array[1:, :] if axis == 0 else array[:, 1:]
    second = array[:-1, :] if axis == 0 else array[:, :-1]
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 16:
        return None
    start = first[valid].astype(np.float64)
    end = second[valid].astype(np.float64)
    center_a, sigma_a = median_sigma(start)
    center_b, sigma_b = median_sigma(end)
    if min(sigma_a, sigma_b) <= 0:
        return None
    close = (abs(start - center_a) <= 4 * sigma_a) & (abs(end - center_b) <= 4 * sigma_b)
    if np.count_nonzero(close) < 16:
        return None
    start = start[close]
    end = end[close]
    da = start - np.mean(start)
    db = end - np.mean(end)
    denominator = sqrt(float(np.dot(da, da) * np.dot(db, db)))
    return finite_or_none(float(np.dot(da, db)) / denominator) if denominator else None


def quality_cells(frame, box, settings):
    cells = []
    for block in block_slices(box, settings.cell):
        array = block.extract(frame.pixels)
        valid = np.isfinite(array)
        valid_n = int(np.count_nonzero(valid))
        usable = valid_n >= max(8, settings.min_support * block.area)
        robust = robust_summary(array, settings.clip, asymmetric=(settings.clip + .3, settings.clip))
        saturation = int(np.count_nonzero(valid & (array >= settings.saturation)))             if settings.saturation is not None else None
        noise = paired_noise(array)
        x, y = block.center
        cells.append({
            "box": block.payload(),
            "x": x,
            "y": y,
            "coverage": valid_n / block.area,
            "finite": valid_n,
            "usable": usable,
            "level": robust.median if usable else None,
            "sky_sigma": robust.sigma if usable else None,
            "noise_x": noise["x"]["per_pixel_noise"] if usable else None,
            "noise_y": noise["y"]["per_pixel_noise"] if usable else None,
            "saturated_pixels": saturation,
        })
    return cells


def metric_stats(cells, key):
    values = np.array([row[key] for row in cells if row[key] is not None], dtype=float)
    if not len(values):
        return {"typical": None, "minimum": None, "maximum": None}
    return {"typical": float(np.median(values)), "minimum": float(values.min()), "maximum": float(values.max())}


def grade_quality(frame, box, settings):
    raw = box.extract(frame.pixels)
    cells = quality_cells(frame, box, settings)
    counts = [
        {"count": row["finite"], "saturated": row["saturated_pixels"]}
        for row in cells
    ]
    total = sum(row["count"] for row in counts)
    saturated = sum(row["saturated"] for row in counts) if settings.saturation is not None else None
    return {
        "region": box.payload(),
        "cells": cells,
        "finite": total,
        "empty": box.area - total,
        "usable_cells": sum(cell["usable"] for cell in cells),
        "saturated": saturated,
        "saturation_level": settings.saturation,
        "sky_levels": metric_stats(cells, "level"),
        "sky_noise": metric_stats(cells, "sky_sigma"),
        "noise_x": metric_stats(cells, "noise_x"),
        "noise_y": metric_stats(cells, "noise_y"),
        "row_correlation": signed_pair_correlation(raw, 1),
        "column_correlation": signed_pair_correlation(raw, 0),
        "unit": frame.record.unit,
    }


def line_summaries(array, axis, minimum_fraction):
    lines = array if axis == 0 else array.T
    length = lines.shape[1]
    if lines.shape[0] > 1600:
        indices = np.linspace(0, lines.shape[0] - 1, 1600, dtype=np.int64)
    else:
        indices = np.arange(lines.shape[0])
    report = []
    for i in indices:
        values = finite_values(lines[i])
        sufficient = len(values) >= max(5, minimum_fraction * length)
        if sufficient:
            center, scatter = median_sigma(values)
        else:
            center = scatter = float("nan")
        report.append({
            "index": int(i),
            "sky": finite_or_none(center),
            "noise": finite_or_none(scatter),
            "coverage": len(values) / length,
        })
    return report


def seam_readout(rows, first_index, cutoff):
    if len(rows) < 3:
        return []
    nearby = []
    for previous, current in zip(rows, rows[1:]):
        if current["index"] != previous["index"] + 1:
            continue
        if current["sky"] is None or previous["sky"] is None:
            continue
        delta = current["sky"] - previous["sky"]
        floor = max(1e-10, min(current["noise"], previous["noise"]))
        ratio = abs(delta) / floor
        if ratio >= cutoff:
            nearby.append({"index": current["index"] + first_index, "delta": delta, "background_sigma": ratio})
    nearby.sort(key=lambda item: -item["background_sigma"])
    return nearby[:30]


def noise_profile(frame, box, settings, axis, cutoff):
    raw = box.extract(frame.pixels)
    dimension = 0 if axis == "row" else 1
    origin = box.y if dimension == 0 else box.x
    rows = line_summaries(raw, dimension, settings.min_support)
    seams = seam_readout(rows, origin, cutoff)
    for row in rows:
        row["index"] += origin
    return {
        "axis": axis,
        "region": box.payload(),
        "cells": rows,
        "seams": seams,
        "source_unit": frame.record.unit,
    }


def axial_average(angles, weights):
    if not len(angles):
        return None, None
    radians = np.deg2rad(np.asarray(angles) * 2)
    wx = float(np.dot(weights, np.cos(radians)))
    wy = float(np.dot(weights, np.sin(radians)))
    w = float(np.sum(weights))
    if not w:
        return None, None
    return (degrees(atan2(wy, wx)) / 2) % 180, hypot(wx, wy) / w


def shape_map(frame, box, detection, cutoff_x, cutoff_e):
    sources = source_catalog(frame, box, detection)
    chosen = []
    for row in sources["sources"]:
        ellipticity = row["ellipticity"]
        if row["touches_edge"] or ellipticity is None or row["fwhm_minor"] < cutoff_x:
            continue
        chosen.append({
            "id": row["id"], "x": row["x"], "y": row["y"],
            "major": row["fwhm_major"], "minor": row["fwhm_minor"],
            "angle": row["angle"], "ellipticity": ellipticity,
        })
    majors = np.array([row["major"] for row in chosen])
    minors = np.array([row["minor"] for row in chosen])
    elongated = [row for row in chosen if row["ellipticity"] >= cutoff_e]
    bearing, coherence = axial_average(
        [row["angle"] for row in elongated], [row["ellipticity"] for row in elongated]
    )
    return {
        "region": box.payload(), "stars": chosen,
        "accepted": len(chosen), "elongated": len(elongated),
        "typical_focus": percentile_summary(np.sqrt(majors * minors), (10, 50, 90)),
        "axis_deg": bearing, "axis_coherence": coherence,
    }
