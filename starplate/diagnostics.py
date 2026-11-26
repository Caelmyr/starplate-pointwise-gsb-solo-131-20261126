from dataclasses import dataclass
from math import floor, hypot

import numpy as np
from scipy.ndimage import find_objects, label, median_filter

from .errors import RequestError
from .geometry import Box, finite_or_none, index_for_point
from .quality import block_slices
from .statistics import finite_values, median_sigma, percentile_summary, robust_summary
from .stellar import geometry_moments


@dataclass(frozen=True)
class DefectSettings:
    sigma: float
    neighborhood: int
    max_area: int
    limit: int
    direction: str

    @classmethod
    def from_fields(cls, fields):
        size = fields.integer("neighborhood", 7, 3, 15)
        if size % 2 == 0:
            raise RequestError("Use an odd anomaly neighborhood size.", "neighborhood")
        return cls(
            fields.number("sigma", 6, 2, 50),
            size,
            fields.integer("max_area", 16, 1, 150),
            fields.integer("limit", 65, 1, 500),
            fields.option("direction", ("both", "hot", "cold"), "both"),
        )


def detector_residual(frame, box, size):
    padding = size // 2
    halo = frame.box.intersect(
        Box(box.x - padding, box.y - padding, box.width + 2 * padding, box.height + 2 * padding)
    )
    pixels = halo.extract(frame.pixels)
    finite = np.isfinite(pixels)
    if np.count_nonzero(finite) < 12:
        raise RequestError("Detector diagnostics need at least 12 valid pixels.", "box")
    fill = float(np.median(pixels[finite]))
    estimate = median_filter(np.where(finite, pixels, fill), size=size, mode="nearest")
    offset = Box(box.x - halo.x, box.y - halo.y, box.width, box.height)
    return offset.extract(pixels), offset.extract(estimate), offset.extract(finite)


def anomaly_regions(raw, residual, mask, box, sign, maximum):
    groups, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(groups.ravel(), minlength=count + 1)
    raw_ranges = find_objects(groups)
    eligible = np.flatnonzero((sizes > 0) & (sizes <= maximum))
    eligible = eligible[eligible > 0]
    if len(eligible) > 1800:
        strengths = np.bincount(
            groups.ravel(), weights=np.where(mask, abs(residual), 0).ravel(), minlength=count + 1
        )
        order = np.argsort(-strengths[eligible], kind="stable")[:1800]
        eligible = eligible[order]
    records = []
    for group in eligible:
        rows, cols = raw_ranges[group - 1]
        fragment = residual[rows, cols]
        current = (groups[rows, cols] == group)
        strength = np.where(current, abs(fragment), -np.inf)
        y, x = np.unravel_index(np.argmax(strength), strength.shape)
        absolute_x = box.x + cols.start + x
        absolute_y = box.y + rows.start + y
        original = raw[rows.start + y, cols.start + x]
        records.append({
            "sign": sign, "x": int(absolute_x), "y": int(absolute_y),
            "value": float(original), "residual": float(fragment[y, x]),
            "area": int(sizes[group]),
            "support_box": {
                "x": int(box.x + cols.start), "y": int(box.y + rows.start),
                "width": int(cols.stop - cols.start), "height": int(rows.stop - rows.start),
            },
        })
    return records


def hot_cold(frame, box, settings):
    original, expected, valid = detector_residual(frame, box, settings.neighborhood)
    residual = original.astype(np.float64) - expected
    base = robust_summary(residual[valid], 3.2)
    sigma = base.sigma
    if not sigma:
        deviations = np.abs(residual[valid] - base.median)
        positive = deviations[deviations > 0]
        sigma = float(np.percentile(positive, 25)) / 2 if len(positive) else 0
    records = []
    if sigma:
        threshold = settings.sigma * sigma
        if settings.direction in ("both", "hot"):
            mask = valid & (residual > base.median + threshold)
            records.extend(anomaly_regions(original, residual, mask, box, "hot", settings.max_area))
        if settings.direction in ("both", "cold"):
            mask = valid & (residual < base.median - threshold)
            records.extend(anomaly_regions(original, residual, mask, box, "cold", settings.max_area))
    records.sort(key=lambda record: (-abs(record["residual"] - base.median), record["y"], record["x"]))
    return {
        "region": box.payload(),
        "sigma": sigma,
        "center_residual": base.median,
        "patch_size": settings.neighborhood,
        "total": len(records),
        "marks": records[:settings.limit],
        "unit": frame.record.unit,
    }


def step_contrast(values, dimension):
    average = np.nanmedian(values, axis=dimension)
    good = np.isfinite(average)
    paired = good[1:] & good[:-1]
    differences = np.diff(average)[paired]
    return robust_summary(differences).payload()


def contrast_by_cell(frame, box, size, low_pct, high_pct):
    rows = []
    for cell in block_slices(box, size):
        levels = finite_values(cell.extract(frame.pixels))
        sufficient = len(levels) >= max(8, int(cell.area * .2))
        x, y = cell.center
        if sufficient:
            lo, md, hi = np.percentile(levels, (low_pct, 50, high_pct))
            sigma = median_sigma(levels)[1]
        else:
            lo = md = hi = sigma = float("nan")
        rows.append({
            "x": x, "y": y, "box": cell.payload(),
            "coverage": len(levels) / cell.area, "usable": sufficient,
            "low": finite_or_none(lo), "median": finite_or_none(md), "high": finite_or_none(hi),
            "mad_sigma": finite_or_none(sigma), "span": finite_or_none(hi - lo),
        })
    spreads = [row["span"] for row in rows if row["span"] is not None]
    return {
        "region": box.payload(),
        "cells": rows,
        "width": size,
        "typical_spread": float(np.median(spreads)) if spreads else None,
        "unit": frame.record.unit,
    }
