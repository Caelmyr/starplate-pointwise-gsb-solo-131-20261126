from dataclasses import dataclass
from math import hypot, sqrt

import numpy as np

from .errors import RequestError
from .geometry import Box, finite_or_none
from .registration import Alignment, Blend, transform_for, transformed_patch
from .statistics import median_sigma, robust_summary, trimmed_indices
from .quality import block_slices


@dataclass(frozen=True)
class ComparisonSettings:
    stride: int
    minimum: int
    normalize: bool
    clip_sigma: float

    @classmethod
    def read(cls, fields):
        return cls(
            fields.integer("cell", 64, 24, 400),
            fields.integer("min_joint", 30, 8, 15_000),
            fields.flag("level_match", True),
            fields.number("clip_sigma", 3.5, 2, 10),
        )


def residual_match(first, second, sigma, match):
    difference = first - second
    retained = trimmed_indices(difference, sigma=sigma)
    if not retained.any():
        return 0, difference
    level_delta = float(np.median(difference[retained])) if match else 0
    return level_delta, difference - level_delta


def structural_correlation(first, second):
    if len(first) < 8:
        return None
    ma, sa = median_sigma(first)
    mb, sb = median_sigma(second)
    if not sa or not sb:
        return None
    a = np.clip((first - ma) / sa, -6, 6)
    b = np.clip((second - mb) / sb, -6, 6)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return finite_or_none(np.dot(a, b) / denominator) if denominator else None


def cell_differences(domain, parent, reference, partner_valid, settings, offset):
    rows = []
    for cell in block_slices(domain, settings.stride):
        local = cell.offset(-domain.x, -domain.y)
        data = local.extract(parent)
        peer = local.extract(reference)
        common = np.isfinite(data) & local.extract(partner_valid)
        count = int(np.count_nonzero(common))
        x, y = cell.center
        if count < settings.minimum:
            rows.append({
                "box": cell.payload(), "x": x, "y": y, "usable": False,
                "joint_fraction": count / cell.area, "joint_pixels": count,
            })
            continue
        first, second = data[common], peer[common]
        diff = first.astype(np.float64) - second - offset
        summary = robust_summary(diff, settings.clip_sigma)
        retained = trimmed_indices(diff, sigma=settings.clip_sigma)
        rows.append({
            "box": cell.payload(), "x": x, "y": y, "usable": True,
            "joint_fraction": count / cell.area, "joint_pixels": count,
            "residual": summary.payload(),
            "rms": float(sqrt(np.mean(np.square(diff[retained])))),
            "correlation": structural_correlation(first, second),
        })
    return rows


def comparison_result(parent, companion, domain, fields):
    settings = ComparisonSettings.read(fields)
    blend = Blend.from_fields(fields)
    registration = Alignment.from_fields(fields)
    mapping, description = transform_for(parent, companion, domain, registration)
    aligned, mass = transformed_patch(companion, domain, mapping, blend)
    primary = domain.extract(parent.pixels)
    companion_valid = np.isfinite(aligned)
    joint = np.isfinite(primary) & companion_valid
    count = int(np.count_nonzero(joint))
    if count < settings.minimum:
        raise RequestError("Two-frame comparison needs more paired original samples.", "box")
    base = primary[joint].astype(np.float64)
    peer = aligned[joint].astype(np.float64)
    offset, remaining = residual_match(base, peer, settings.clip_sigma, settings.normalize)
    cells = cell_differences(domain, primary, aligned, companion_valid, settings, offset)
    return {
        "region": domain.payload(),
        "partner": companion.record.key,
        "alignment": description,
        "adjustment": offset,
        "joint_pixels": count,
        "joint_fraction": count / domain.area,
        "cells": cells,
        "residual": robust_summary(remaining, settings.clip_sigma).payload(),
        "interpolation_support": robust_summary(mass[joint], iterations=0).payload(),
        "correlation": structural_correlation(base, peer),
    }


def route_compare(workspace, frame, fields):
    key = fields.text("partner", minimum=1, maximum=72)
    partner = workspace.open(key)
    if frame.record.unit and partner.record.unit and frame.record.unit != partner.record.unit:
        raise RequestError("Residual comparison requires the same physical pixel units.", "partner")
    box = Box.from_fields(fields.child("box"), frame.shape)
    if box.area > 2_600_000:
        raise RequestError("Use a smaller region for pairwise residual comparison.", "box", 413)
    return comparison_result(frame, partner, box, fields)


def overlap_polyline(first, partner, fields):
    alignment = Alignment.from_fields(fields)
    mapping, details = transform_for(first, partner, first.box, alignment)
    points_per_edge = fields.integer("samples", 40, 6, 125)
    h, w = first.shape
    d = np.linspace(0, 1, points_per_edge, endpoint=False)
    zeros = np.zeros_like(d)
    segments = (
        (d * (w - 1), zeros),
        (zeros + w - 1, d * (h - 1)),
        ((1 - d) * (w - 1), zeros + h - 1),
        (zeros, (1 - d) * (h - 1)),
    )
    rows = []
    for x, y in segments:
        mx, my = mapping(x, y)
        line = []
        for i, (px, py) in enumerate(zip(mx, my)):
            point = finite_or_none(px), finite_or_none(py)
            line.append({
                "first": [float(x[i]), float(y[i])],
                "second": list(point),
                "visible_on_second": all(value is not None for value in point)
                and partner.box.covers(point),
            })
        rows.append(line)
    return {
        "footprint": rows,
        "alignment": details,
        "reference_key": first.record.key,
        "partner_key": partner.record.key,
    }
