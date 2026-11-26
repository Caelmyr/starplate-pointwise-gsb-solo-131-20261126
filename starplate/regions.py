from dataclasses import dataclass
from math import cos, hypot, pi, radians, sin, sqrt

import numpy as np
from scipy.ndimage import label

from .errors import RequestError
from .geometry import Box, finite_or_none, point_in_frame
from .masking import polygon_pixels, polygon_points
from .statistics import finite_values, median_sigma, percentile_summary, robust_summary


@dataclass(frozen=True)
class RegionSelection:
    mode: str
    cut: Box
    points: tuple
    contrast: float

    @classmethod
    def from_fields(cls, frame, fields, box):
        mode = fields.option("shape", ("ellipse", "polygon", "box"), "ellipse")
        if mode == "polygon":
            points = tuple(polygon_points(fields, box))
        elif mode == "ellipse":
            center = point_in_frame(fields.point("point"), frame.shape)
            if not box.covers(center):
                raise RequestError("Ellipse center must lie in the inspection box.", "point")
            a = fields.number("major", 18, 1, 1600)
            b = fields.number("minor", 12, 1, a)
            angle = fields.number("angle", 0, -3600, 3600) % 180
            points = center, (a, b), angle
        else:
            points = ()
        return cls(mode, box, points, fields.number("sky", 0))


def ellipse_select(box, center, major, minor, angle):
    x, y = box.centers()
    angle = radians(angle)
    offset_x = x - center[0]
    offset_y = y - center[1]
    first = (cos(angle) * offset_x + sin(angle) * offset_y) / major
    second = (-sin(angle) * offset_x + cos(angle) * offset_y) / minor
    return first ** 2 + second ** 2 <= 1


def region_mask(selection):
    box = selection.cut
    if selection.mode == "ellipse":
        center, (major, minor), angle = selection.points
        return ellipse_select(box, center, major, minor, angle)
    if selection.mode == "polygon":
        return polygon_pixels(box, selection.points)
    return np.ones((box.height, box.width), dtype=bool)


def flux_barycenter(box, weights):
    x, y = box.centers()
    volume = float(np.sum(weights, dtype=np.float64))
    if volume <= 0:
        return None
    return {
        "x": float(np.sum(weights * x, dtype=np.float64) / volume),
        "y": float(np.sum(weights * y, dtype=np.float64) / volume),
    }


def finite_extreme(box, pixels, good, direction):
    x, y = box.centers()
    if not good.any():
        return None
    if direction == "high":
        index = np.argmax(np.where(good, pixels, -np.inf))
    else:
        index = np.argmin(np.where(good, pixels, np.inf))
    row, column = np.unravel_index(index, good.shape)
    return {"x": int(x[0, column]), "y": int(y[row, 0]), "value": float(pixels[row, column])}


def aperture_region(frame, region):
    box = region.cut
    footprint = region_mask(region)
    pixels = box.extract(frame.pixels)
    valid = footprint & np.isfinite(pixels)
    baseline = float(region.contrast)
    sample = np.asarray(pixels[valid], dtype=np.float64)
    volume = float(np.sum(sample - baseline, dtype=np.float64)) if len(sample) else None
    weights = np.where(valid, np.maximum(0, pixels.astype(float) - baseline), 0)
    valid_count = int(np.count_nonzero(valid))
    total_count = int(np.count_nonzero(footprint))
    clipped = bool((footprint[0, :] | footprint[-1, :]).any() or (footprint[:, 0] | footprint[:, -1]).any())
    bbox = selected_bounds(footprint, box)
    return {
        "region": box.payload(),
        "shape": region.mode,
        "points": region.points,
        "support_box": bbox.payload() if bbox is not None else None,
        "valid": valid_count,
        "missing": total_count - valid_count,
        "footprint_count": total_count,
        "touches_box": clipped,
        "sky_offset": baseline,
        "raw_sum": float(np.sum(sample, dtype=np.float64)) if len(sample) else None,
        "sky_subtracted_sum": finite_or_none(volume),
        "positive_centroid": flux_barycenter(box, weights),
        "max_pixel": finite_extreme(box, pixels, valid, "high"),
        "min_pixel": finite_extreme(box, pixels, valid, "low"),
        "robust": robust_summary(sample).payload(),
        "unit": frame.record.unit,
    }


def selected_bounds(mask, domain):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    min_x, max_x = cols.min(), cols.max()
    min_y, max_y = rows.min(), rows.max()
    return Box(
        int(domain.x + min_x), int(domain.y + min_y),
        int(max_x - min_x + 1), int(max_y - min_y + 1),
    )


def centroid_samples(pixel_map, box, footprint, min_level, max_items):
    valid = np.isfinite(pixel_map)
    include = (pixel_map >= min_level) & footprint & valid
    groups, number = label(include, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(groups.ravel(), minlength=number + 1)
    values = np.asarray(np.where(include, pixel_map - min_level, 0), dtype=np.float64)
    totals = np.bincount(groups.ravel(), weights=values.ravel(), minlength=number + 1)
    ids = sorted(range(1, number + 1), key=lambda index: (-totals[index], -sizes[index], index))[:max_items]
    xs, ys = box.centers()
    x_moment = np.bincount(groups.ravel(), weights=(values * xs).ravel(), minlength=number + 1)
    y_moment = np.bincount(groups.ravel(), weights=(values * ys).ravel(), minlength=number + 1)
    return [
        {
            "area": int(sizes[index]),
            "excess": float(totals[index]),
            "x": finite_or_none(x_moment[index] / totals[index]) if totals[index] else None,
            "y": finite_or_none(y_moment[index] / totals[index]) if totals[index] else None,
        }
        for index in ids
    ]


def compare_annular_bands(frame, box, center, radii, reference_level):
    if len(radii) != 3 or not (0 < radii[0] < radii[1] < radii[2]):
        raise RequestError("Three increasing radial limits are required.", "radii")
    point_in_frame(center, frame.shape)
    if not box.covers(center):
        raise RequestError("Mark the ring center inside the inspection box.", "point")
    xs, ys = box.centers()
    distance = np.square(xs - center[0]) + np.square(ys - center[1])
    data = box.extract(frame.pixels)
    profiles = []
    for low, high in zip((0, *radii), radii):
        in_ring = (distance > low ** 2) & (distance <= high ** 2) if low else distance <= high ** 2
        sample = finite_values(data[in_ring])
        theoretical_area = pi * (high ** 2 - low ** 2)
        supported = min(1.0, len(sample) / theoretical_area)
        profiles.append({
            "inner": low, "outer": high,
            "support": supported,
            "pixels": len(sample),
            "median": float(np.median(sample)) if len(sample) else None,
            "net_sum": float(np.sum(sample - reference_level, dtype=np.float64)) if len(sample) else None,
            "percentiles": percentile_summary(sample, (10, 25, 50, 75, 90)),
        })
    return {"center": list(center), "bands": profiles, "unit": frame.record.unit}
