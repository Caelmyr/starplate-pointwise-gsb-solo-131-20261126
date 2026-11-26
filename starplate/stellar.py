from dataclasses import dataclass
from math import atan2, degrees, hypot, sqrt

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, label, sum as region_sum

from .errors import RequestError
from .geometry import Box, finite_or_none, point_in_frame
from .statistics import robust_summary


@dataclass(frozen=True)
class DetectionSettings:
    threshold: float
    smoothing: float
    min_area: int
    max_area: int
    limit: int

    @classmethod
    def from_fields(cls, fields):
        minimum = fields.integer("min_area", 4, 1, 100)
        return cls(
            fields.number("threshold", 3.5, 1.5, 30),
            fields.number("smoothing", .8, 0, 4),
            minimum,
            fields.integer("max_area", 350, minimum, 10_000),
            fields.integer("limit", 150, 1, 500),
        )


def smooth_retaining_missing(values, smooth):
    valid = np.isfinite(values)
    if smooth == 0:
        return values.copy()
    signal = gaussian_filter(np.where(valid, values, 0).astype(np.float64), smooth, mode="reflect")
    support = gaussian_filter(valid.astype(np.float64), smooth, mode="reflect")
    out = np.full_like(signal, np.nan)
    np.divide(signal, support, where=(support > .8) & valid, out=out)
    return out


def geometry_moments(signal, x, y):
    weight = np.sum(signal, dtype=np.float64)
    if weight <= 0:
        return None
    cx = np.sum(signal * x, dtype=np.float64) / weight
    cy = np.sum(signal * y, dtype=np.float64) / weight
    dx = x - cx
    dy = y - cy
    xx = np.sum(signal * dx * dx, dtype=np.float64) / weight
    yy = np.sum(signal * dy * dy, dtype=np.float64) / weight
    xy = np.sum(signal * dx * dy, dtype=np.float64) / weight
    separation = hypot(xx - yy, 2 * xy)
    major = sqrt(max(0, (xx + yy + separation) / 2))
    minor = sqrt(max(0, (xx + yy - separation) / 2))
    bearing = degrees(atan2(2 * xy, xx - yy)) / 2 % 180
    return {
        "x": float(cx),
        "y": float(cy),
        "sigma_major": float(major),
        "sigma_minor": float(minor),
        "fwhm_major": float(major * 2.35482),
        "fwhm_minor": float(minor * 2.35482),
        "angle": float(bearing),
        "ellipticity": float(1 - minor / major) if major > 0 else None,
    }


def describe_detection(raw, labels, domain, segment, region_label, background):
    rows, cols = segment
    view = raw[rows, cols]
    match = labels[rows, cols] == region_label
    use = match & np.isfinite(view)
    if not np.any(use):
        return None
    x0 = domain.x + cols.start
    y0 = domain.y + rows.start
    ys, xs = np.ogrid[y0:domain.y + rows.stop, x0:domain.x + cols.stop]
    weights = np.where(use, np.maximum(view - background, 0), 0)
    metrics = geometry_moments(weights, xs, ys)
    if metrics is None:
        return None
    local_peak = np.unravel_index(np.argmax(np.where(use, view, -np.inf)), view.shape)
    abs_peak = (x0 + int(local_peak[1]), y0 + int(local_peak[0]))
    bounds = Box(x0, y0, view.shape[1], view.shape[0])
    return {
        **metrics,
        "peak": float(view[local_peak]),
        "peak_x": abs_peak[0],
        "peak_y": abs_peak[1],
        "isophotal_flux": float(np.sum(weights, dtype=np.float64)),
        "area": int(np.count_nonzero(use)),
        "box": bounds.payload(),
        "touches_edge": (
            rows.start == 0 or cols.start == 0
            or rows.stop == domain.height or cols.stop == domain.width
        ),
    }


def source_catalog(frame, box, settings: DetectionSettings):
    raw = box.extract(frame.pixels)
    smooth = smooth_retaining_missing(raw, settings.smoothing)
    sky = robust_summary(smooth, 3, asymmetric=(3.5, 2.5))
    if sky.count < 16 or not np.isfinite(sky.sigma):
        raise RequestError("Source finding needs at least 16 usable pixels.", "box")
    cutoff = sky.median + settings.threshold * sky.sigma
    foreground = np.isfinite(smooth) & (smooth > cutoff)
    labels, number = label(foreground, structure=np.ones((3, 3), dtype=bool))
    histogram = np.bincount(labels.ravel(), minlength=number + 1)
    eligible = np.flatnonzero((histogram >= settings.min_area) & (histogram <= settings.max_area))
    eligible = eligible[eligible > 0]
    parts = find_objects(labels)
    if len(eligible) > settings.limit * 5:
        residual = np.where(foreground, smooth - sky.median, 0)
        preliminary = region_sum(residual, labels, index=eligible)
        strongest = np.argsort(-preliminary, kind="stable")[:settings.limit * 5]
        eligible = eligible[strongest]
    detections = []
    for region_label in eligible:
        fragment = parts[region_label - 1]
        if fragment is None:
            continue
        detection = describe_detection(raw, labels, box, fragment, region_label, sky.median)
        if detection:
            detections.append(detection)
    detections.sort(key=lambda event: (-event["isophotal_flux"], event["peak_y"], event["peak_x"]))
    for index, detection in enumerate(detections[:settings.limit], start=1):
        detection["id"] = index
    return {
        "region": box.payload(),
        "threshold_value": float(cutoff),
        "positive_components": int(number),
        "selected_components": len(detections),
        "background": sky.payload(),
        "sources": detections[:settings.limit],
    }


def psf_shape(frame, box, point, radius, clipping=2.5):
    point_in_frame(point, frame.shape)
    if not box.covers(point):
        raise RequestError("The requested PSF center is outside the inspection box.", "point")
    domain = box.intersect(Box.around(point, radius * 1.55))
    raw = domain.extract(frame.pixels)
    x, y = domain.centers()
    squared = np.square(x - point[0]) + np.square(y - point[1])
    finite = np.isfinite(raw)
    annulus = finite & (squared > radius * radius) & (squared <= (radius * 1.55) ** 2)
    background = robust_summary(raw[annulus], 3.0)
    inside = finite & (squared <= radius * radius)
    support = int(np.count_nonzero(inside))
    if background.retained < 8 or support < 8:
        return {"point": list(point), "psf": None, "support": support, "box": domain.payload()}
    threshold = background.median + clipping * background.sigma
    weights = np.where(inside & (raw > threshold), raw - background.median, 0)
    shape = geometry_moments(weights, x, y)
    residual = np.where(inside, raw - background.median, np.nan)
    return {
        "point": list(point),
        "psf": shape,
        "support": support,
        "positive_support": int(np.count_nonzero(weights)),
        "background": background.payload(),
        "highest_residual": finite_or_none(np.nanmax(residual)),
        "box": domain.payload(),
        "unit": frame.record.unit,
    }
