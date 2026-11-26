from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, gaussian_laplace, maximum_filter, minimum_filter

from .errors import RequestError
from .geometry import Box, finite_or_none, index_for_point
from .statistics import finite_values, median_sigma, percentile_summary, robust_summary


@dataclass(frozen=True)
class FilterSettings:
    operation: str
    fine: float
    broad: float
    minimum_mass: float
    signed: bool

    @classmethod
    def from_fields(cls, fields):
        fine = fields.number("fine", 1.1, .4, 16)
        broad = fields.number("broad", max(4.5, fine + .5), fine + .2, 30)
        return cls(
            fields.option("filter", ("gaussian", "bandpass", "laplacian", "gradient"), "bandpass"),
            fine, broad,
            fields.number("minimum_mass", .9, .3, 1),
            fields.flag("signed", True),
        )

    def payload(self):
        return {
            "filter": self.operation, "fine": self.fine, "broad": self.broad,
            "minimum_mass": self.minimum_mass, "signed": self.signed,
        }


def normalized_gaussian(pixels, scale, support_cut):
    sampled = np.isfinite(pixels)
    prepared = np.where(sampled, pixels, 0).astype(np.float64)
    blurred = gaussian_filter(prepared, scale, mode="reflect", truncate=3.5)
    mass = gaussian_filter(sampled.astype(np.float64), scale, mode="reflect", truncate=3.5)
    output = np.full(blurred.shape, np.nan)
    np.divide(blurred, mass, where=(mass >= support_cut) & sampled, out=output)
    return output, mass


def pad_source_box(frame, box, settings):
    extent = settings.broad if settings.operation == "bandpass" else settings.fine
    halo = int(np.ceil(extent * 3.5)) + 2
    candidate = Box(box.x - halo, box.y - halo, box.width + 2 * halo, box.height + 2 * halo)
    aligned = frame.box.intersect(candidate)
    inner = Box(box.x - aligned.x, box.y - aligned.y, box.width, box.height)
    return aligned, inner


def gradient_response(small):
    missing = ~np.isfinite(small)
    central_x = np.array([[-.5, 0, .5]])
    central_y = central_x.T
    use = np.where(missing, 0, small)
    dx = convolve(use, -central_x, mode="reflect")
    dy = convolve(use, -central_y, mode="reflect")
    consistent = minimum_filter((~missing).astype(np.uint8), size=3, mode="reflect").astype(bool)
    magnitude = np.where(consistent, np.hypot(dx, dy), np.nan)
    direction = np.where(consistent, np.degrees(np.arctan2(dy, dx)) % 360, np.nan)
    return magnitude, direction


def curvature_response(pixels, small, scale, support_cut):
    valid = np.isfinite(small)
    median = np.nanmedian(small) if valid.any() else 0
    combined = np.where(np.isfinite(pixels), pixels, np.where(valid, small, median))
    laplacian = -gaussian_laplace(combined, scale, mode="reflect", truncate=3.5)
    laplacian = laplacian * scale ** 2
    laplacian[~valid] = np.nan
    return laplacian


def compute_filter(frame, box, settings):
    source, inner = pad_source_box(frame, box, settings)
    pixels = source.extract(frame.pixels)
    small, mass = normalized_gaussian(pixels, settings.fine, settings.minimum_mass)
    direction = None
    if settings.operation == "bandpass":
        broad, broad_mass = normalized_gaussian(pixels, settings.broad, settings.minimum_mass)
        values = small - broad
        mass = np.minimum(mass, broad_mass)
    elif settings.operation == "laplacian":
        values = curvature_response(pixels, small, settings.fine, settings.minimum_mass)
    elif settings.operation == "gradient":
        values, direction = gradient_response(small)
    else:
        values = small
    result = inner.extract(values)
    support = np.isfinite(result)
    if not support.any():
        raise RequestError("Filter support is too limited inside the selection.", "minimum_mass")
    if not settings.signed and settings.operation in ("bandpass", "laplacian"):
        result = np.where(support, np.maximum(result, 0), np.nan)
    details = {
        "region": box.payload(),
        "source_halo": source.payload(),
        "settings": settings.payload(),
        "finite": int(np.count_nonzero(support)),
        "missing": int(np.count_nonzero(~support)),
        "support_mass": percentile_summary(finite_values(inner.extract(mass)[support]), (10, 50, 90)),
        "result_statistics": robust_summary(result).payload(),
    }
    return result, details, inner.extract(direction) if direction is not None else None


def filter_point(frame, box, point, data, direction):
    x, y = index_for_point(point)
    if not box.covers((x, y)):
        return {"x": x, "y": y, "value": None, "original": None, "direction": None}
    local_y, local_x = y - box.y, x - box.x
    return {
        "x": x,
        "y": y,
        "value": finite_or_none(data[local_y, local_x]),
        "original": finite_or_none(float(frame.pixels[y, x])),
        "direction": finite_or_none(direction[local_y, local_x]) if direction is not None else None,
    }


def local_extrema(frame, box, level, distance, max_points, sign):
    patch = box.extract(frame.pixels).astype(np.float64)
    valid = np.isfinite(patch)
    sign_value = 1 if sign == "highest" else -1
    transformed = np.where(valid, patch * sign_value, -np.inf)
    neighbors = maximum_filter(transformed, size=2 * distance + 1, mode="constant", cval=-np.inf)
    extrema = valid & (transformed == neighbors) & (transformed >= level * sign_value)
    points_y, points_x = np.nonzero(extrema)
    values = transformed[points_y, points_x]
    chosen_indices = np.lexsort((points_x, points_y, -values))[:max_points]
    selected = []
    for index in chosen_indices:
        px = points_x[index]
        py = points_y[index]
        selected.append({"x": int(box.x + px), "y": int(box.y + py), "value": float(patch[py, px])})
    return {
        "region": box.payload(), "sign": sign, "level": level,
        "peaks": selected, "total_found": len(points_x),
        "unit": frame.record.unit,
    }
