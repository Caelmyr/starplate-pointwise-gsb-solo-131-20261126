from dataclasses import dataclass
from math import hypot

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, generate_binary_structure

from .errors import RequestError
from .geometry import Box, finite_or_none
from .statistics import robust_summary


@dataclass(frozen=True)
class MaskSettings:
    mode: str
    lower: float
    upper: float
    dilate: int

    @classmethod
    def from_fields(cls, fields):
        mode = fields.option("mode", ("inside_range", "outside_range", "polygon"), "outside_range")
        lower = fields.number("lower", -.05)
        upper = fields.number("upper", .15)
        if mode != "polygon" and upper < lower:
            raise RequestError("Upper mask level must be at least the lower level.", "upper")
        return cls(mode, lower, upper, fields.integer("dilate", 0, 0, 12))


def polygon_points(fields, box):
    raw = fields.list("vertices", minimum=3, maximum=30)
    entries = []
    for index, pair in enumerate(raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise RequestError("Polygon vertices must be x and y pairs.", f"vertices.{index}")
        try:
            point = tuple(map(float, pair))
        except (ValueError, TypeError) as error:
            raise RequestError("Polygon vertices must be finite.", f"vertices.{index}") from error
        if not all(np.isfinite(point)) or not box.covers(point):
            raise RequestError("Mask polygon must lie within the inspection box.", f"vertices.{index}")
        if not entries or point != entries[-1]:
            entries.append(point)
    if entries[-1] == entries[0]:
        entries.pop()
    if len(set(entries)) < 3:
        raise RequestError("Polygon must have three distinct vertices.", "vertices")
    return entries


def polygon_pixels(box, vertices):
    xs, ys = box.centers()
    inside = np.zeros((box.height, box.width), dtype=bool)
    edge = np.zeros_like(inside)
    before = vertices[-1]
    for after in vertices:
        x1, y1 = before
        x2, y2 = after
        dx, dy = x2 - x1, y2 - y1
        eps = 1e-10 * max(1, hypot(dx, dy))
        cross = np.abs(dx * (ys - y1) - dy * (xs - x1))
        bounds = (xs >= min(x1, x2) - eps) & (xs <= max(x1, x2) + eps)
        bounds = bounds & (ys >= min(y1, y2) - eps) & (ys <= max(y1, y2) + eps)
        edge |= (cross <= eps) & bounds
        if dy != 0:
            intercept = x1 + dx * (ys - y1) / dy
            crossing = ((y1 > ys) != (y2 > ys)) & (xs < intercept)
            inside ^= crossing
        before = after
    return inside | edge


def bad_pixel_mask(frame, box, settings, vertices=None):
    values = box.extract(frame.pixels)
    valid = np.isfinite(values)
    if settings.mode == "polygon":
        of_interest = polygon_pixels(box, vertices)
    elif settings.mode == "outside_range":
        of_interest = (values < settings.lower) | (values > settings.upper)
    else:
        of_interest = (values >= settings.lower) & (values <= settings.upper)
    base = of_interest & valid
    expanded = base
    if settings.dilate:
        structure = generate_binary_structure(2, 1)
        expanded = binary_dilation(base, structure=structure, iterations=settings.dilate) & valid
    return expanded, {
        "region": box.payload(),
        "mode": settings.mode,
        "input_invalid": int(np.count_nonzero(~valid)),
        "threshold_hits": int(np.count_nonzero(base)),
        "new_invalid": int(np.count_nonzero(expanded)),
        "finite_remaining": int(np.count_nonzero(valid & ~expanded)),
        "result_statistics": robust_summary(values[valid & ~expanded]).payload(),
    }


def masked_pixels(frame, box, local_mask):
    pixels = np.array(frame.pixels, dtype=np.float32)
    subset = pixels[box.slices]
    subset[local_mask] = np.nan
    return pixels


def mask_boundaries(mask, box, sample_limit=1600):
    interior = binary_erosion(mask, structure=generate_binary_structure(2, 1), border_value=0)
    ys, xs = np.nonzero(mask & ~interior)
    if len(xs) > sample_limit:
        indices = np.linspace(0, len(xs) - 1, sample_limit, dtype=np.int64)
        xs, ys = xs[indices], ys[indices]
    return [[int(x + box.x), int(y + box.y)] for x, y in zip(xs, ys)]


def lost_support(pixels, point, max_radius=8):
    x, y = [int(index) for index in point]
    domain = Box.full(pixels.shape).intersect(Box.around((x, y), max_radius))
    patch = domain.extract(pixels)
    valid = np.isfinite(patch)
    if not valid.any():
        return {"distance": None, "nearest": None, "value": None}
    local_x, local_y = x - domain.x, y - domain.y
    if valid[local_y, local_x]:
        return {"distance": 0, "nearest": [x, y], "value": finite_or_none(patch[local_y, local_x])}
    distances, source = distance_transform_edt(~valid, return_indices=True)
    nearest_y, nearest_x = source[:, local_y, local_x]
    return {
        "distance": float(distances[local_y, local_x]),
        "nearest": [int(nearest_x + domain.x), int(nearest_y + domain.y)],
        "value": finite_or_none(patch[nearest_y, nearest_x]),
    }


def compare_validity(first, second):
    if first.shape != second.shape:
        raise RequestError("Both planes must have the same pixel dimensions.", "frame")
    before = np.isfinite(first)
    after = np.isfinite(second)
    keep = before & after
    changed = before & ~after
    return {
        "common": int(np.count_nonzero(keep)),
        "gained": int(np.count_nonzero(~before & after)),
        "lost": int(np.count_nonzero(changed)),
        "both_missing": int(np.count_nonzero(~before & ~after)),
    }
