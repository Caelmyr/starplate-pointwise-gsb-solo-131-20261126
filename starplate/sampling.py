from math import ceil, floor

import numpy as np

from .errors import RequestError
from .geometry import Box, contiguous_ranges, finite_or_none, index_for_point, point_in_frame, sample_segment
from .statistics import finite_values, values_at_points


def nearest(image, x, y):
    x = np.floor(np.asarray(x) + .5).astype(int)
    y = np.floor(np.asarray(y) + .5).astype(int)
    return values_at_points(image, x, y)


def interpolate(image, x, y, minimum_weight=1.0 - 1e-9):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    geometrically_possible = (
        np.isfinite(x) & np.isfinite(y) & (x >= -1) & (y >= -1)
        & (x <= image.shape[1]) & (y <= image.shape[0])
    )
    x = np.where(geometrically_possible, x, -2)
    y = np.where(geometrically_possible, y, -2)
    left = np.floor(x).astype(np.int64)
    top = np.floor(y).astype(np.int64)
    dx = x - left
    dy = y - top
    weights = (
        ((1 - dx) * (1 - dy), left, top),
        (dx * (1 - dy), left + 1, top),
        ((1 - dx) * dy, left, top + 1),
        (dx * dy, left + 1, top + 1),
    )
    numerator = np.zeros_like(x)
    denominator = np.zeros_like(x)
    for weight, ix, iy in weights:
        original = values_at_points(image, ix, iy)
        is_valid = np.isfinite(original)
        numerator += np.where(is_valid, original, 0) * weight
        denominator += np.where(is_valid, weight, 0)
    values = np.full_like(x, np.nan)
    use = (denominator >= minimum_weight) & (denominator > 0)
    np.divide(numerator, denominator, out=values, where=use)
    return values, denominator


def summarize_neighborhood(frame, point, radius):
    x, y = index_for_point(point)
    outer = Box(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1)
    box = outer.intersect(frame.box)
    values = finite_values(box.extract(frame.pixels))
    return {
        "window": box.payload(),
        "finite": len(values),
        "missing": outer.area - len(values),
        "mean": finite_or_none(values.mean()) if len(values) else None,
        "median": finite_or_none(np.median(values)) if len(values) else None,
        "minimum": finite_or_none(values.min()) if len(values) else None,
        "maximum": finite_or_none(values.max()) if len(values) else None,
    }


def patch_values(frame, center, half_size=4):
    x, y = index_for_point(center)
    xs = np.arange(x - half_size, x + half_size + 1)
    ys = np.arange(y - half_size, y + half_size + 1)
    x_grid, y_grid = np.meshgrid(xs, ys)
    values = values_at_points(frame.pixels, x_grid, y_grid)
    return {
        "x": xs.tolist(),
        "y": ys.tolist(),
        "values": [[finite_or_none(value) for value in row] for row in values],
    }


def probe(frame, point, radius=1, patch_size=4):
    point_in_frame(point, frame.shape)
    x, y = index_for_point(point)
    interpolated, support = interpolate(frame.pixels, point[0], point[1])
    center = float(frame.pixels[y, x])
    return {
        "frame_key": frame.record.key,
        "position": list(point),
        "original_pixel": {"x": x, "y": y, "value": finite_or_none(center)},
        "interpolated": finite_or_none(float(interpolated)),
        "interpolated_support": float(support),
        "neighborhood": summarize_neighborhood(frame, point, radius),
        "patch": patch_values(frame, point, patch_size),
        "unit": frame.record.unit,
    }


def line_profile(frame, start, end, spacing, interpolation, width):
    point_in_frame(start, frame.shape, "start")
    point_in_frame(end, frame.shape, "end")
    distances, x, y = sample_segment(start, end, spacing=spacing)
    if not distances[-1]:
        raise RequestError("The profile endpoints must be different.", "end")
    normal_x = -(end[1] - start[1]) / distances[-1]
    normal_y = (end[0] - start[0]) / distances[-1]
    offsets = np.arange(-(width // 2), width // 2 + 1)
    xs = x[None, :] + offsets[:, None] * normal_x
    ys = y[None, :] + offsets[:, None] * normal_y
    if interpolation == "nearest":
        ribbon = nearest(frame.pixels, xs, ys)
    else:
        ribbon, _ = interpolate(frame.pixels, xs, ys)
    valid = np.isfinite(ribbon)
    populated = np.sum(valid, axis=0)
    sums = np.sum(np.where(valid, ribbon, 0), axis=0, dtype=np.float64)
    averaged = np.full(len(distances), np.nan)
    np.divide(sums, populated, where=populated > 0, out=averaged)
    ix = np.floor(x + .5).astype(int)
    iy = np.floor(y + .5).astype(int)
    return {
        "length": float(distances[-1]),
        "start": list(start),
        "end": list(end),
        "interpolation": interpolation,
        "width": width,
        "samples": [
            {
                "distance": float(d), "x": float(sx), "y": float(sy),
                "original_x": int(ox), "original_y": int(oy),
                "value": finite_or_none(value), "valid": int(coverage),
            }
            for d, sx, sy, ox, oy, value, coverage in zip(distances, x, y, ix, iy, averaged, populated)
        ],
        "valid_ranges": contiguous_ranges(populated > 0),
        "unit": frame.record.unit,
    }


def local_subpixel_grid(box, steps):
    offset = (np.arange(steps, dtype=float) + .5) / steps - .5
    xs, ys = box.centers()
    return xs[:, :, None, None] + offset[None, None, :, None],         ys[:, :, None, None] + offset[None, None, None, :]


def circle_pixel_weights(box, center, inner, outer, steps=5):
    sub_x, sub_y = local_subpixel_grid(box, steps)
    squared = np.square(sub_x - center[0]) + np.square(sub_y - center[1])
    support = squared <= outer * outer
    if inner:
        support &= squared > inner * inner
    return np.mean(support, axis=(2, 3), dtype=np.float64)


def pixel_box_for_circle(center, radius):
    x, y = center
    left = floor(x - radius + .5)
    right = ceil(x + radius + .5)
    top = floor(y - radius + .5)
    bottom = ceil(y + radius + .5)
    return Box(left, top, right - left, bottom - top)


def resample_rows(image, x, y, minimum_weight=1):
    rows = max(1, 350_000 // x.shape[1])
    grid = np.full(x.shape, np.nan, dtype=np.float32)
    support = np.zeros(x.shape, dtype=np.float32)
    for start in range(0, len(grid), rows):
        part = slice(start, start + rows)
        values, fraction = interpolate(image, x[part], y[part], minimum_weight)
        grid[part] = values
        support[part] = fraction
    return grid, support
