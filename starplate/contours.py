from collections import defaultdict
from dataclasses import dataclass
from math import hypot

import numpy as np
from scipy.ndimage import label

from .errors import RequestError
from .geometry import Box, finite_or_none
from .statistics import finite_values, percentile_summary


CASE_EDGES = {
    1: (("l", "t"),),
    2: (("t", "r"),),
    3: (("l", "r"),),
    4: (("r", "b"),),
    6: (("t", "b"),),
    7: (("l", "b"),),
    8: (("b", "l"),),
    9: (("t", "b"),),
    11: (("r", "b"),),
    12: (("l", "r"),),
    13: (("t", "r"),),
    14: (("l", "t"),),
}
EDGE_VERTICES = {
    "t": (0, 1),
    "r": (1, 2),
    "b": (3, 2),
    "l": (0, 3),
}
VERTICES = ((0, 0), (1, 0), (1, 1), (0, 1))


@dataclass(frozen=True)
class Linelet:
    start: tuple
    end: tuple


def contour_cases(raw, threshold):
    upper_left = raw[:-1, :-1]
    upper_right = raw[:-1, 1:]
    lower_right = raw[1:, 1:]
    lower_left = raw[1:, :-1]
    valid = (
        np.isfinite(upper_left)
        & np.isfinite(upper_right)
        & np.isfinite(lower_right)
        & np.isfinite(lower_left)
    )
    pattern = np.zeros(upper_left.shape, dtype=np.uint8)
    for index, vertex in enumerate((upper_left, upper_right, lower_right, lower_left)):
        pattern |= (vertex >= threshold).astype(np.uint8) << index
    return pattern, valid


def solve_saddle(pattern, values, level):
    a, b, c, d = [float(value - level) for value in values]
    diagonal_advantage = a * c - b * d
    if diagonal_advantage == 0:
        diagonal_advantage = 1 if pattern == 5 else -1
    if diagonal_advantage > 0:
        return (("t", "r"), ("b", "l"))
    return (("l", "t"), ("r", "b"))


def edge_point(edge, x, y, values, threshold):
    left, right = EDGE_VERTICES[edge]
    a, b = values[left], values[right]
    fraction = float((threshold - a) / (b - a)) if b != a else .5
    fraction = max(0.0, min(1.0, fraction))
    dx1, dy1 = VERTICES[left]
    dx2, dy2 = VERTICES[right]
    return (x + dx1 + fraction * (dx2 - dx1), y + dy1 + fraction * (dy2 - dy1))


def trace_segments(raw, box, threshold, budget):
    patterns, valid = contour_cases(raw, threshold)
    crosses = valid & (patterns != 0) & (patterns != 15)
    rows, cols = np.nonzero(crosses)
    if len(rows) > budget:
        raise RequestError("Contour is too dense. Use a smaller region or another level.", "level")
    result = []
    for y, x in zip(rows, cols):
        pattern = patterns[y, x]
        values = (raw[y, x], raw[y, x + 1], raw[y + 1, x + 1], raw[y + 1, x])
        pairs = solve_saddle(pattern, values, threshold) if pattern in (5, 10) else CASE_EDGES[pattern]
        for before, after in pairs:
            start = edge_point(before, x + box.x, y + box.y, values, threshold)
            end = edge_point(after, x + box.x, y + box.y, values, threshold)
            if start != end:
                result.append(Linelet(start, end))
    return result, int(np.count_nonzero(~valid))


def vertex_key(point):
    return tuple(round(value, 7) for value in point)


def assemble_paths(segments):
    adjacent = defaultdict(list)
    for index, segment in enumerate(segments):
        adjacent[vertex_key(segment.start)].append((index, True))
        adjacent[vertex_key(segment.end)].append((index, False))
    remaining = set(range(len(segments)))
    paths = []
    open_vertices = [key for key, edges in adjacent.items() if len(edges) != 2]
    seed_vertices = [*open_vertices, *adjacent]
    for origin in seed_vertices:
        for edge, forward in adjacent[origin]:
            if edge not in remaining:
                continue
            current_edge, move_forward = edge, forward
            initial = segments[edge]
            trail = [initial.start if forward else initial.end]
            while current_edge in remaining:
                remaining.remove(current_edge)
                fragment = segments[current_edge]
                point = fragment.end if move_forward else fragment.start
                trail.append(point)
                key = vertex_key(point)
                if key == origin or len(adjacent[key]) != 2:
                    break
                available = [(index, onward) for index, onward in adjacent[key] if index in remaining]
                if not available:
                    break
                current_edge, move_forward = available[0]
            paths.append(trail)
    return paths


def path_statistics(points):
    closed = vertex_key(points[0]) == vertex_key(points[-1]) and len(points) > 3
    xs, ys = zip(*points)
    length = sum(hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(points, points[1:]))
    signed_area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:])) / 2 if closed else None
    return {
        "closed": closed,
        "length": float(length),
        "area": abs(float(signed_area)) if signed_area is not None else None,
        "bounds": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
    }


def isolines(frame, box, requested_levels, minimum_length=7):
    raw = box.extract(frame.pixels)
    if min(raw.shape) < 2:
        raise RequestError("Contour requires at least two rows and columns.", "box")
    levels = sorted(set(requested_levels))
    reports = []
    for level in levels:
        segments, missing = trace_segments(raw, box, level, 35_000)
        paths = assemble_paths(segments)
        boundaries = []
        for path in paths:
            summary = path_statistics(path)
            if summary["length"] < minimum_length:
                continue
            boundaries.append({
                **summary,
                "points": [[float(x), float(y)] for x, y in path],
            })
        boundaries.sort(key=lambda path: (-path["length"], path["bounds"][1], path["bounds"][0]))
        reports.append({
            "level": level,
            "discarded_invalid_cells": missing,
            "path_count": len(boundaries),
            "paths": boundaries[:60],
        })
    return {"region": box.payload(), "levels": reports, "unit": frame.record.unit}


def isophote_regions(frame, box, threshold, min_area):
    raw = box.extract(frame.pixels)
    foreground = np.isfinite(raw) & (raw >= threshold)
    labels, _ = label(foreground)
    counts = np.bincount(labels.ravel())
    eligible = np.flatnonzero(counts >= min_area)
    members = eligible[eligible > 0]
    if len(members) > 100:
        members = members[np.argsort(-counts[members], kind="stable")[:100]]
    lookup = np.zeros(len(counts), dtype=bool)
    lookup[members] = True
    area = int(np.count_nonzero(lookup[labels]))
    values = finite_values(raw[lookup[labels]])
    return {
        "region": box.payload(),
        "level": threshold,
        "component_count": len(members),
        "area": area,
        "maximum": finite_or_none(np.max(values)) if len(values) else None,
        "sum_above_level": float(np.sum(values - threshold)),
        "percentiles": percentile_summary(values),
    }
