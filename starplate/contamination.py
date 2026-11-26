"""Evidence-backed satellite-trail contamination bands.

The analysis is deliberately row oriented: for every detector row crossed by
the user's two peak anchors it looks for one faint-trail peak and the two
original-pixel shoulders belonging to that same trace.  Rows near a crossing
can be returned as undecided instead of forcing an ambiguous shoulder
assignment.
"""

from __future__ import annotations

from math import ceil, floor
from typing import Any

import numpy as np

from .errors import RequestError
from .geometry import finite_or_none


BAND_SCHEMA = "starplate.contamination-band/v1"
ROW_SCHEMA = "starplate.contamination-row/v1"


def _as_float_image(pixels: Any) -> np.ndarray:
    array = np.asarray(pixels)
    if array.ndim != 2:
        raise RequestError("Contamination analysis needs a two-dimensional image.", "image")
    return array


def estimate_noise_sigma(pixels: np.ndarray) -> float:
    """Robust image sigma; safe for synthetic images with exactly flat sky."""
    array = np.asarray(pixels)
    if array.size == 0:
        return 0.0
    step_y = max(1, int(ceil(array.shape[0] / 700)))
    step_x = max(1, int(ceil(array.shape[1] / 700)))
    sample = np.asarray(array[::step_y, ::step_x], dtype=np.float64)
    finite = sample[np.isfinite(sample)]
    if finite.size < 9:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return 1.4826 * mad


def _expected_x(start, end, y: float) -> float:
    y0 = float(start[1])
    y1 = float(end[1])
    if abs(y1 - y0) < 1e-12:
        return float(start[0]) if abs(y - y0) < 1e-9 else float(
            start[0] + (end[0] - start[0]) * (y - y0) / (y1 - y0 + 1e-12)
        )
    return float(start[0] + (end[0] - start[0]) * (y - y0) / (y1 - y0))


def _slope_x_per_row(start, end) -> float:
    dy = float(end[1]) - float(start[1])
    if abs(dy) < 1e-12:
        return 0.0
    return (float(end[0]) - float(start[0])) / dy


def _rounded_row(value: float) -> int:
    return int(floor(float(value) + 0.5))


def _side_shoulder(
    row: np.ndarray,
    peak: int,
    direction: int,
    min_drop: float,
    max_width: int,
    tolerance: float,
) -> dict[str, Any] | None:
    """Find the first base-like shoulder on one side of a peak.

    ``direction`` is -1 for image left and +1 for image right.  The returned
    coordinate is always a finite original pixel.  A shoulder is accepted only
    after the measured peak-to-shoulder drop reaches the required threshold and
    the profile has stopped materially descending.  Thus the search stops at a
    valley before a neighbouring bright line rather than crossing that line.
    """
    width = row.size
    last_value = float(row[peak])
    if not np.isfinite(last_value):
        return None

    for distance in range(1, max_width + 1):
        x = peak + direction * distance
        if x < 0 or x >= width:
            return {
                "x": None,
                "distance": distance - 1,
                "reason": "image_edge",
            }
        value = row[x]
        if not np.isfinite(value):
            return {"x": None, "distance": distance, "reason": "missing_pixel"}

        drop = last_value - float(value)
        drop_tolerance = 1e-6 * max(1.0, abs(float(min_drop)))
        if drop + drop_tolerance < min_drop:
            continue

        inward = x - direction
        outward = x + direction
        inward_ok = inward < 0 or inward >= width or not np.isfinite(row[inward]) or (
            float(row[inward]) >= float(value) - tolerance
        )
        if not inward_ok:
            continue

        if outward < 0 or outward >= width or not np.isfinite(row[outward]):
            # No contradictory outward sample: the thresholded original pixel
            # can still serve as an edge shoulder.
            return {"x": x, "value": float(value), "drop": float(drop), "distance": distance}

        outward_value = float(row[outward])
        if outward_value >= float(value) - tolerance:
            return {"x": x, "value": float(value), "drop": float(drop), "distance": distance}

    return {"x": None, "distance": max_width, "reason": "shoulder_not_found"}


def _peak_candidates(row: np.ndarray, expected_x: float, radius: int, tolerance: float) -> list[int]:
    width = row.size
    center = int(floor(expected_x + 0.5))
    lo = max(1, center - radius)
    hi = min(width - 2, center + radius)
    candidates: set[int] = set()
    if 0 <= center < width and np.isfinite(row[center]):
        candidates.add(center)

    for x in range(lo, hi + 1):
        if not np.isfinite(row[x]):
            continue
        left_ok = x == 0 or not np.isfinite(row[x - 1]) or row[x] >= row[x - 1] - tolerance
        right_ok = x == width - 1 or not np.isfinite(row[x + 1]) or row[x] >= row[x + 1] - tolerance
        if left_ok and right_ok:
            candidates.add(x)

    # The brightest pixel in a one-pixel neighbourhood of the expected trace
    # is useful on gently rounded/noisy peaks.
    near_lo = max(0, center - 1)
    near_hi = min(width, center + 2)
    finite_near = [x for x in range(near_lo, near_hi) if np.isfinite(row[x])]
    if finite_near:
        candidates.add(max(finite_near, key=lambda x: float(row[x])))

    return sorted(candidates)


def _configs_for_row(
    row: np.ndarray,
    y: int,
    start,
    end,
    *,
    peak_radius: int,
    shoulder_radius: int,
    min_drop: float,
    edge_tolerance: float,
) -> list[dict[str, Any]]:
    expected = _expected_x(start, end, y)
    configs: list[dict[str, Any]] = []
    for peak_x in _peak_candidates(row, expected, peak_radius, max(0.5, edge_tolerance * 0.5)):
        peak_value = row[peak_x]
        if not np.isfinite(peak_value):
            continue
        left = _side_shoulder(row, peak_x, -1, min_drop, shoulder_radius, edge_tolerance)
        right = _side_shoulder(row, peak_x, 1, min_drop, shoulder_radius, edge_tolerance)
        if not left or left.get("x") is None or not right or right.get("x") is None:
            continue
        lx = int(left["x"])
        rx = int(right["x"])
        if lx >= peak_x or rx <= peak_x:
            continue
        configs.append(
            {
                "row": int(y),
                "peak_x": int(peak_x),
                "left_x": lx,
                "right_x": rx,
                "peak_value": float(peak_value),
                "left_value": float(left["value"]),
                "right_value": float(right["value"]),
                "left_drop": float(left["drop"]),
                "right_drop": float(right["drop"]),
                "left_width": int(peak_x - lx),
                "right_width": int(rx - peak_x),
                "expected_x": float(expected),
                "peak_residual": float(peak_x - expected),
            }
        )
    return configs


def _median(values: list[float], default: float) -> float:
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float(default)


def _mad_scale(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return 1.4826 * float(np.median(np.abs(array - median)))


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    judged = bool(row.get("judged"))
    y = int(row["row"])
    result: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "row": y,
        "original_y": y,
        "y": y,
        "judged": judged,
        "status": "judged" if judged else "undecided",
        "reason": row.get("reason"),
        "issues": list(row.get("issues", [])),
        "expected_x": finite_or_none(row.get("expected_x")),
        "peak": None,
        "peak_x": None,
        "x": None,
        "left_shoulder": None,
        "left_x": None,
        "right_shoulder": None,
        "right_x": None,
        "peak_value": None,
        "left_value": None,
        "right_value": None,
        "left_drop_electrons": None,
        "right_drop_electrons": None,
        "left_drop": None,
        "right_drop": None,
    }
    if judged or row.get("peak_x") is not None:
        px = row.get("peak_x")
        if px is not None:
            result.update(
                peak=[float(px), float(y)], peak_x=int(px), x=int(px),
                peak_value=finite_or_none(row.get("peak_value")),
                peak_residual=finite_or_none(row.get("peak_residual")),
            )
    if judged:
        lx = int(row["left_x"])
        rx = int(row["right_x"])
        result.update(
            left_shoulder=[float(lx), float(y)], left_x=lx,
            right_shoulder=[float(rx), float(y)], right_x=rx,
            left_value=finite_or_none(row.get("left_value")),
            right_value=finite_or_none(row.get("right_value")),
            left_drop_electrons=finite_or_none(row.get("left_drop")),
            right_drop_electrons=finite_or_none(row.get("right_drop")),
            left_drop=finite_or_none(row.get("left_drop")),
            right_drop=finite_or_none(row.get("right_drop")),
            band_pixel_count=int(rx - lx + 1),
        )
    return result


def contamination_band(
    pixels: Any,
    start,
    end,
    *,
    min_drop_electrons: float = 40.0,
    drop_electrons: float | None = None,
    peak_radius: int = 10,
    row_search_radius: int | None = None,
    shoulder_radius: int = 18,
    max_half_width: int | None = None,
    trace_tolerance: float = 3.0,
    width_tolerance: float = 2.75,
    peak_ratio_limit: float = 3.5,
    edge_tolerance: float | None = None,
    noise_sigma: float | None = None,
    min_sky_samples: int = 3,
    frame_key: str | None = None,
) -> dict[str, Any]:
    """Build the maximally evidenced row-wise band between two peak anchors."""
    if drop_electrons is not None:
        min_drop_electrons = float(drop_electrons)
    if row_search_radius is not None:
        peak_radius = int(row_search_radius)
    if max_half_width is not None:
        shoulder_radius = int(max_half_width)
    if not np.isfinite(float(min_drop_electrons)) or min_drop_electrons <= 0:
        raise RequestError("The shoulder drop must be a positive number of electrons.", "drop_electrons")
    if int(min_sky_samples) < 1:
        raise RequestError("Require at least one sky sample.", "min_sky_samples")

    image = _as_float_image(pixels)
    height, width = image.shape
    start = (float(start[0]), float(start[1]))
    end = (float(end[0]), float(end[1]))
    for point, name in ((start, "start"), (end, "end")):
        if not all(np.isfinite(point)) or not (-0.5 <= point[0] <= width - 0.5 and -0.5 <= point[1] <= height - 0.5):
            raise RequestError("Contamination peak anchors must lie inside the original image.", name)
    if abs(end[0] - start[0]) + abs(end[1] - start[1]) < 1e-9:
        raise RequestError("The two contamination peaks must be different.", "end")

    if noise_sigma is None:
        noise_sigma = estimate_noise_sigma(image)
    if edge_tolerance is None:
        edge_tolerance = max(1.75, min(10.0, 1.75 * float(noise_sigma)))

    y_a = _rounded_row(start[1])
    y_b = _rounded_row(end[1])
    row_lo = max(0, min(y_a, y_b))
    row_hi = min(height - 1, max(y_a, y_b))
    if row_hi < row_lo:
        raise RequestError("The contamination line does not cross a detector row.", "start")

    raw_rows: list[dict[str, Any]] = []
    all_configs: dict[int, list[dict[str, Any]]] = {}
    for y in range(row_lo, row_hi + 1):
        row = np.asarray(image[y], dtype=np.float64)
        configs = _configs_for_row(
            row, y, start, end,
            peak_radius=max(1, int(peak_radius)),
            shoulder_radius=max(2, int(shoulder_radius)),
            min_drop=float(min_drop_electrons),
            edge_tolerance=float(edge_tolerance),
        )
        all_configs[y] = configs
        expected = _expected_x(start, end, y)
        raw_rows.append({"row": y, "expected_x": expected, "configs": configs})

    # Establish the faint line's shoulder widths from rows that already track
    # the anchor-defined trace.  Median/MAD prevent the crossing's shared or
    # bright-line shoulders from moving the template.
    near_trace: list[dict[str, Any]] = []
    for y, configs in all_configs.items():
        near_trace.extend(c for c in configs if abs(c["peak_residual"]) <= float(trace_tolerance) + 1.0)
    left_template = _median([c["left_width"] for c in near_trace], 3.0)
    right_template = _median([c["right_width"] for c in near_trace], 3.0)
    left_scale = _mad_scale([c["left_width"] for c in near_trace])
    right_scale = _mad_scale([c["right_width"] for c in near_trace])
    left_allowed = max(float(width_tolerance), 3.0 * left_scale + 0.75)
    right_allowed = max(float(width_tolerance), 3.0 * right_scale + 0.75)

    anchor_peak_levels: list[float] = []
    for y in (row_lo, row_hi):
        configs = all_configs.get(y, [])
        if configs:
            chosen = min(configs, key=lambda c: abs(c["peak_x"] - _expected_x(start, end, y)))
            anchor_peak_levels.append(chosen["peak_value"])
    peak_level = _median(anchor_peak_levels or [c["peak_value"] for cs in all_configs.values() for c in cs], 0.0)

    judged_internal: list[dict[str, Any]] = []
    for item in raw_rows:
        y = item["row"]
        expected = item["expected_x"]
        configs = item["configs"]
        if not configs:
            item.update(judged=False, reason="no_peak_shoulder_evidence", issues=["no_valid_shoulder_pair"])
            continue

        def score(c: dict[str, Any]) -> float:
            return (
                4.0 * abs(c["peak_residual"])
                + 1.25 * abs(c["left_width"] - left_template)
                + 1.25 * abs(c["right_width"] - right_template)
                + 0.002 * abs(c["peak_value"] - peak_level)
            )

        chosen = min(configs, key=score)
        issues: list[str] = []
        if abs(chosen["peak_residual"]) > float(trace_tolerance):
            issues.append("peak_off_trace")
        if abs(chosen["left_width"] - left_template) > left_allowed:
            issues.append("left_shoulder_not_on_trace")
        if abs(chosen["right_width"] - right_template) > right_allowed:
            issues.append("right_shoulder_not_on_trace")
        # A very large brightness change is a useful secondary guard against
        # jumping to a much brighter neighbouring streak.  Crossing-overlap
        # rows are normally removed by shoulder continuity before this guard.
        if peak_level > 0 and chosen["peak_value"] > float(peak_ratio_limit) * peak_level + 30.0:
            issues.append("possible_brighter_neighbour_line")

        item.update(chosen)
        item["issues"] = issues
        if issues:
            item["judged"] = False
            item["reason"] = "ambiguous_crossing_or_trace"
        else:
            item["judged"] = True
            item["reason"] = None
            judged_internal.append(item)

    # Row-to-row continuity catches a candidate that is individually plausible
    # but swaps traces for one row.  Compare with the nearest judged row on each
    # side; the anchor line supplies the expected motion across any gap.
    slope = _slope_x_per_row(start, end)
    judged_by_y = {int(r["row"]): r for r in judged_internal}
    ys = sorted(judged_by_y)
    for index, y in enumerate(ys):
        row = judged_by_y[y]
        neighbours = []
        if index > 0:
            neighbours.append(judged_by_y[ys[index - 1]])
        if index + 1 < len(ys):
            neighbours.append(judged_by_y[ys[index + 1]])
        bad: list[str] = []
        for other in neighbours:
            dy = y - int(other["row"])
            if dy == 0:
                continue
            expected_dx = slope * dy
            peak_jump = abs((int(row["peak_x"]) - int(other["peak_x"])) - expected_dx)
            left_jump = abs((int(row["left_x"]) - int(other["left_x"])) - expected_dx)
            right_jump = abs((int(row["right_x"]) - int(other["right_x"])) - expected_dx)
            gap_scale = max(1, abs(dy) ** 0.5)
            if peak_jump > (float(trace_tolerance) + 1.5) * gap_scale:
                bad.append("peak_discontinuous")
            if left_jump > (float(width_tolerance) + 1.5) * gap_scale:
                bad.append("left_shoulder_discontinuous")
            if right_jump > (float(width_tolerance) + 1.5) * gap_scale:
                bad.append("right_shoulder_discontinuous")
        if bad:
            row["judged"] = False
            row["reason"] = "trace_continuity_failed"
            row["issues"] = sorted(set(row.get("issues", []) + bad))

    judged_internal = [r for r in judged_internal if r.get("judged")]
    judged_by_y = {int(r["row"]): r for r in judged_internal}
    public_rows = [_public_row({**item, "judged": int(item["row"]) in judged_by_y}) for item in raw_rows]

    judged_y = sorted(judged_by_y)
    shoulder_points = [
        {
            "row": y,
            "left": [float(judged_by_y[y]["left_x"]), float(y)],
            "right": [float(judged_by_y[y]["right_x"]), float(y)],
        }
        for y in judged_y
    ]
    left_path = [[float(judged_by_y[y]["left_x"]), float(y)] for y in judged_y]
    right_path = [[float(judged_by_y[y]["right_x"]), float(y)] for y in judged_y]
    peak_path = [[float(judged_by_y[y]["peak_x"]), float(y)] for y in judged_y]
    undecided = [
        {
            "row": int(item["row"]),
            "point": [finite_or_none(item.get("expected_x")), float(item["row"])],
            "expected_x": finite_or_none(item.get("expected_x")),
            "reason": item.get("reason"),
            "issues": item.get("issues", []),
        }
        for item in raw_rows
        if int(item["row"]) not in judged_by_y
    ]

    anchor_start_row = row_lo if row_lo in judged_by_y else None
    anchor_end_row = row_hi if row_hi in judged_by_y else None
    adoptable = len(judged_internal) >= 2 and anchor_start_row is not None and anchor_end_row is not None

    # Connect the evidenced trace back to the exact clicked peaks.  These two
    # exact anchors are drawing/topological endpoints even when coordinate
    # rounding puts them just outside the integer row list.
    centerline = [[float(start[0]), float(start[1])], *peak_path, [float(end[0]), float(end[1])]]
    band_pixels = sum(int(judged_by_y[y]["right_x"] - judged_by_y[y]["left_x"] + 1) for y in judged_y)

    # Contiguous judged runs give safe polygon fills without bridging a
    # crossing as if its shoulders were evidenced.
    segments: list[dict[str, Any]] = []
    run: list[int] = []
    for y in range(row_lo, row_hi + 1):
        if y in judged_by_y:
            run.append(y)
        elif run:
            segments.append(_segment_payload(run, judged_by_y))
            run = []
    if run:
        segments.append(_segment_payload(run, judged_by_y))

    adoption = {
        "schema": BAND_SCHEMA,
        "kind": "satellite_contamination_adoption",
        "frame_key": frame_key,
        "start": [float(start[0]), float(start[1])],
        "end": [float(end[0]), float(end[1])],
        "row_start": row_lo,
        "row_end": row_hi,
        "min_drop_electrons": float(min_drop_electrons),
        "min_sky_samples": int(min_sky_samples),
        "peak_radius": int(peak_radius),
        "shoulder_radius": int(shoulder_radius),
        "trace_tolerance": float(trace_tolerance),
        "left_shoulder_template": finite_or_none(left_template),
        "right_shoulder_template": finite_or_none(right_template),
        "noise_sigma": finite_or_none(noise_sigma),
        "rows": public_rows,
        "judged_rows": judged_y,
        "judged_count": len(judged_y),
        "contaminated_rows": judged_y,
        "contaminated_row_count": len(judged_y),
        "evidenced_rows": judged_y,
        "undecided_rows": [int(u["row"]) for u in undecided],
        "unjudged_rows": [int(u["row"]) for u in undecided],
        "centerline": centerline,
        "peak_path": peak_path,
        "shoulders": shoulder_points,
        "left_shoulders": left_path,
        "right_shoulders": right_path,
        "undecided": undecided,
        "segments": segments,
        "band_polygons": [segment["polygon"] for segment in segments],
        "judged_row_count": len(judged_y),
        "undecided_row_count": row_hi - row_lo + 1 - len(judged_y),
        "unjudged_row_count": row_hi - row_lo + 1 - len(judged_y),
        "evidenced_sample_count": len(judged_y),
        "contaminated_pixel_count": band_pixels,
        "contaminated_pixels": band_pixels,
        "contaminated_original_pixel_count": band_pixels,
        "adoptable": bool(adoptable),
        "adopted": bool(adoptable),
        "anchor_rows_judged": {"start": anchor_start_row, "end": anchor_end_row},
    }
    return adoption


def _segment_payload(rows: list[int], judged_by_y: dict[int, dict[str, Any]]) -> dict[str, Any]:
    left = [[float(judged_by_y[y]["left_x"]), float(y)] for y in rows]
    right = [[float(judged_by_y[y]["right_x"]), float(y)] for y in rows]
    polygon = left + list(reversed(right))
    return {
        "rows": list(rows),
        "left": left,
        "right": right,
        "polygon": polygon,
        "pixel_count": sum(int(judged_by_y[y]["right_x"] - judged_by_y[y]["left_x"] + 1) for y in rows),
    }


# Friendly aliases for callers/tests that name the band operation differently.
find_contamination_band = contamination_band
build_contamination_band = contamination_band
adopt_contamination_band = contamination_band


def _clean_sky_values(
    row: np.ndarray,
    shoulder_x: int,
    side: str,
    required: int,
    maximum: int,
    noise_sigma: float,
    z_limit: float = 5.0,
    tolerance_electrons: float = 2.0,
) -> dict[str, Any]:
    direction = -1 if side == "left" else 1
    coords: list[int] = []
    values: list[float] = []
    reason = None
    # A single image-wide robust scale prevents a short run on the wing of a
    # neighbouring bright line from defining its own large dispersion and
    # declaring itself clean.  The small floor also handles noiseless fixtures.
    scale = max(float(noise_sigma), float(tolerance_electrons), 1e-6)
    x = int(shoulder_x) + direction
    for _ in range(maximum):
        if x < 0 or x >= row.size:
            reason = "image_edge"
            break
        value = row[x]
        if not np.isfinite(value):
            reason = "missing_pixel"
            break

        if values:
            baseline = float(np.median(np.asarray(values, dtype=np.float64)))
            if abs(float(value) - baseline) > z_limit * scale:
                reason = "not_clean"
                break

        coords.append(x)
        values.append(float(value))
        x += direction

    clean = len(values) >= required
    used_values = values[:required]
    used_coords = coords[:required]
    level = float(np.median(np.asarray(used_values, dtype=np.float64))) if used_values else None
    return {
        "side": side,
        "clean": clean,
        "coordinates": used_coords,
        "values": [finite_or_none(v) for v in used_values],
        "run_length": len(values),
        "level": finite_or_none(level),
        "reason": None if clean else (reason or "insufficient_contiguous_clean_samples"),
        "required_samples": required,
    }


def _coordinate_x(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("x", value.get("coordinate"))
    elif isinstance(value, (list, tuple)) and value:
        value = value[0]
    if value is None:
        return None
    try:
        return int(floor(float(value) + 0.5))
    except (TypeError, ValueError):
        return None


def measure_adoption_row(
    pixels: Any,
    adoption: dict[str, Any],
    row: int | float,
    *,
    min_sky_samples: int | None = None,
    sky_max_samples: int = 12,
    gain: float = 1.0,
    noise_sigma: float | None = None,
) -> dict[str, Any]:
    """Measure one adopted row's shoulder-to-shoulder electron excess.

    The value is explicitly ``None`` when the row is undecided or either side
    lacks the required contiguous clean sky evidence.  The original profile
    remains available to the UI independent of this scientific result.
    """
    image = _as_float_image(pixels)
    y = _rounded_row(row)
    base: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "adoption_schema": adoption.get("schema") if isinstance(adoption, dict) else None,
        "row": y,
        "original_y": y,
        "requested_row": float(row),
        "eligible": False,
        "judged": False,
        "excess_electrons": None,
        "band_sum_electrons": None,
        "sky_level_electrons_per_pixel": None,
        "band_pixel_count": None,
        "left_sky": None,
        "right_sky": None,
        "reason": None,
    }
    if not isinstance(adoption, dict) or not adoption.get("rows"):
        base["reason"] = "missing_adoption"
        return base
    if y < 0 or y >= image.shape[0]:
        base["reason"] = "row_outside_image"
        return base

    required = int(min_sky_samples if min_sky_samples is not None else adoption.get("min_sky_samples", 3))
    required = max(1, required)
    record = next(
        (
            item for item in adoption["rows"]
            if int(item.get("row", item.get("original_y", item.get("y", -999999)))) == y
        ),
        None,
    )
    if record is None:
        base["reason"] = "row_not_in_adoption"
        return base
    is_judged = record.get("judged")
    if is_judged is None:
        is_judged = record.get("status") == "judged"
    base["judged"] = bool(is_judged)
    left_value = record.get("left_x")
    right_value = record.get("right_x")
    if left_value is None:
        left_value = record.get("left_shoulder", record.get("left"))
    if right_value is None:
        right_value = record.get("right_shoulder", record.get("right"))
    lx = _coordinate_x(left_value)
    rx = _coordinate_x(right_value)
    if not is_judged or lx is None or rx is None:
        base["reason"] = "row_undecided"
        return base

    if not (0 <= lx <= rx < image.shape[1]):
        base["reason"] = "shoulders_outside_image"
        return base

    if noise_sigma is None:
        noise_sigma = estimate_noise_sigma(image)
    row_values = np.asarray(image[y], dtype=np.float64)
    left_sky = _clean_sky_values(row_values, lx, "left", required, max(required, int(sky_max_samples)), noise_sigma)
    right_sky = _clean_sky_values(row_values, rx, "right", required, max(required, int(sky_max_samples)), noise_sigma)
    base["left_sky"] = left_sky
    base["right_sky"] = right_sky
    if not left_sky["clean"] or not right_sky["clean"]:
        base["reason"] = "insufficient_sky_evidence"
        return base

    finite_band = row_values[lx:rx + 1]
    if not np.all(np.isfinite(finite_band)):
        base["reason"] = "missing_band_pixel"
        return base

    sky_level = 0.5 * (float(left_sky["level"]) + float(right_sky["level"]))
    count = int(rx - lx + 1)
    band_sum = float(np.sum(finite_band, dtype=np.float64))
    excess = (band_sum - count * sky_level) * float(gain)
    base.update(
        eligible=True,
        reason=None,
        left_x=lx,
        right_x=rx,
        peak_x=record.get("peak_x"),
        band_pixel_count=count,
        band_sum_electrons=finite_or_none(band_sum * float(gain)),
        sky_level_electrons_per_pixel=finite_or_none(sky_level * float(gain)),
        excess_electrons=finite_or_none(excess),
        # Common shorter aliases for API consumers.
        excess=finite_or_none(excess),
        value=finite_or_none(excess),
        contamination_electrons=finite_or_none(excess),
        excess_electron_count=finite_or_none(excess),
        net_electrons=finite_or_none(excess),
        unit="electron",
    )
    return base


measure_contamination_row = measure_adoption_row
row_contamination = measure_adoption_row


def handle_contamination_band(frame, fields) -> dict[str, Any]:
    start = fields.point("start")
    end = fields.point("end")
    min_drop = fields.raw("min_drop_electrons")
    if min_drop is None:
        min_drop = fields.number("drop_electrons", 40.0, 0.001, 1_000_000)
    else:
        min_drop = fields.number("min_drop_electrons", 40.0, 0.001, 1_000_000)
    peak_radius = fields.integer(
        "peak_radius", fields.integer("row_search_radius", 10, 1, 200), 1, 500
    )
    shoulder_radius = fields.integer(
        "shoulder_radius", fields.integer("max_half_width", 18, 2, 500), 2, 500
    )
    kwargs = {
        "min_drop_electrons": min_drop,
        "peak_radius": peak_radius,
        "shoulder_radius": shoulder_radius,
        "trace_tolerance": fields.number("trace_tolerance", 3.0, 0.1, 100),
        "width_tolerance": fields.number("width_tolerance", 2.75, 0.1, 100),
        "peak_ratio_limit": fields.number("peak_ratio_limit", 3.5, 1.0, 1000),
        "min_sky_samples": fields.integer("min_sky_samples", 3, 1, 50),
        "frame_key": frame.record.key,
    }
    if fields.raw("edge_tolerance") is not None:
        kwargs["edge_tolerance"] = fields.number("edge_tolerance", 2.0, 0.0, 10_000)
    adoption = contamination_band(frame.pixels, start, end, **kwargs)
    return {
        "adoption": adoption,
        "band": adoption,
        "rows": adoption["rows"],
        "judged_rows": adoption["judged_rows"],
        "undecided_rows": adoption["undecided_rows"],
        "judged_row_count": adoption["judged_row_count"],
        "undecided_row_count": adoption["undecided_row_count"],
        "contaminated_row_count": adoption["contaminated_row_count"],
        "contaminated_pixel_count": adoption["contaminated_pixel_count"],
        "adoptable": adoption["adoptable"],
        "unit": "electron",
    }


def handle_contamination_row(frame, fields) -> dict[str, Any]:
    adoption = fields.raw("adoption") or fields.raw("band")
    if not isinstance(adoption, dict):
        raise RequestError("Adopt a contamination band before reading a profile row.", "adoption")
    if adoption.get("frame_key") and adoption["frame_key"] != frame.record.key:
        raise RequestError("That contamination adoption belongs to another frame.", "adoption")
    row = fields.raw("row", fields.raw("original_y", fields.raw("y")))
    if row is None and fields.raw("sample") is not None:
        sample = fields.child("sample")
        row = sample.raw("original_y", sample.raw("y"))
    if row is None:
        point = fields.raw("point")
        if isinstance(point, dict):
            row = point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            row = point[1]
    if row is None:
        raise RequestError("Choose an original image row.", "row")
    try:
        row_number = float(row)
    except (TypeError, ValueError) as error:
        raise RequestError("Row must be a whole detector row number.", "row") from error
    gain = fields.number("gain", 1.0, 0.0, 1_000_000)
    min_sky = fields.raw("min_sky_samples")
    result = measure_adoption_row(
        frame.pixels,
        adoption,
        row_number,
        min_sky_samples=int(min_sky) if min_sky is not None else None,
        gain=gain,
    )
    return result
