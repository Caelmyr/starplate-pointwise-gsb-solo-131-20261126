"""Seam-guided mosaicking of two WCS-aligned exposures.

The reference frame keeps its own detector grid.  The companion (partner)
exposure is resampled onto that grid with the existing registration mapping,
and a single pixel-wide seam is chosen column by column across the inspection
box.  Above the seam the composite reads the reference frame; the seam row and
every row below it read the aligned companion.  A source without a usable
sample leaves the composite pixel empty (NaN).

A seam is a function that returns one row for every column, connects two
user-marked endpoints, and changes its row by at most one between adjacent
columns.  A row is eligible at a column only when the seam row and its
immediate neighbours above and below carry complete, finite original samples in
*both* frames and every one of those samples is below the user brightness
limit.  The search stays within a fixed pixel band around the user's drafted
join line.  Among all admissible seams, dynamic programming minimises a
compromise between deviation from that drafted line and the original-value
jump across the frame switch.
"""

from dataclasses import dataclass

import numpy as np

from .errors import RequestError
from .geometry import Box, finite_or_none, index_for_point, point_in_frame
from .provenance import reference_sample
from .registration import Alignment, Blend, transform_for, transformed_patch
from .sampling import interpolate, nearest
from .skycoords import world_at


BAND_RADIUS = 18
EDGE_TOLERANCE = 0.75


@dataclass(frozen=True)
class SeamSettings:
    start: tuple
    end: tuple
    brightness_limit: float
    alignment: Alignment
    blend: Blend
    band: int

    @classmethod
    def from_fields(cls, fields):
        start = fields.point("start")
        end = fields.point("end")
        limit = fields.number("brightness_limit", minimum=-1e30, maximum=1e30)
        alignment = Alignment.from_fields(fields)
        blend = Blend.from_fields(fields)
        band = fields.integer("band", BAND_RADIUS, 1, 64)
        return cls(start, end, limit, alignment, blend, band)

    def payload(self):
        return {
            "start": list(self.start),
            "end": list(self.end),
            "brightness_limit": self.brightness_limit,
            "band_radius": self.band,
            "alignment": self.alignment.payload(),
            "blend": self.blend.payload(),
        }


def _ordered_endpoints(settings, box):
    """Snap the drafted endpoints onto the left and right box columns."""
    points = sorted((settings.start, settings.end), key=lambda point: point[0])
    (x0, y0), (x1, y1) = points
    if abs(x0 - box.x) > EDGE_TOLERANCE or abs(x1 - (box.right - 1)) > EDGE_TOLERANCE:
        raise RequestError(
            "Place one seam endpoint on the left edge of the inspection box and the other on the right edge.",
            "start",
        )
    left_row = int(np.rint(y0))
    right_row = int(np.rint(y1))
    if not (box.y <= left_row < box.bottom and box.y <= right_row < box.bottom):
        raise RequestError("Both seam endpoints must stay inside the inspection box.", "start")
    return left_row, right_row


def _aligned_partner(reference, partner, box, settings):
    mapping, _details = transform_for(reference, partner, box, settings.alignment)
    aligned, support = transformed_patch(partner, box, mapping, settings.blend)
    return aligned, support, mapping


def _eligibility(reference_plane, aligned, support, settings):
    """Rows whose r-1, r, r+1 neighbourhood is fully sampled below the limit."""
    height, width = aligned.shape
    finite_ref = np.isfinite(reference_plane)
    finite_partner = np.isfinite(aligned) & (support >= settings.blend.minimum_weight)
    below_ref = finite_ref & (reference_plane < settings.brightness_limit)
    below_partner = finite_partner & (aligned < settings.brightness_limit)
    column_ok = np.zeros((height, width), dtype=bool)
    if height >= 3:
        good = below_ref & below_partner
        column_ok[1:-1] = good[:-2] & good[1:-1] & good[2:]
    return column_ok, finite_ref, finite_partner


def _guide_for(left_row, right_row, width):
    columns = np.arange(width, dtype=np.float64)
    return left_row + (right_row - left_row) * columns / max(1, width - 1)


def _candidate_bounds(column, width, height, left_row, right_row, band):
    guide = left_row + (right_row - left_row) * column / max(1, width - 1)
    # Hard band limit around the drafted join line: an integer row r is kept
    # only when |r - guide| <= band, clipped to interior rows because every
    # seam row needs an adjacent row above and below.
    low = max(1, int(np.ceil(guide - band)))
    high = min(height - 2, int(np.floor(guide + band)))
    # Keep only rows that can still connect to both fixed endpoints with unit
    # slopes.  This can narrow the band but must never widen it.
    reach_left_low = left_row - column
    reach_left_high = left_row + column
    reach_right_low = right_row - (width - 1 - column)
    reach_right_high = right_row + (width - 1 - column)
    low = max(low, reach_left_low, reach_right_low)
    high = min(high, reach_left_high, reach_right_high)
    return low, high


def _jump_scale(reference_plane, aligned, eligible):
    """Typical frame-to-frame mismatch, floored by the reference image noise."""
    finite_reference = reference_plane[np.isfinite(reference_plane)].astype(np.float64)
    reference_sigma = 1.0
    if finite_reference.size:
        median = float(np.median(finite_reference))
        reference_sigma = max(1.4826 * float(np.median(np.abs(finite_reference - median))), 1e-12)
    if np.any(eligible):
        rows, columns = np.nonzero(eligible)
        mismatch = np.abs(
            reference_plane[rows, columns].astype(np.float64)
            - aligned[rows, columns].astype(np.float64)
        )
        mismatch = mismatch[np.isfinite(mismatch)]
        if mismatch.size:
            mismatch_median = float(np.median(mismatch))
            mismatch_mad = 1.4826 * float(np.median(np.abs(mismatch - mismatch_median)))
            return max(mismatch_median + mismatch_mad, reference_sigma)
    return reference_sigma


def find_seam(reference_plane, aligned, support, settings, left_row, right_row):
    """Shortest admissible path; returns local seam rows or raises."""
    height, width = aligned.shape
    eligible, _, _ = _eligibility(reference_plane, aligned, support, settings)
    guide = _guide_for(left_row, right_row, width)
    band = max(1, settings.band)
    start_row, finish_row = left_row, right_row
    if not eligible[start_row, 0] or not eligible[finish_row, width - 1]:
        raise RequestError(
            "No usable seam: at least one endpoint lacks complete, below-limit "
            "samples in both frames at the seam row and its adjacent rows.",
            "start",
        )

    reference64 = reference_plane.astype(np.float64, copy=False)
    aligned64 = aligned.astype(np.float64, copy=False)
    jump_scale = _jump_scale(reference64, aligned64, eligible)

    infinity = np.inf
    costs = [None] * width
    parents = [None] * width
    first_low, first_high = _candidate_bounds(0, width, height, start_row, finish_row, band)
    first_rows = np.arange(first_low, first_high + 1)
    if start_row not in first_rows or not eligible[start_row, 0]:
        raise RequestError("No usable seam: the left endpoint is blocked by the sample or brightness rule.", "start")
    initial = np.full(first_rows.shape, infinity, dtype=np.float64)
    initial[first_rows == start_row] = abs(start_row - guide[0]) / band
    costs[0] = (first_rows, initial)

    for column in range(1, width):
        low, high = _candidate_bounds(column, width, height, start_row, finish_row, band)
        rows = np.arange(low, high + 1)
        node_cost = np.abs(rows - guide[column]) / band
        ok = eligible[rows, column]
        previous_rows, previous_cost = costs[column - 1]
        # Every candidate row may arrive from r-1, r or r+1 in the previous column.
        transition = np.full((rows.size, 3), infinity, dtype=np.float64)
        # The frame-switch jump belongs to the candidate seam position itself:
        # both readings describe this same detector row in the new column.
        position_jump = (
            np.abs(reference64[rows, column] - aligned64[rows, column]) / jump_scale
        )
        for offset, source_row in enumerate((rows - 1, rows, rows + 1)):
            located = np.searchsorted(previous_rows, source_row)
            inside = (located >= 0) & (located < previous_rows.size)
            inside &= previous_rows[np.clip(located, 0, previous_rows.size - 1)] == source_row
            safe_index = np.clip(located, 0, previous_rows.size - 1)
            prior = np.where(inside, previous_cost[safe_index], infinity)
            transition[:, offset] = prior + position_jump
        best_offset = np.argmin(transition, axis=1)
        best = transition[np.arange(rows.size), best_offset]
        total = np.where(ok, node_cost + best, infinity)
        costs[column] = (rows, total)
        parents[column] = (rows, best_offset)

    final_rows, final_cost = costs[width - 1]
    if not np.isfinite(final_cost[final_rows == finish_row]).any():
        raise RequestError(
            "No usable seam stays within the search band while every seam row and "
            "its adjacent rows keep complete, below-limit samples in both frames. "
            "Widen the band, raise the brightness limit or redraft the join line.",
            "brightness_limit",
        )

    seam = np.empty(width, dtype=np.int64)
    seam[width - 1] = finish_row
    for column in range(width - 1, 0, -1):
        rows, best_offset = parents[column]
        position = int(np.searchsorted(rows, seam[column]))
        # Offset 0/1/2 marks a predecessor at current-1, current or current+1.
        seam[column - 1] = seam[column] + int(best_offset[position]) - 1
    return seam, guide, jump_scale, float(band)


def compose(reference_plane, aligned, seam_rows):
    """Reference above the seam, aligned companion at and below the seam."""
    height, width = aligned.shape
    row_index = np.arange(height, dtype=np.int64)[:, None]
    use_partner = row_index >= seam_rows[None, :]
    composite = np.where(use_partner, aligned, reference_plane)
    composite = composite.astype(np.float32, copy=False)
    reference_used = (~use_partner) & np.isfinite(reference_plane)
    partner_used = use_partner & np.isfinite(aligned)
    return composite, int(np.count_nonzero(reference_used)), int(np.count_nonzero(partner_used))


def seam_report(reference, partner, box, settings):
    if reference.record.key == partner.record.key:
        raise RequestError("Choose a second image for the seam composite.", "partner")
    if reference.record.unit and partner.record.unit and reference.record.unit != partner.record.unit:
        raise RequestError("Both input images must use the same pixel units.", "partner")
    left_row, right_row = _ordered_endpoints(settings, box)
    aligned, support, mapping = _aligned_partner(reference, partner, box, settings)
    reference_plane = box.extract(reference.pixels)
    if not np.any(np.isfinite(reference_plane) & np.isfinite(aligned)):
        raise RequestError("No usable detector pixels overlap in the inspection box.", "box")
    seam_rows, guide, jump_scale, deviation_scale = find_seam(
        reference_plane, aligned, support, settings,
        left_row - box.y, right_row - box.y,
    )
    composite, reference_count, partner_count = compose(reference_plane, aligned, seam_rows)

    columns = np.arange(box.width)
    global_rows = seam_rows + box.y
    deviations = np.abs(global_rows - np.rint(guide + box.y))
    jumps = np.abs(
        reference_plane[seam_rows, columns]
        - aligned[seam_rows, columns]
    )
    finite_jumps = jumps[np.isfinite(jumps)]
    eligible, _, finite_partner = _eligibility(reference_plane, aligned, support, settings)
    return {
        "seam": [[int(box.x + column), int(row)] for column, row in zip(columns, global_rows)],
        "guide_line": [
            [box.x, left_row], [box.right - 1, right_row],
        ],
        "endpoints": {
            "left": [box.x, int(left_row)],
            "right": [box.right - 1, int(right_row)],
        },
        "search_band_radius": settings.band,
        "brightness_limit": settings.brightness_limit,
        "region": box.payload(),
        "partner": partner.record.key,
        "alignment": settings.alignment.payload(),
        "deviation": {
            "max_px": int(np.max(deviations)),
            "mean_px": float(np.mean(deviations)),
            "scale_px": deviation_scale,
        },
        "jump": {
            "max": finite_or_none(float(np.max(finite_jumps))) if finite_jumps.size else None,
            "mean": finite_or_none(float(np.mean(finite_jumps))) if finite_jumps.size else None,
            "robust_sigma_scale": finite_or_none(jump_scale),
        },
        "samples": {
            "reference_pixels": reference_count,
            "partner_pixels": partner_count,
            "empty_pixels": int(box.area - np.count_nonzero(np.isfinite(composite))),
            "eligible_seam_positions": int(np.count_nonzero(eligible)),
            "partner_valid_positions": int(np.count_nonzero(finite_partner)),
        },
        "composite": composite,
        "aligned": aligned,
        "mapping": mapping,
    }


def seam_source_report(workspace, frame, point, lineage):
    """Trace one composite pixel back to the frame that actually supplied it."""
    point_in_frame(point, frame.shape)
    x_local, y_local = point
    origin_x, origin_y = lineage.get("origin", (0, 0))
    parent_x = x_local + origin_x
    parent_y = y_local + origin_y
    parameters = lineage["parameters"]
    region = lineage.get("region_on_parent", {})
    seam = np.asarray(parameters["seam_rows"], dtype=np.int64)
    reference_key, partner_key = lineage["parents"]
    reference = workspace.open(reference_key)
    partner = workspace.open(partner_key)
    column = int(round(parent_x)) - region.get("x", 0)
    seam_row = int(seam[column]) + region.get("y", 0)
    uses_partner = parent_y >= seam_row
    result = reference_sample(frame, point)

    if not uses_partner:
        source = reference_sample(reference, (parent_x, parent_y))
        source["role"] = "reference"
        source["adopted"] = bool(source["available"])
        return {
            "result": result,
            "adopted_source": source,
            "direct_parent_samples": [source, _unadopted_partner(partner)],
            "operation": "seam",
            "switch_row": [column + region.get("x", 0), seam_row],
        }

    settings = _rebuild_settings(parameters)
    mapping, _ = transform_for(reference, partner, Box.full(reference.shape), settings.alignment)
    mapped_x, mapped_y = mapping(np.array([[parent_x]]), np.array([[parent_y]]))
    partner_point = (float(mapped_x[0, 0]), float(mapped_y[0, 0]))
    if settings.blend.interpolation == "nearest":
        values = nearest(partner.pixels, np.array([[partner_point[0]]]), np.array([[partner_point[1]]]))
        weight = np.where(np.isfinite(values), 1.0, 0.0)
    else:
        values, weight = interpolate(
            partner.pixels, np.array([[partner_point[0]]]), np.array([[partner_point[1]]]),
            settings.blend.minimum_weight,
        )
    value = finite_or_none(float(values[0, 0])) if np.isfinite(values[0, 0]) else None
    partner_pixel = index_for_point(partner_point) if value is not None else None
    sky = None
    if partner_pixel is not None:
        from .frames import celestial_wcs
        sky = world_at(celestial_wcs(partner.header), *partner_point)
    source = {
        "key": partner.record.key,
        "point": [finite_or_none(partner_point[0]), finite_or_none(partner_point[1])],
        "pixel": list(partner_pixel) if partner_pixel else None,
        "value": value,
        "sky": sky,
        "available": value is not None,
        "interpolation_support": float(weight[0, 0]),
        "unit": partner.record.unit,
        "role": "partner",
        "adopted": value is not None,
    }
    reference_not_used = reference_sample(reference, (parent_x, parent_y))
    reference_not_used["role"] = "reference"
    reference_not_used["adopted"] = False
    return {
        "result": result,
        "adopted_source": source,
        "direct_parent_samples": [reference_not_used, source],
        "operation": "seam",
        "switch_row": [column + region.get("x", 0), seam_row],
    }


def _unadopted_partner(partner):
    return {
        "key": partner.record.key,
        "point": None,
        "pixel": None,
        "value": None,
        "sky": None,
        "available": False,
        "unit": partner.record.unit,
        "role": "partner",
        "adopted": False,
    }


def _rebuild_settings(parameters):
    alignment_fields = parameters["alignment"]
    centers = tuple(tuple(point) for point in alignment_fields.get("anchors", ()))
    moving = tuple(tuple(point) for point in alignment_fields.get("moving_anchors", ()))
    alignment = Alignment(
        alignment_fields["mode"], alignment_fields["shift_x"], alignment_fields["shift_y"],
        centers, moving,
    )
    blend_fields = parameters["blend"]
    blend = Blend(
        blend_fields["mode"], blend_fields["interpolation"], blend_fields["minimum_weight"],
        blend_fields["alpha"], blend_fields["normalize"], blend_fields["require_pair"],
    )
    return SeamSettings(
        tuple(parameters["start"]), tuple(parameters["end"]),
        parameters["brightness_limit"], alignment, blend, parameters.get("band_radius", BAND_RADIUS),
    )
