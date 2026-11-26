import math
import numpy as np

from .analysis_tools import TOOLS
from .background import BackgroundSettings, background_point, estimate_background
from .contamination import handle_contamination_band, handle_contamination_row
from .contours import isolines, isophote_regions
from .errors import RequestError
from .frames import celestial_wcs
from .geometry import Box, point_in_frame
from .photometry import ApertureSettings, aperture_summary, radius_curve, recenter_aperture
from .rendering import Stretch, stretch_summary
from .sampling import line_profile, probe
from .session import selected_region
from .skycoords import lookup_world, world_at
from .statistics import bin_members, compare_regions, histogram
from .stellar import DetectionSettings, psf_shape, source_catalog


def handle_probe(frame, fields):
    point = fields.point("point")
    radius = fields.integer("radius", 1, 0, 16)
    half = fields.integer("patch", 4, 1, 10)
    result = probe(frame, point, radius, half)
    wcs = celestial_wcs(frame.header)
    result["world"] = world_at(wcs, *point)
    return result


def handle_histogram(frame, fields):
    box = selected_region(frame, fields)
    bounds = (fields.number(key) if fields.raw(key) not in (None, "") else None for key in ("low", "high"))
    return histogram(frame, box, *bounds, bins=fields.integer("bins", 96, 16, 240))


def handle_histogram_bin(frame, fields):
    box = selected_region(frame, fields)
    return bin_members(
        frame, box, fields.number("left"), fields.number("right"),
        include_upper=fields.flag("include_upper", False),
    )


def handle_compare(frame, fields):
    first = selected_region(frame, fields)
    second = Box.from_fields(fields.child("reference"), frame.shape)
    if second.area > 4_500_000:
        raise RequestError("Comparison box exceeds the supported pixel count.", "reference", 413)
    return compare_regions(frame, first, second)


def handle_profile(frame, fields):
    start = fields.point("start")
    end = fields.point("end")
    spacing = fields.number("spacing", 1, .25, 16)
    mode = fields.option("interpolation", ("nearest", "linear"), "linear")
    width = fields.integer("width", 1, 1, 15)
    if width % 2 == 0:
        raise RequestError("Profile width must be an odd number of pixels.", "width")
    result = line_profile(frame, start, end, spacing, mode, width)
    adoption = fields.raw("contamination_adoption", fields.raw("adoption"))
    if isinstance(adoption, dict):
        from .contamination import estimate_noise_sigma, measure_adoption_row

        noise_sigma = estimate_noise_sigma(frame.pixels)
        for sample in result["samples"]:
            row_report = measure_adoption_row(
                frame.pixels, adoption, sample["original_y"], noise_sigma=noise_sigma
            )
            sample["contamination_excess_electrons"] = row_report["excess_electrons"]
            sample["contamination_eligible"] = row_report["eligible"]
            sample["contamination_reason"] = row_report["reason"]
            sample["contamination"] = row_report
    return result


def handle_aperture(frame, fields):
    center = fields.point("point")
    settings = ApertureSettings.from_fields(fields)
    box = Box.from_fields(fields.child("box"), frame.shape)
    recenter = fields.flag("recenter", False)
    if recenter:
        center = point_in_frame(center, frame.shape)
        moved = recenter_aperture(frame, box, center, min(settings.radius, 6))
        if moved["usable"]:
            center = tuple(moved["point"])
    result = aperture_summary(frame, box, center, settings)
    if recenter:
        result["recenter"] = moved
    return result


def handle_curve(frame, fields):
    point = fields.point("point")
    box = Box.from_fields(fields.child("box"), frame.shape)
    settings = ApertureSettings.from_fields(fields)
    return radius_curve(
        frame, box, point, settings,
        fields.integer("samples", 12, 2, 30),
        fields.number("curve_start", 1, .5, settings.radius),
    )


def handle_stars(frame, fields):
    box = selected_region(frame, fields, limit=3_500_000)
    return source_catalog(frame, box, DetectionSettings.from_fields(fields))


def handle_psf(frame, fields):
    point = fields.point("point")
    box = Box.from_fields(fields.child("box"), frame.shape)
    return psf_shape(
        frame, box, point,
        fields.number("radius", 8, 2, 60),
        fields.number("clipping", 2.5, .5, 12),
    )


def handle_contours(frame, fields):
    box = selected_region(frame, fields, limit=1_400_000)
    raw = fields.list("levels", minimum=1, maximum=8)
    levels = []
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise RequestError("Contour levels must be numbers.", "levels") from error
        if not math.isfinite(number):
            raise RequestError("Contour levels must be finite.", "levels")
        levels.append(number)
    return isolines(frame, box, levels, fields.number("minimum_length", 7, 0, 10_000))


def handle_isophote(frame, fields):
    box = selected_region(frame, fields, limit=3_500_000)
    return isophote_regions(frame, box, fields.number("level"), fields.integer("min_area", 6, 1, 5000))


def handle_background(frame, fields):
    box = selected_region(frame, fields, limit=3_500_000)
    config = BackgroundSettings.from_fields(fields)
    model, residual, report = estimate_background(frame, box, config)
    if fields.raw("point") is not None:
        report["probe"] = background_point(frame, box, model, fields.point("point"))
    report["rms_residual"] = residual_rms(residual)
    return report


def residual_rms(values):
    finite = np.isfinite(values)
    if not finite.any():
        return None
    return float(np.sqrt(np.mean(np.square(values[finite].astype(np.float64)))))


def handle_display(frame, fields):
    box = selected_region(frame, fields)
    return stretch_summary(frame, box, Stretch.from_fields(fields.child("stretch")))


OPERATIONS = {
    **TOOLS,
    "probe": handle_probe,
    "histogram": handle_histogram,
    "histogram-bin": handle_histogram_bin,
    "compare": handle_compare,
    "profile": handle_profile,
    "contamination-band": handle_contamination_band,
    "contamination": handle_contamination_band,
    "satellite": handle_contamination_band,
    "satellite-trail": handle_contamination_band,
    "satellite-contamination": handle_contamination_band,
    "satellite-band": handle_contamination_band,
    "streak": handle_contamination_band,
    "streak-band": handle_contamination_band,
    "contamination-row": handle_contamination_row,
    "contamination-flux": handle_contamination_row,
    "satellite-row": handle_contamination_row,
    "satellite-excess": handle_contamination_row,
    "aperture": handle_aperture,
    "aperture-curve": handle_curve,
    "stars": handle_stars,
    "psf": handle_psf,
    "contours": handle_contours,
    "isophote": handle_isophote,
    "background": handle_background,
    "world": lookup_world,
    "display": handle_display,
}


def run_measurement(frame, operation, fields):
    if operation not in OPERATIONS:
        raise RequestError("Choose a known measurement.", "operation", 404)
    result = OPERATIONS[operation](frame, fields)
    return {"operation": operation, "frame_key": frame.record.key, "result": result}
