from math import isfinite

from .astrometry import local_wcs_error, measure_segment, observation_info, sky_footprint
from .diagnostics import DefectSettings, contrast_by_cell, hot_cold
from .errors import RequestError
from .filtering import local_extrema
from .geometry import Box
from .quality import QualitySettings, grade_quality, noise_profile, shape_map
from .regions import RegionSelection, aperture_region, centroid_samples, compare_annular_bands, region_mask
from .seeing import FitSettings, fit_stellar_profile
from .session import selected_region
from .stellar import DetectionSettings
from .strip import StripSettings, strip_point, strip_read


def quality_measure(frame, fields):
    box = selected_region(frame, fields, limit=3_400_000)
    return grade_quality(frame, box, QualitySettings.from_fields(fields))


def detector_noise(frame, fields):
    box = selected_region(frame, fields, limit=3_400_000)
    settings = QualitySettings.from_fields(fields)
    axis = fields.option("axis", ("row", "column"), "row")
    cutoff = fields.number("seam_sigma", 2.7, .5, 10)
    return noise_profile(frame, box, settings, axis, cutoff)


def detector_anomalies(frame, fields):
    box = selected_region(frame, fields, limit=1_300_000)
    config = DefectSettings.from_fields(fields)
    if box.area > 650_000 and config.neighborhood > 9:
        raise RequestError("Use a smaller region for this anomaly neighborhood.", "box")
    return hot_cold(frame, box, config)


def contrast_cells(frame, fields):
    box = selected_region(frame, fields, limit=3_500_000)
    size = fields.integer("cell", 80, 16, 500)
    low = fields.number("low_pct", 5, 0, 49.99)
    high = fields.number("high_pct", 95, 50.01, 100)
    return contrast_by_cell(frame, box, size, low, high)


def star_shapes(frame, fields):
    box = selected_region(frame, fields, limit=3_400_000)
    config = DetectionSettings.from_fields(fields)
    return shape_map(
        frame, box, config,
        fields.number("minimum_minor", 1.7, .2, 12),
        fields.number("elongated", .16, .02, .95),
    )


def fit_source(frame, fields):
    fit = FitSettings.from_fields(fields)
    box = Box.from_fields(fields.child("box"), frame.shape)
    return fit_stellar_profile(frame, box, fields.point("point"), fit)


def region_photometry(frame, fields):
    box = selected_region(frame, fields, limit=2_600_000)
    region = RegionSelection.from_fields(frame, fields, box)
    return aperture_region(frame, region)


def patch_centroids(frame, fields):
    box = selected_region(frame, fields, limit=1_600_000)
    selection = RegionSelection.from_fields(frame, fields, box)
    mask = region_mask(selection)
    level = fields.number("level", .1)
    limit = fields.integer("limit", 20, 1, 100)
    groups = centroid_samples(box.extract(frame.pixels), box, mask, level, limit)
    return {"region": box.payload(), "level": level, "sources": groups}


def rings(frame, fields):
    box = selected_region(frame, fields, limit=2_600_000)
    radii = (
        fields.number("first", 5, .5, 50),
        fields.number("middle", 11, 1, 100),
        fields.number("last", 20, 2, 180),
    )
    return compare_annular_bands(frame, box, fields.point("point"), radii, fields.number("sky", 0))


def extrema(frame, fields):
    box = selected_region(frame, fields, limit=2_500_000)
    sign = fields.option("sign", ("highest", "lowest"), "highest")
    distance = fields.integer("isolation", 4, 1, 32)
    cutoff = fields.number("level", .12 if sign == "highest" else -.07)
    return local_extrema(frame, box, cutoff, distance, fields.integer("limit", 40, 1, 350), sign)


def scale_grid(frame, fields):
    xs = fields.integer("columns", 5, 2, 8)
    ys = fields.integer("rows", 4, 2, 8)
    return local_wcs_error(frame, xs, ys)


def footprint(frame, fields):
    box = Box.from_fields(fields.child("box"), frame.shape)
    if min(box.width, box.height) < 2:
        raise RequestError("A footprint needs at least a two-by-two detector region.", "box")
    return sky_footprint(frame, box, fields.integer("samples", 12, 3, 30))


def observation(frame, fields):
    return observation_info(frame)


def ribbon_summary(frame, fields):
    box = Box.from_fields(fields.child("box"), frame.shape)
    config = StripSettings.from_fields(fields)
    start = fields.point("start")
    end = fields.point("end")
    if fields.raw("sample") is not None:
        query = fields.child("sample")
        along = query.integer("along", minimum=0, maximum=1399)
        offset = query.integer("offset", minimum=0, maximum=2 * config.width)
        return strip_point(frame, box, start, end, config, along, offset)
    return strip_read(frame, box, start, end, config)


TOOLS = {
    "quality": quality_measure,
    "detector-noise": detector_noise,
    "defects": detector_anomalies,
    "contrast-cells": contrast_cells,
    "source-shape-map": star_shapes,
    "fit-source": fit_source,
    "region-flux": region_photometry,
    "region-centroids": patch_centroids,
    "rings": rings,
    "extrema": extrema,
    "sky-distance": measure_segment,
    "wcs-scale": scale_grid,
    "footprint": footprint,
    "observation": observation,
    "strip": ribbon_summary,
}
