from dataclasses import dataclass
from math import isfinite

import numpy as np

from .errors import RequestError
from .geometry import Box, finite_or_none
from .statistics import robust_summary


@dataclass(frozen=True)
class CalibrationSettings:
    dark_scale: float
    min_flat_fraction: float
    max_flat_fraction: float
    minimum_reference: int

    @classmethod
    def from_fields(cls, fields):
        minimum = fields.number("flat_min_fraction", .25, .001, 1)
        maximum = fields.number("flat_max_fraction", 2.0, 1, 100)
        return cls(
            fields.number("dark_scale", 1, 0, 200),
            minimum,
            maximum,
            fields.integer("minimum_reference", 20, 5, 50_000),
        )

    def payload(self):
        return {
            "dark_scale": self.dark_scale,
            "flat_min_fraction": self.min_flat_fraction,
            "flat_max_fraction": self.max_flat_fraction,
            "minimum_reference": self.minimum_reference,
        }


def check_plane(reference, frame, role):
    if reference is None:
        return
    if reference.record.key == frame.record.key:
        raise RequestError("The science frame cannot serve as its own calibration reference.", role)
    if reference.shape != frame.shape:
        raise RequestError("Calibration planes must match science-frame dimensions.", role)


def flat_normalization(flat, box, settings):
    values = box.extract(flat.pixels)
    positive = np.isfinite(values) & (values > 0)
    good = values[positive]
    sample = robust_summary(good, 3.2)
    if sample.retained < settings.minimum_reference or not isfinite(sample.median) or sample.median <= 0:
        raise RequestError("Flat field has too few positive usable pixels in the normalization box.", "flat")
    return float(sample.median), {
        "region": box.payload(),
        "candidate_pixels": len(good),
        "fit_pixels": sample.retained,
        "normalization": sample.median,
        "sigma": sample.sigma,
    }


def camera_correct(frame, dark, flat, box, settings):
    if dark is None and flat is None:
        raise RequestError("Choose a dark frame or a flat field to calibrate this image.", "dark")
    check_plane(dark, frame, "dark")
    check_plane(flat, frame, "flat")
    if dark is not None and flat is not None and dark.record.key == flat.record.key:
        raise RequestError("Dark and flat must use different source planes.", "flat")
    median = 1
    flat_report = None
    if flat is not None:
        median, flat_report = flat_normalization(flat, box, settings)
    shape = frame.shape
    calibrated = np.full(shape, np.nan, dtype=np.float32)
    reason_counts = dict(science=0, dark=0, flat=0, usable=0)
    increment = max(1, 180_000 // shape[1])
    for start in range(0, shape[0], increment):
        part = slice(start, start + increment)
        science = np.asarray(frame.pixels[part], dtype=np.float64)
        valid = np.isfinite(science)
        reason_counts["science"] += int(np.count_nonzero(~valid))
        if dark is not None:
            dark_section = np.asarray(dark.pixels[part], dtype=np.float64)
            dark_ok = np.isfinite(dark_section)
            reason_counts["dark"] += int(np.count_nonzero(valid & ~dark_ok))
            valid &= dark_ok
            selected_dark = np.where(dark_ok, dark_section, 0)
            science -= selected_dark * settings.dark_scale
        if flat is not None:
            flat_section = np.asarray(flat.pixels[part], dtype=np.float64) / median
            flat_ok = (
                np.isfinite(flat_section) & (flat_section >= settings.min_flat_fraction)
                & (flat_section <= settings.max_flat_fraction)
            )
            reason_counts["flat"] += int(np.count_nonzero(valid & ~flat_ok))
            valid &= flat_ok
            np.divide(science, flat_section, out=science, where=valid)
        representable = (np.abs(science) <= np.finfo(np.float32).max) & np.isfinite(science)
        valid &= representable
        reason_counts["usable"] += int(np.count_nonzero(valid))
        calibrated[part] = np.where(valid, science, np.nan)
    if not reason_counts["usable"]:
        raise RequestError("Calibration leaves no usable output pixels.", "flat")
    stats = robust_summary(box.extract(calibrated))
    return calibrated, {
        "parameters": settings.payload(),
        "shape": list(shape),
        "dark": dark.record.key if dark is not None else None,
        "flat": flat.record.key if flat is not None else None,
        "flat_reference": flat_report,
        "pixel_counts": reason_counts,
        "result_statistics": stats.payload(),
    }


def explain_correction(frame, dark, flat, point, normalization, settings):
    x, y = map(int, point)
    if not frame.box.covers((x, y)):
        raise RequestError("Calibration pixel is outside the physical detector.", "point")
    original = float(frame.pixels[y, x])
    dark_value = float(dark.pixels[y, x]) if dark is not None else 0
    flat_value = float(flat.pixels[y, x]) if flat is not None else 1
    normalized = flat_value / normalization
    use = (
        isfinite(original) and isfinite(dark_value) and isfinite(normalized)
        and settings.min_flat_fraction <= normalized <= settings.max_flat_fraction
    )
    corrected = (original - dark_value * settings.dark_scale) / normalized if use else None
    return {
        "x": x,
        "y": y,
        "science": finite_or_none(original),
        "scaled_dark": finite_or_none(dark_value * settings.dark_scale),
        "normalized_flat": finite_or_none(normalized),
        "result": finite_or_none(corrected),
    }


def stack_reference_frames(frames, box, mode):
    if not 2 <= len(frames) <= 8:
        raise RequestError("Choose two to eight calibration exposures.", "frames")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise RequestError("A master calibration frame requires matching image sizes.", "frames")
    if len({frame.record.key for frame in frames}) != len(frames):
        raise RequestError("Select each source image once.", "frames")
    stack = np.full(shape, np.nan, dtype=np.float32)
    support_minimum = (len(frames) + 1) // 2
    rows = max(1, 100_000 // shape[1])
    scales = [1.0] * len(frames)
    if mode == "normalized_flat":
        for i, frame in enumerate(frames):
            values = box.extract(frame.pixels)
            stat = robust_summary(values[np.isfinite(values) & (values > 0)])
            if stat.retained < 8:
                raise RequestError(f"Reference {frame.record.title} has too few usable normalization pixels.", "frames")
            scales[i] = stat.median
    for start in range(0, shape[0], rows):
        patch = slice(start, start + rows)
        data = np.stack([frame.pixels[patch] / scale for frame, scale in zip(frames, scales)])
        finite = np.count_nonzero(np.isfinite(data), axis=0)
        masked = np.ma.masked_invalid(data)
        values = np.ma.median(masked, axis=0).filled(np.nan)
        stack[patch] = np.where(finite >= support_minimum, values, np.nan)
    return stack, {
        "mode": mode,
        "minimum_exposures": support_minimum,
        "normalization_levels": scales,
        "statistics": robust_summary(box.extract(stack)).payload(),
    }
