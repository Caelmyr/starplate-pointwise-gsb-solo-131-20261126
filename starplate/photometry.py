from dataclasses import dataclass
from math import hypot, pi, sqrt

import numpy as np

from .errors import RequestError
from .geometry import Box, finite_or_none, point_in_frame
from .sampling import circle_pixel_weights, pixel_box_for_circle
from .statistics import robust_summary, trimmed_indices


@dataclass(frozen=True)
class ApertureSettings:
    radius: float
    sky_inner: float
    sky_outer: float
    min_aperture_coverage: float
    min_sky_coverage: float
    clip_sigma: float
    gain: float

    @classmethod
    def from_fields(cls, fields):
        radius = fields.number("radius", 4.5, 1.0, 55.0)
        sky_inner = fields.number("sky_inner", 7.5, radius + 0.5, 85.0)
        sky_outer = fields.number("sky_outer", 12.5, sky_inner + 1.0, 104.0)
        return cls(
            radius,
            sky_inner,
            sky_outer,
            fields.number("min_aperture_coverage", .9, .2, 1.0),
            fields.number("min_sky_coverage", .65, .2, 1.0),
            fields.number("clip_sigma", 3, 1.5, 8),
            fields.number("gain", 0, 0, 1e7),
        )

    def payload(self):
        return {
            "radius": self.radius,
            "sky_inner": self.sky_inner,
            "sky_outer": self.sky_outer,
            "min_aperture_coverage": self.min_aperture_coverage,
            "min_sky_coverage": self.min_sky_coverage,
            "clip_sigma": self.clip_sigma,
            "gain": self.gain,
        }


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    at = int(np.searchsorted(np.cumsum(sorted_weights), np.sum(sorted_weights) / 2))
    return float(sorted_values[min(at, len(sorted_values) - 1)])


def sky_estimate(values, weights, sigma):
    good = np.isfinite(values) & (weights > 0)
    kept_values = values[good].astype(np.float64)
    kept_weights = weights[good]
    if len(kept_values) < 8:
        return None
    clipped = trimmed_indices(kept_values, sigma=sigma)
    selected = kept_values[clipped]
    selected_weights = kept_weights[clipped]
    if len(selected) < 8 or selected_weights.sum() < 5:
        return None
    level = weighted_median(selected, selected_weights)
    scatter = 1.4826 * weighted_median(np.abs(selected - level), selected_weights)
    total = float(np.sum(selected_weights))
    effective = total * total / float(np.dot(selected_weights, selected_weights))
    return {
        "level": level,
        "sigma": scatter,
        "input_pixels": len(kept_values),
        "retained_pixels": len(selected),
        "weight": total,
        "effective_pixels": effective,
    }


def aperture_footprints(domain, point, settings: ApertureSettings):
    clipped = domain.intersect(pixel_box_for_circle(point, settings.sky_outer))
    if clipped is None:
        raise RequestError("No aperture support lies in the current inspection box.", "point")
    aperture_weights = circle_pixel_weights(clipped, point, 0, settings.radius)
    sky_weights = circle_pixel_weights(clipped, point, settings.sky_inner, settings.sky_outer)
    return clipped, aperture_weights, sky_weights


def region_coverage(weights, valid, theoretical_area):
    recorded = float(np.sum(weights[valid], dtype=np.float64))
    return min(1.0, recorded / theoretical_area), recorded


def aperture_summary(frame, box, center, settings: ApertureSettings):
    point_in_frame(center, frame.shape)
    if not box.covers(center):
        raise RequestError("Aperture center is outside the current inspection box.", "point")
    footprint, aperture, annulus = aperture_footprints(box, center, settings)
    pixels = footprint.extract(frame.pixels)
    valid = np.isfinite(pixels)
    radius = settings.radius
    aperture_coverage, area = region_coverage(aperture, valid, pi * radius * radius)
    sky_area = pi * (settings.sky_outer ** 2 - settings.sky_inner ** 2)
    sky_coverage, annulus_area = region_coverage(annulus, valid, sky_area)
    has_aperture = valid & (aperture > 0)
    raw_sum = float(np.dot(pixels[has_aperture].astype(np.float64), aperture[has_aperture]))
    sky = sky_estimate(pixels, annulus, settings.clip_sigma)
    flux = error = signal_to_noise = None
    reasons = []
    if aperture_coverage < settings.min_aperture_coverage:
        reasons.append("aperture_support")
    if sky_coverage < settings.min_sky_coverage:
        reasons.append("sky_support")
    if sky is None:
        reasons.append("sky_unavailable")
    if not reasons:
        flux = raw_sum - sky["level"] * area
        variance = aperture_variance(aperture[has_aperture], area, sky, flux, settings.gain)
        error = sqrt(variance)
        signal_to_noise = flux / error if error > 0 else None
    raw_values = pixels[has_aperture]
    return {
        "point": list(center),
        "region": box.payload(),
        "footprint": footprint.payload(),
        "aperture_coverage": aperture_coverage,
        "sky_coverage": sky_coverage,
        "aperture_area": area,
        "annulus_area": annulus_area,
        "aperture_pixels": int(np.count_nonzero(has_aperture)),
        "input_peak": finite_or_none(np.max(raw_values)) if len(raw_values) else None,
        "raw_sum": raw_sum,
        "flux": flux,
        "uncertainty": error,
        "snr": signal_to_noise,
        "sky": sky,
        "status": "usable" if not reasons else "insufficient_support",
        "reasons": reasons,
        "settings": settings.payload(),
        "unit": frame.record.unit,
    }


def aperture_variance(weights, area, sky, flux, gain):
    weighted_area = float(np.dot(weights, weights))
    effective = sky["effective_pixels"]
    background_term = sky["sigma"] ** 2 * (weighted_area + pi / 2 * area ** 2 / effective)
    shot_noise = max(0.0, flux) / gain if gain > 0 else 0.0
    return max(0.0, background_term + shot_noise)


def radius_curve(frame, box, center, settings, samples, minimum):
    if minimum >= settings.radius:
        raise RequestError("Curve start must be smaller than the final radius.", "curve_start")
    radii = np.linspace(minimum, settings.radius, samples)
    results = []
    for radius in radii:
        selected = ApertureSettings(
            float(radius), settings.sky_inner, settings.sky_outer,
            settings.min_aperture_coverage, settings.min_sky_coverage, settings.clip_sigma, settings.gain,
        )
        summary = aperture_summary(frame, box, center, selected)
        results.append({
            "radius": float(radius),
            "flux": summary["flux"],
            "uncertainty": summary["uncertainty"],
            "coverage": summary["aperture_coverage"],
        })
    final = results[-1]["flux"]
    normalized = [
        {"relative_flux": row["flux"] / final if final and row["flux"] is not None else None, **row}
        for row in results
    ]
    return {"point": list(center), "measurements": normalized, "sky_radius": settings.sky_outer}


def recenter_aperture(frame, box, guess, radius):
    search = box.intersect(Box.around(guess, radius))
    if search is None:
        return {"point": list(guess), "shift": None, "usable": False}
    values = search.extract(frame.pixels)
    background = robust_summary(values, 2.8, asymmetric=(3, 2)).median
    xs, ys = search.centers()
    support = np.square(xs - guess[0]) + np.square(ys - guess[1]) <= radius ** 2
    if not np.isfinite(background):
        return {"point": list(guess), "shift": None, "usable": False}
    weights = np.where(np.isfinite(values) & support, np.maximum(0, values - background), 0)
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0:
        return {"point": list(guess), "shift": None, "usable": False}
    point = (
        float(np.sum(weights * xs, dtype=np.float64) / total),
        float(np.sum(weights * ys, dtype=np.float64) / total),
    )
    return {
        "point": list(point),
        "start": list(guess),
        "shift": hypot(point[0] - guess[0], point[1] - guess[1]),
        "usable": True,
    }
