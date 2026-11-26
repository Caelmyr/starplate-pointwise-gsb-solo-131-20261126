from dataclasses import dataclass
from math import cos, degrees, hypot, pi, radians, sin, sqrt

import numpy as np
from scipy.optimize import least_squares

from .errors import RequestError
from .geometry import Box, finite_or_none, point_in_frame
from .photometry import recenter_aperture
from .statistics import median_sigma, robust_summary


FWHM = 2 * sqrt(2 * np.log(2))


@dataclass(frozen=True)
class FitSettings:
    radius: int
    fix_center: bool
    background: str
    max_offset: float
    positive: bool

    @classmethod
    def from_fields(cls, fields):
        radius = fields.integer("radius", 8, 4, 30)
        return cls(
            radius, fields.flag("fix_center", False),
            fields.option("background", ("constant", "plane"), "constant"),
            fields.number("max_offset", 3, .1, radius / 2),
            fields.flag("positive", True),
        )


def profile_coordinates(data, middle, x, y, radius):
    sx = x - middle[0]
    sy = y - middle[1]
    distance = sx * sx + sy * sy
    valid = np.isfinite(data)
    ring = valid & (distance >= (radius * .78) ** 2)
    use = valid & (distance <= radius ** 2)
    return sx, sy, use, ring


def gaussian_initial(data, x, y, use, ring, config):
    background = robust_summary(data[ring], asymmetric=(3.0, 2.5))
    if background.retained < 8:
        background = robust_summary(data[use], asymmetric=(3.0, 2.5))
    if background.count < 16:
        raise RequestError("The fit needs more usable original pixels.", "point")
    level = background.median
    scatter = max(background.sigma, 1e-7)
    weights = np.where(use, np.maximum(0, data - level), 0)
    strength = float(np.max(weights))
    if not config.positive:
        low = float(np.min(data[use]) - level)
        if abs(low) > strength:
            strength = low
    if abs(strength) <= np.finfo(float).eps:
        raise RequestError("No measurable peak rises above the selected local background.", "point")
    summed = float(np.sum(weights))
    if summed > 0:
        scale = sqrt(float(np.sum(weights * (x * x + y * y))) / (summed * 2))
    else:
        scale = 1.5
    scale = min(max(scale, .8), config.radius / 2)
    return np.array([strength, 0, 0, scale, scale * .92, 0, level, 0, 0]), scatter


def parameter_limits(parameters, config, data):
    max_level = max(float(np.max(np.abs(data))) * 10, 1)
    center_range = config.max_offset if not config.fix_center else 1e-7
    gradient = max_level if config.background == "plane" else 1e-10
    lower = np.array([
        0 if config.positive else -max_level, -center_range, -center_range,
        .3, .3, -pi, -max_level, -gradient, -gradient,
    ])
    upper = np.array([
        max_level, center_range, center_range, config.radius, config.radius,
        pi, max_level, gradient, gradient,
    ])
    return np.clip(parameters, lower + 1e-12, upper - 1e-12), lower, upper


def model_pixels(parameters, x, y):
    amplitude, cx, cy, major, minor, theta, level, dx, dy = parameters
    parallel = cos(theta) * (x - cx) + sin(theta) * (y - cy)
    normal = -sin(theta) * (x - cx) + cos(theta) * (y - cy)
    signal = amplitude * np.exp(-.5 * ((parallel / major) ** 2 + (normal / minor) ** 2))
    return level + dx * x + dy * y + signal


def optimize_gaussian(data, x, y, use, starting, sigma, config):
    selected = data[use]
    if len(selected) < 20:
        raise RequestError("At least 20 supported pixels are required for a usable Gaussian fit.", "point")
    xx = np.broadcast_to(x, data.shape)[use]
    yy = np.broadcast_to(y, data.shape)[use]
    initial, low, high = parameter_limits(starting, config, selected)
    answer = least_squares(
        lambda parameters: (model_pixels(parameters, xx, yy) - selected) / sigma,
        initial, bounds=(low, high), loss="soft_l1", f_scale=2.0,
        max_nfev=180, x_scale="jac",
    )
    return answer, selected, xx, yy


def report_fit(answer, x, y, data, center, domain, sigma, config):
    amplitude, cx, cy, major, minor, theta, level, dx, dy = answer.x
    if minor > major:
        major, minor = minor, major
        theta += pi / 2
    expected = model_pixels(answer.x, x, y)
    error = data - expected
    fwhm_major = float(major * FWHM)
    fwhm_minor = float(minor * FWHM)
    return {
        "selection": list(center),
        "box": domain.payload(),
        "converged": bool(answer.success),
        "status": str(answer.message),
        "center": [float(center[0] + cx), float(center[1] + cy)],
        "center_shift": float(hypot(cx, cy)),
        "major_px": fwhm_major,
        "minor_px": fwhm_minor,
        "area_px": float(2 * pi * major * minor),
        "angle_deg": float(degrees(theta) % 180),
        "ellipticity": float(1 - minor / major),
        "height": float(amplitude),
        "analytic_flux": float(2 * pi * major * minor * amplitude),
        "background": {"level": float(level), "dx": float(dx), "dy": float(dy), "sigma": sigma},
        "support": len(data),
        "rms_residual": float(sqrt(np.mean(error ** 2))),
        "residual": robust_summary(error).payload(),
        "fitted_pixels": radial_samples(answer.x, x, y, data, expected, center),
    }


def radial_samples(params, x, y, values, model, center):
    distances = np.hypot(x - params[1], y - params[2])
    order = np.argsort(distances, kind="stable")
    if len(order) > 200:
        order = order[np.linspace(0, len(order) - 1, 200, dtype=int)]
    return [
        {
            "x": float(x[index] + center[0]),
            "y": float(y[index] + center[1]),
            "r": float(distances[index]),
            "original": finite_or_none(values[index]),
            "model": finite_or_none(model[index]),
        }
        for index in order
    ]


def fit_stellar_profile(frame, box, point, settings):
    point_in_frame(point, frame.shape)
    if not box.covers(point):
        raise RequestError("Mark a fit center inside the inspection box.", "point")
    domain = box.intersect(Box.around(point, settings.radius))
    pixels = domain.extract(frame.pixels)
    x, y = domain.centers()
    tx, ty, use, ring = profile_coordinates(pixels, point, x, y, settings.radius)
    first, sigma = gaussian_initial(pixels, tx, ty, use, ring, settings)
    answer, data, sample_x, sample_y = optimize_gaussian(pixels, tx, ty, use, first, sigma, settings)
    report = report_fit(answer, sample_x, sample_y, data, point, domain, sigma, settings)
    report["unit"] = frame.record.unit
    return report
