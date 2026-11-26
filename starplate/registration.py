from dataclasses import dataclass
from math import atan2, degrees, hypot
import warnings

from astropy.wcs import NoConvergence
import numpy as np
from scipy.ndimage import gaussian_filter

from .errors import RequestError
from .frames import celestial_wcs
from .geometry import Box, finite_or_none, point_in_frame
from .sampling import interpolate, nearest
from .statistics import finite_values, median_sigma, robust_summary


@dataclass(frozen=True)
class Alignment:
    mode: str
    shift_x: float
    shift_y: float
    centers: tuple
    partner_centers: tuple

    @classmethod
    def from_fields(cls, fields):
        mode = fields.option("alignment", ("wcs", "shift", "anchors", "phase"), "wcs")
        anchors = moving = ()
        dx = dy = 0
        if mode in ("shift", "phase"):
            dx = fields.number("shift_x", 0, -8192, 8192)
            dy = fields.number("shift_y", 0, -8192, 8192)
        elif mode == "anchors":
            anchors = fields.point("anchor_1"), fields.point("anchor_2")
            moving = fields.point("moving_1"), fields.point("moving_2")
        return cls(mode, dx, dy, anchors, moving)

    def payload(self):
        return {
            "mode": self.mode,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
            "anchors": self.centers,
            "moving_anchors": self.partner_centers,
        }


@dataclass(frozen=True)
class Blend:
    mode: str
    interpolation: str
    minimum_weight: float
    alpha: float
    normalize: str
    require_pair: bool

    @classmethod
    def from_fields(cls, fields):
        return cls(
            fields.option("blend", ("mean", "difference", "maximum", "reference"), "mean"),
            fields.option("interpolation", ("linear", "nearest"), "linear"),
            fields.number("support", .999, .5, 1),
            fields.number("alpha", .5, 0, 1),
            fields.option("normalize", ("none", "offset", "affine"), "none"),
            fields.flag("require_pair", False),
        )

    def payload(self):
        return {
            "mode": self.mode,
            "interpolation": self.interpolation,
            "minimum_weight": self.minimum_weight,
            "alpha": self.alpha,
            "normalize": self.normalize,
            "require_pair": self.require_pair,
        }


@dataclass(frozen=True)
class Similarity:
    a: float
    b: float
    tx: float
    ty: float

    def evaluate(self, x, y):
        return self.a * x - self.b * y + self.tx, self.b * x + self.a * y + self.ty

    def payload(self):
        return {
            "scale": hypot(self.a, self.b),
            "angle": degrees(atan2(self.b, self.a)),
            "shift_x": self.tx,
            "shift_y": self.ty,
        }


def anchored_similarity(reference, moving, settings):
    origin, end = settings.centers
    target, target_end = settings.partner_centers
    for index, point in enumerate((origin, end), start=1):
        point_in_frame(point, reference.shape, f"anchor_{index}")
    for index, point in enumerate((target, target_end), start=1):
        point_in_frame(point, moving.shape, f"moving_{index}")
    source_delta = complex(end[0] - origin[0], end[1] - origin[1])
    target_delta = complex(target_end[0] - target[0], target_end[1] - target[1])
    if min(abs(source_delta), abs(target_delta)) < 4:
        raise RequestError("Choose anchors at least four pixels apart in each image.", "anchor_2")
    coefficient = target_delta / source_delta
    if not .2 <= abs(coefficient) <= 5:
        raise RequestError("Anchor pairs imply a scale outside the supported range.", "moving_2")
    translation = complex(*target) - coefficient * complex(*origin)
    return Similarity(coefficient.real, coefficient.imag, translation.real, translation.imag)


def centered_patch(image, box):
    array = box.extract(image)
    finite = np.isfinite(array)
    if np.count_nonzero(finite) < max(array.size // 2, 64):
        raise RequestError("Alignment needs at least half valid pixels in both comparison windows.", "box")
    median, scatter = median_sigma(finite_values(array))
    filled = np.where(finite, array, median).astype(np.float64)
    highpass = filled - gaussian_filter(filled, 3.0, mode="reflect")
    scale = max(scatter * 6, float(np.std(highpass)) / 2, 1e-10)
    highpass = np.clip(highpass / scale, -2, 2)
    h, w = array.shape
    taper = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    return highpass * taper


def phase_peak(first, second, maximum):
    fa = np.fft.rfft2(first)
    fb = np.fft.rfft2(second)
    product = fb * np.conjugate(fa)
    magnitude = abs(product)
    np.divide(product, magnitude, where=magnitude > 1e-12, out=product)
    correlation = np.fft.irfft2(product, s=first.shape)
    correlation = np.fft.fftshift(correlation)
    h, w = first.shape
    cx, cy = w // 2, h // 2
    window = correlation[
        max(0, cy - maximum):min(h, cy + maximum + 1),
        max(0, cx - maximum):min(w, cx + maximum + 1),
    ]
    peak = np.unravel_index(np.argmax(window), window.shape)
    y = peak[0] + max(0, cy - maximum)
    x = peak[1] + max(0, cx - maximum)
    confidence = phase_confidence(window, peak)
    dx = x - cx + subpixel_peak(correlation[y], x)
    dy = y - cy + subpixel_peak(correlation[:, x], y)
    return dx, dy, confidence


def phase_confidence(window, peak):
    comparison = np.ones(window.shape, dtype=bool)
    y, x = peak
    comparison[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = False
    if np.count_nonzero(comparison) < 8:
        return None
    values = window[comparison]
    md = np.median(values)
    scale = np.std(values)
    return finite_or_none((window[y, x] - md) / scale) if scale > 0 else None


def subpixel_peak(values, point):
    if point == 0 or point >= len(values) - 1:
        return 0.0
    before, middle, after = values[point - 1:point + 2]
    bottom = before - 2 * middle + after
    if bottom >= 0:
        return 0.0
    fraction = .5 * (before - after) / bottom
    return float(np.clip(fraction, -.5, .5))


def phase_shift(reference, moving, box, near_x, near_y):
    shift_x = round(near_x)
    shift_y = round(near_y)
    overlap = box.intersect(moving.box.offset(-shift_x, -shift_y))
    if overlap is None or min(overlap.height, overlap.width) < 32:
        raise RequestError("Shift search needs a common window of at least 32 by 32 pixels.", "box")
    if overlap.area > 1_500_000:
        raise RequestError("Use a smaller region for phase-correlation alignment.", "box")
    first = centered_patch(reference.pixels, overlap)
    other_box = overlap.offset(shift_x, shift_y)
    second = centered_patch(moving.pixels, other_box)
    dx, dy, confidence = phase_peak(first, second, min(75, min(first.shape) // 4))
    if confidence is None or confidence < 3.5:
        raise RequestError("Phase correlation did not find a clear common shift.", "alignment")
    return Similarity(1, 0, dx + shift_x, dy + shift_y), confidence


def wcs_mapping(reference, moving):
    first_wcs = celestial_wcs(reference.header)
    second_wcs = celestial_wcs(moving.header)
    if first_wcs is None or second_wcs is None:
        raise RequestError("WCS alignment requires a celestial solution on each image.", "alignment")

    def map_to_partner(x, y):
        try:
            world = first_wcs.pixel_to_world(x, y)
            other_x, other_y = second_wcs.world_to_pixel(world)
            return np.asarray(other_x), np.asarray(other_y)
        except NoConvergence as error:
            solution = error.best_solution
            if solution is None or solution.shape != (x.size, 2):
                return np.full_like(x, np.nan), np.full_like(y, np.nan)
            best = np.asarray(solution, dtype=np.float64).copy()
            if error.divergent is not None:
                best[error.divergent] = np.nan
            if error.slow_conv is not None:
                best[error.slow_conv] = np.nan
            return best[:, 0].reshape(x.shape), best[:, 1].reshape(y.shape)

    return map_to_partner


def transform_for(reference, moving, box, settings):
    if reference.record.key == moving.record.key:
        raise RequestError("Choose a second image for registration.", "partner")
    details = {"mode": settings.mode}
    if settings.mode == "wcs":
        return wcs_mapping(reference, moving), details
    if settings.mode == "anchors":
        transform = anchored_similarity(reference, moving, settings)
    elif settings.mode == "phase":
        transform, confidence = phase_shift(reference, moving, box, settings.shift_x, settings.shift_y)
        details["confidence"] = confidence
    else:
        transform = Similarity(1, 0, settings.shift_x, settings.shift_y)
    return transform.evaluate, {**details, **transform.payload()}


def transformed_patch(frame, box, map_to_partner, settings):
    height, width = box.height, box.width
    aligned = np.full((height, width), np.nan, dtype=np.float32)
    support = np.full(aligned.shape, 0, dtype=np.float32)
    stride = max(1, 110_000 // width)
    x_range = np.arange(box.x, box.right, dtype=float)
    for start in range(0, height, stride):
        stop = min(height, start + stride)
        y_range = np.arange(box.y + start, box.y + stop, dtype=float)
        x, y = np.meshgrid(x_range, y_range)
        target_x, target_y = map_to_partner(x, y)
        if settings.interpolation == "nearest":
            possible = np.isfinite(target_x) & np.isfinite(target_y)
            original = nearest(
                frame.pixels, np.where(possible, target_x, -1), np.where(possible, target_y, -1)
            )
            weights = np.isfinite(original).astype(np.float32)
        else:
            original, weights = interpolate(frame.pixels, target_x, target_y, settings.minimum_weight)
        aligned[start:stop] = original
        support[start:stop] = weights
    return aligned, support


def photometric_normalization(original, aligned, common, mode):
    if mode == "none":
        return 1, 0
    first = original[common].astype(np.float64)
    second = aligned[common].astype(np.float64)
    if len(first) < 30:
        raise RequestError("Level matching needs at least 30 common source pixels.", "normalize")
    first_median, first_sigma = median_sigma(first)
    second_median, second_sigma = median_sigma(second)
    factor = first_sigma / second_sigma if mode == "affine" and second_sigma > 0 and first_sigma > 0 else 1
    if factor > 30 or factor < 1 / 30:
        raise RequestError("Images have too different a scale for affine level matching.", "normalize")
    return factor, first_median - factor * second_median


def blend_registered(first, second, settings):
    valid_first = np.isfinite(first)
    valid_second = np.isfinite(second)
    both = valid_first & valid_second
    first_only = valid_first & ~valid_second
    second_only = ~valid_first & valid_second
    result = np.full(first.shape, np.nan, dtype=np.float32)
    if settings.mode == "mean":
        result[both] = (1 - settings.alpha) * first[both] + settings.alpha * second[both]
    elif settings.mode == "difference":
        result[both] = first[both] - second[both]
    elif settings.mode == "maximum":
        result[both] = np.maximum(first[both], second[both])
    else:
        result[both] = second[both]
    if not settings.require_pair and settings.mode != "difference":
        if settings.mode != "reference":
            result[first_only] = first[first_only]
        result[second_only] = second[second_only]
    return result, {
        "both": int(np.count_nonzero(both)),
        "reference_only": int(np.count_nonzero(first_only)),
        "partner_only": int(np.count_nonzero(second_only)),
        "neither": int(np.count_nonzero(~valid_first & ~valid_second)),
        "result_pixels": int(np.count_nonzero(np.isfinite(result))),
    }


def registration_result(reference, partner, box, alignment, settings):
    if reference.record.unit and partner.record.unit and reference.record.unit != partner.record.unit:
        raise RequestError("Both input images must use the same pixel units.", "partner")
    mapping, transform = transform_for(reference, partner, box, alignment)
    original = box.extract(reference.pixels)
    aligned, weights = transformed_patch(partner, box, mapping, settings)
    common = np.isfinite(original) & np.isfinite(aligned)
    if not np.any(common):
        raise RequestError("No usable detector pixels overlap in the inspection box.", "box")
    factor, offset = photometric_normalization(original, aligned, common, settings.normalize)
    aligned = aligned * factor + offset
    result, support = blend_registered(original, aligned, settings)
    residuals = original[common] - aligned[common]
    return result, {
        "transform": transform,
        "normalization": {"scale": factor, "offset": offset},
        "region": box.payload(),
        "support": support,
        "settings": settings.payload(),
        "source_interpolation": robust_summary(weights[common], iterations=0).payload(),
        "difference_statistics": robust_summary(residuals).payload(),
        "result_statistics": robust_summary(result).payload(),
    }


def alignment_point(reference, partner, point, box, alignment, settings):
    point_in_frame(point, reference.shape)
    mapping, transform = transform_for(reference, partner, box, alignment)
    xx = np.array([[point[0]]])
    yy = np.array([[point[1]]])
    mx, my = mapping(xx, yy)
    moving_x = float(mx[0, 0])
    moving_y = float(my[0, 0])
    parent, parent_support = interpolate(reference.pixels, xx, yy)
    if settings.interpolation == "linear":
        peer, peer_support = interpolate(partner.pixels, mx, my, settings.minimum_weight)
    else:
        possible = np.isfinite(mx) & np.isfinite(my)
        peer = nearest(partner.pixels, np.where(possible, mx, -1), np.where(possible, my, -1))
        peer_support = np.isfinite(peer).astype(float)
    return {
        "reference": {
            "point": list(point),
            "value": finite_or_none(float(parent[0, 0])),
            "support": float(parent_support[0, 0]),
        },
        "partner": {
            "point": [finite_or_none(moving_x), finite_or_none(moving_y)],
            "value": finite_or_none(float(peer[0, 0])),
            "support": float(peer_support[0, 0]),
        },
        "transform": transform,
    }
