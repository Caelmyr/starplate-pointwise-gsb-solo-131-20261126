from dataclasses import dataclass

import numpy as np

from .errors import RequestError
from .geometry import finite_or_none


@dataclass(frozen=True)
class RobustSummary:
    count: int
    retained: int
    median: float
    mean: float
    sigma: float
    minimum: float
    maximum: float

    def payload(self):
        return {
            "count": self.count,
            "retained": self.retained,
            "median": finite_or_none(self.median),
            "mean": finite_or_none(self.mean),
            "sigma": finite_or_none(self.sigma),
            "min": finite_or_none(self.minimum),
            "max": finite_or_none(self.maximum),
        }


def finite_values(values):
    values = np.asarray(values)
    return values[np.isfinite(values)].astype(np.float64, copy=False)


def median_sigma(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    median = float(np.median(values))
    return median, float(1.4826 * np.median(np.abs(values - median)))


def trimmed_indices(values, sigma=3.0, iterations=4, asymmetric=None):
    values = np.asarray(values, dtype=np.float64)
    retained = np.isfinite(values)
    for _ in range(iterations):
        selected = values[retained]
        if len(selected) < 5:
            break
        center, scatter = median_sigma(selected)
        if not scatter or not np.isfinite(scatter):
            break
        if asymmetric is None:
            low, high = sigma, sigma
        else:
            low, high = asymmetric
        revised = retained & (values >= center - low * scatter) & (values <= center + high * scatter)
        if np.array_equal(revised, retained) or np.count_nonzero(revised) < 3:
            break
        retained = revised
    return retained


def robust_summary(array, sigma=3.0, *, iterations=4, asymmetric=None):
    values = finite_values(array)
    if len(values) == 0:
        return RobustSummary(0, 0, *(float("nan"),) * 5)
    keep = trimmed_indices(values, sigma=sigma, iterations=iterations, asymmetric=asymmetric)
    selected = values[keep]
    center, scatter = median_sigma(selected)
    return RobustSummary(
        len(values), len(selected), center, float(np.mean(selected)), scatter,
        float(np.min(values)), float(np.max(values)),
    )


def values_at_points(image, x, y):
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    good = (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
    result = np.full(x.shape, np.nan, dtype=np.float64)
    result[good] = image[y[good], x[good]]
    return result


def equal_width_range(values, low=None, high=None):
    if values.size == 0:
        raise RequestError("The selected region does not contain usable pixels.", "box")
    if low is None:
        low = float(np.min(values))
    if high is None:
        high = float(np.max(values))
    if high < low:
        raise RequestError("The upper limit must be greater than the lower limit.", "high")
    if high == low:
        padding = max(1e-8, abs(low) * 1e-7)
        low -= padding
        high += padding
    return low, high


def percentile_summary(values, percentiles=(1, 5, 25, 50, 75, 95, 99)):
    if not len(values):
        return {str(percentile): None for percentile in percentiles}
    quantiles = np.percentile(values, percentiles)
    return {
        str(percentile): finite_or_none(quantile)
        for percentile, quantile in zip(percentiles, quantiles)
    }


def histogram(frame, box, low=None, high=None, bins=96):
    values = finite_values(box.extract(frame.pixels))
    requested = (low, high)
    low, high = equal_width_range(values, low, high)
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    above = int(np.count_nonzero(values > high))
    below = int(np.count_nonzero(values < low))
    count = len(values)
    cumulative = np.cumsum(counts) + below
    peak_bin = int(np.argmax(counts)) if np.any(counts) else None
    return {
        "region": box.payload(),
        "bins": [
            {
                "left": float(edges[i]), "right": float(edges[i + 1]),
                "count": int(bin_count), "cumulative": int(cumulative[i]),
            }
            for i, bin_count in enumerate(counts)
        ],
        "clip": {
            "below": below,
            "above": above,
            "requested_low": requested[0],
            "requested_high": requested[1],
        },
        "peak_bin": peak_bin,
        "finite": count,
        "missing": box.area - count,
        "percentiles": percentile_summary(values),
        "robust": robust_summary(values).payload(),
        "mean": finite_or_none(np.mean(values)),
        "sd": finite_or_none(np.std(values)),
        "unit": frame.record.unit,
    }


def compare_regions(frame, first, second):
    a = finite_values(first.extract(frame.pixels))
    b = finite_values(second.extract(frame.pixels))
    med_a, scatter_a = median_sigma(a)
    med_b, scatter_b = median_sigma(b)
    delta = med_a - med_b
    contrast = delta / scatter_b if scatter_b > 0 else float("nan")
    overlap = first.intersect(second)
    return {
        "first": {
            "box": first.payload(),
            "statistics": robust_summary(a).payload(),
        },
        "second": {
            "box": second.payload(),
            "statistics": robust_summary(b).payload(),
        },
        "median_difference": finite_or_none(delta),
        "difference_in_reference_sigma": finite_or_none(contrast),
        "first_mad_sigma": finite_or_none(scatter_a),
        "second_mad_sigma": finite_or_none(scatter_b),
        "overlap_pixels": overlap.area if overlap else 0,
    }


def bin_members(frame, box, left, right, include_upper=False, limit=300):
    if right <= left:
        raise RequestError("Select a nonempty histogram bin.", "right")
    patch = box.extract(frame.pixels)
    limit_mask = patch <= right if include_upper else patch < right
    selected = np.isfinite(patch) & (patch >= left) & limit_mask
    all_count = int(np.count_nonzero(selected))
    if all_count == 0:
        return {"count": 0, "samples": [], "limits": [left, right]}
    selected_y, selected_x = np.nonzero(selected)
    indices = np.linspace(0, all_count - 1, min(all_count, limit), dtype=np.int64)
    xs = selected_x[indices]
    ys = selected_y[indices]
    return {
        "count": all_count,
        "samples": [
            {"x": int(x + box.x), "y": int(y + box.y), "value": float(patch[y, x])}
            for x, y in zip(xs, ys)
        ],
        "limits": [left, right],
    }
