from dataclasses import dataclass
from io import BytesIO
from math import isfinite

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .errors import RequestError
from .geometry import Box, downsample_shape, finite_or_none
from .statistics import finite_values, percentile_summary


PALETTES = {
    "gray": ((0, 0, 0), (255, 255, 255)),
    "night": ((4, 7, 17), (38, 43, 100), (76, 124, 201),
              (132, 221, 220), (248, 249, 239)),
    "amber": ((11, 10, 17), (69, 23, 48), (153, 63, 38),
              (230, 162, 67), (255, 242, 201)),
    "balance": ((22, 60, 137), (115, 169, 216), (238, 237, 224),
                (212, 112, 69), (115, 13, 40)),
}
MISSING = np.array((30, 39, 52), dtype=np.uint8)


@dataclass(frozen=True)
class Stretch:
    mode: str
    low_pct: float
    high_pct: float
    low: float | None
    high: float | None
    strength: float
    palette: str
    invert: bool

    @classmethod
    def from_fields(cls, fields):
        low_pct = fields.number("low_pct", 1, 0, 100)
        high_pct = fields.number("high_pct", 99.8, 0, 100)
        if high_pct <= low_pct:
            raise RequestError("High percentile must be above low percentile.", "high_pct")
        low = fields.number("low") if fields.raw("low") not in (None, "") else None
        high = fields.number("high") if fields.raw("high") not in (None, "") else None
        if low is not None and high is not None and high <= low:
            raise RequestError("High level must exceed low level.", "high")
        return cls(
            fields.option("mode", ("linear", "sqrt", "log", "asinh", "gamma"), "asinh"),
            low_pct,
            high_pct,
            low,
            high,
            fields.number("strength", 6, .1, 100),
            fields.option("palette", tuple(PALETTES), "gray"),
            fields.flag("invert", False),
        )

    def payload(self):
        return {
            "mode": self.mode, "low_pct": self.low_pct, "high_pct": self.high_pct,
            "low": self.low, "high": self.high, "strength": self.strength, "palette": self.palette,
            "invert": self.invert,
        }


def display_range(pixels, stretch):
    values = finite_values(pixels)
    if not len(values):
        raise RequestError("Display range needs at least one usable pixel.", "box")
    if stretch.low is None:
        low = float(np.percentile(values, stretch.low_pct))
    else:
        low = stretch.low
    if stretch.high is None:
        high = float(np.percentile(values, stretch.high_pct))
    else:
        high = stretch.high
    if high <= low:
        if stretch.low is not None or stretch.high is not None:
            raise RequestError("Display bounds leave no positive range in this box.", "high")
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum != maximum:
            low, high = minimum, maximum
        else:
            step = max(abs(minimum) * 1e-6, 1e-6)
            low, high = minimum - step, maximum + step
    return low, high, values


def apply_curve(normalized, mode, strength):
    if mode == "sqrt":
        return np.sqrt(normalized)
    if mode == "log":
        return np.log1p(normalized * strength) / np.log1p(strength)
    if mode == "asinh":
        return np.arcsinh(normalized * strength) / np.arcsinh(strength)
    if mode == "gamma":
        return normalized ** (1 / strength)
    return normalized


def inverse_curve(strengths, mode, strength):
    if mode == "sqrt":
        return np.square(strengths)
    if mode == "log":
        return np.expm1(strengths * np.log1p(strength)) / strength
    if mode == "asinh":
        return np.sinh(strengths * np.arcsinh(strength)) / strength
    if mode == "gamma":
        return strengths ** strength
    return strengths


def palette_lut(palette):
    colors = np.asarray(PALETTES[palette])
    points = np.linspace(0, 255, len(colors))
    sample = np.arange(256)
    result = np.stack([np.interp(sample, points, color) for color in colors.T], axis=-1)
    return np.rint(result).astype(np.uint8)


def display_rgb(pixels, stretch, levels):
    valid = np.isfinite(pixels)
    low, high = levels
    normalized = np.clip((np.where(valid, pixels, low) - low) / (high - low), 0, 1)
    encoded = apply_curve(normalized, stretch.mode, stretch.strength)
    if stretch.invert:
        encoded = 1 - encoded
    quantized = np.rint(encoded * 255).astype(np.uint8)
    rgb = palette_lut(stretch.palette)[quantized]
    rgb[~valid] = MISSING
    return rgb


def displayed_image(frame, domain, range_domain, stretch, size):
    pixels = domain.extract(frame.pixels)
    low, high, _ = display_range(range_domain.extract(frame.pixels), stretch)
    rgb = display_rgb(pixels, stretch, (low, high))
    image = Image.fromarray(rgb)
    h, w = downsample_shape(pixels.shape, size)
    if image.size != (w, h):
        image = image.resize((w, h), resample=Image.Resampling.BOX)
    return image, (low, high)


def encoded_png(image):
    data = BytesIO()
    image.save(data, format="PNG", optimize=False, compress_level=4)
    data.seek(0)
    return data


def stretch_summary(frame, box, stretch):
    pixels = box.extract(frame.pixels)
    low, high, values = display_range(pixels, stretch)
    levels = np.linspace(0, 1, 9)
    inverse = inverse_curve(levels, stretch.mode, stretch.strength)
    if stretch.invert:
        inverse = inverse[::-1]
    targets = inverse * (high - low) + low
    lut = palette_lut(stretch.palette)
    return {
        "region": box.payload(),
        "low": low,
        "high": high,
        "clipped_low": int(np.count_nonzero(values <= low)),
        "clipped_high": int(np.count_nonzero(values >= high)),
        "missing": int(pixels.size - len(values)),
        "finite": len(values),
        "unit": frame.record.unit,
        "settings": stretch.payload(),
        "anchors": [
            {"display": float(level), "original": finite_or_none(value),
             "rgb": [int(channel) for channel in lut[round(level * 255)]]}
            for level, value in zip(levels, targets)
        ],
        "percentiles": percentile_summary(values),
    }


def note_markers(image, box, notes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    sx = image.width / box.width
    sy = image.height / box.height
    for note in notes:
        if not box.covers((note.x, note.y)):
            continue
        x = (note.x + .5 - box.x) * sx
        y = (note.y + .5 - box.y) * sy
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="#7bfbdd", width=2)
        measured = draw.textbbox((x + 8, y - 7), note.title[:36], font=font)
        draw.rectangle((measured[0] - 2, measured[1] - 2, measured[2] + 2, measured[3] + 1), fill="#0a121e")
        draw.text((x + 8, y - 7), note.title[:36], font=font, fill="#d7fff6")
    return image


def make_preview_sheet(frame, box, stretch, notes, size=1450):
    panel, edges = displayed_image(frame, box, box, stretch, size)
    panel = note_markers(panel, box, notes)
    title_area = 62
    footer = 36
    canvas = Image.new("RGB", (max(panel.width, 480), panel.height + title_area + footer), "#111925")
    canvas.paste(panel, (0, title_area))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=12)
    draw.text((14, 12), frame.record.title, font=font, fill="#e8f1fd")
    left = f"Original x {box.x} · y {box.y} · {box.width} × {box.height} px"
    draw.text((14, 36), left, font=small, fill="#aebed0")
    right = f"{stretch.mode} range {edges[0]:.6g} to {edges[1]:.6g} {frame.record.unit}"
    draw.text((14, panel.height + title_area + 10), right, font=small, fill="#aebed0")
    return canvas
