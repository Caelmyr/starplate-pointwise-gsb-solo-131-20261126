from dataclasses import dataclass
from math import isfinite

from flask import Blueprint, current_app, jsonify, request, send_file
import numpy as np
from PIL import Image

from .errors import RequestError
from .geometry import Box, downsample_shape, finite_or_none, index_for_point, point_in_frame
from .inputs import Fields
from .registration import Alignment, Blend, transformed_patch, wcs_mapping
from .rendering import encoded_png
from .session import selected_region
from .statistics import finite_values, percentile_summary


color_api = Blueprint("color", __name__, url_prefix="/api")
CHANNEL_NAMES = ("red", "green", "blue")


@dataclass(frozen=True)
class Channel:
    key: str
    low_pct: float
    high_pct: float
    weight: float

    @classmethod
    def read(cls, fields, default_key):
        low = fields.number("low_pct", 5, 0, 99)
        high = fields.number("high_pct", 99.7, 1, 100)
        if high <= low:
            raise RequestError("Channel high percentile must exceed low percentile.", fields.path("high_pct"))
        return cls(
            fields.text("frame", default_key, 1, 72), low, high, fields.number("weight", 1, .01, 10)
        )

    def range(self, values):
        levels = finite_values(values)
        if len(levels) < 4:
            raise RequestError("Color channel has fewer than four usable pixels.", self.key)
        low, high = np.percentile(levels, (self.low_pct, self.high_pct))
        if high <= low:
            low = np.min(levels)
            high = np.max(levels)
        if high <= low:
            high = low + max(1e-9, abs(low) * 1e-7)
        return float(low), float(high)


@dataclass(frozen=True)
class ColorSetup:
    mode: str
    align: str
    stretch: float
    saturation: float
    luminance: float
    channels: tuple[Channel, ...]

    @classmethod
    def read(cls, fields, first_key):
        modes = ("linked", "per-channel")
        return cls(
            fields.option("mode", modes, "linked"),
            fields.option("alignment", ("detector", "wcs"), "detector"),
            fields.number("stretch", 5, .1, 90),
            fields.number("saturation", 1, 0, 2),
            fields.number("luminance", 1, .15, 4),
            tuple(Channel.read(fields.child(name), first_key) for name in CHANNEL_NAMES),
        )


def aligned_channel(reference, peer, box, alignment):
    if reference.record.key == peer.record.key:
        samples = box.extract(reference.pixels)
        return samples, np.isfinite(samples)
    if alignment == "detector":
        if peer.shape != reference.shape:
            raise RequestError("Detector color layers must have the same width and height.", "alignment")
        samples = box.extract(peer.pixels)
        return samples, np.isfinite(samples)
    geometry = wcs_mapping(reference, peer)
    blend = Blend("reference", "linear", .999, 1, "none", True)
    samples, support = transformed_patch(peer, box, geometry, blend)
    return samples, np.isfinite(samples) & (support >= .999)


def normalized_channel(src, mask, low, high, weight):
    plane = np.where(mask, (src.astype(float) - low) / (high - low) * weight, 0)
    return np.maximum(plane, 0)


def color_transfer(pixels, setup):
    if setup.mode == "per-channel":
        transformed = np.arcsinh(pixels * setup.stretch) / np.arcsinh(setup.stretch)
    else:
        intensity = pixels @ np.array([.2126, .7152, .0722])
        scaling = np.zeros(intensity.shape)
        response = np.arcsinh(intensity * setup.stretch) / np.arcsinh(setup.stretch)
        np.divide(response, intensity, where=intensity > 0, out=scaling)
        transformed = pixels * scaling[:, :, None]
    transformed *= setup.luminance
    grayscale = (transformed @ np.array([.2126, .7152, .0722]))[:, :, None]
    saturated = grayscale + setup.saturation * (transformed - grayscale)
    headroom = np.clip(np.max(saturated, axis=-1), 1, None)
    return np.clip(saturated / headroom[:, :, None], 0, 1)


def generate_rgb(workspace, reference, box, setup, size):
    components = []
    support = []
    properties = []
    for name, band in zip(CHANNEL_NAMES, setup.channels):
        peer = workspace.open(band.key)
        samples, mask = aligned_channel(reference, peer, box, setup.align)
        low, high = band.range(samples[mask])
        component = normalized_channel(samples, mask, low, high, band.weight)
        components.append(component)
        support.append(mask)
        properties.append({
            "channel": name,
            "frame_key": peer.record.key,
            "coverage": int(np.count_nonzero(mask)),
            "range": [low, high],
            "percentiles": percentile_summary(finite_values(samples[mask]), (5, 50, 95)),
            "weight": band.weight,
        })
    shared = np.logical_and.reduce(support)
    if not shared.any():
        raise RequestError("Selected channels have no shared original support in this region.", "box")
    stack = np.stack(components, axis=2)
    colors = np.rint(color_transfer(stack, setup) * 255).astype(np.uint8)
    colors[~shared] = (18, 28, 42)
    image = Image.fromarray(colors)
    h, w = downsample_shape(shared.shape, size)
    if image.size != (w, h):
        image = image.resize((w, h), Image.Resampling.BOX)
    return image, {
        "region": box.payload(),
        "channels": properties,
        "common": int(np.count_nonzero(shared)),
        "missing": int(np.count_nonzero(~shared)),
        "mode": setup.mode,
        "alignment": setup.align,
        "image_size": [w, h],
    }


def trace_channels(workspace, reference, setup, point):
    x, y = point_in_frame(point, reference.shape)
    ix, iy = index_for_point((x, y))
    tiny = Box(ix, iy, 1, 1)
    records = []
    for name, channel in zip(CHANNEL_NAMES, setup.channels):
        peer = workspace.open(channel.key)
        plane, region = aligned_channel(reference, peer, tiny, setup.align)
        value = float(plane[0, 0])
        records.append({
            "channel": name,
            "source": peer.record.key,
            "value": finite_or_none(value),
            "available": bool(region[0, 0]),
            "unit": peer.record.unit,
        })
    return {"reference_point": [x, y], "original_pixel": [ix, iy], "channels": records}


@color_api.post("/frames/<key>/color")
def compose_color(key):
    if not request.is_json:
        raise RequestError("Send channel selections as JSON.", "channels", 415)
    fields = Fields(request.get_json())
    workspace = current_app.config["FRAME_STORE"]
    reference = workspace.open(key)
    box = selected_region(reference, fields, limit=1_800_000)
    setup = ColorSetup.read(fields, reference.record.key)
    size = fields.integer("size", 1200, 64, 1800)
    format_name = fields.option("format", ("preview", "report", "download"), "preview")
    if format_name == "report" and fields.raw("point") is not None:
        return {"result": trace_channels(workspace, reference, setup, fields.point("point"))}
    image, result = generate_rgb(workspace, reference, box, setup, size)
    if format_name == "report":
        return {"result": result}
    if format_name == "download":
        return send_file(encoded_png(image), mimetype="image/png", as_attachment=True, download_name="channel-composite.png")
    response = send_file(encoded_png(image), mimetype="image/png")
    response.headers["X-Common-Pixels"] = str(result["common"])
    return response
