from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from .background import BackgroundSettings, background_point, estimate_background, high_frequency
from .calibration import CalibrationSettings, camera_correct, explain_correction, stack_reference_frames
from .errors import RequestError
from .filtering import FilterSettings, compute_filter, filter_point
from .frames import ImagePlane, crop_header, encode_header
from .geometry import Box, point_in_frame
from .inputs import Fields
from .limits import MAX_REGION_PIXELS
from .masking import MaskSettings, bad_pixel_mask, mask_boundaries, masked_pixels, polygon_points
from .registration import Alignment, Blend, alignment_point, registration_result
from .store import json_text


@dataclass(frozen=True)
class DerivedPlane:
    pixels: np.ndarray
    region: Box
    reports: dict[str, Any]
    label: str
    operation: str
    parents: tuple[str, ...]
    parameters: dict[str, Any]
    pixel_origin: tuple[int, int]


def selected_region(frame, fields, *, limit=MAX_REGION_PIXELS):
    box = Box.from_fields(fields.child("box"), frame.shape)
    if box.area > limit:
        raise RequestError(f"Use a region of at most {limit:,} pixels.", "box", 413)
    return box


def require_output_size(box):
    if min(box.width, box.height) < 2:
        raise RequestError("Saved outputs must have at least two rows and columns.", "box")


def optional_frame(store, parameters, key):
    frame_key = parameters.text(key, maximum=72)
    return store.open(frame_key) if frame_key else None


def background_output(frame, fields, box):
    config = BackgroundSettings.from_fields(fields)
    surface, residual, report = estimate_background(frame, box, config)
    output = fields.option("output", ("residual", "background"), "residual")
    labels = {"residual": "Background removed", "background": "Background surface"}
    if fields.raw("point") is not None:
        report["probe"] = background_point(frame, box, surface, fields.point("point"))
    return DerivedPlane(
        residual if output == "residual" else surface,
        box, report, labels[output], "background",
        (frame.record.key,), {**config.payload(), "output": output}, (box.x, box.y),
    )


def filter_output(frame, fields, box):
    width = fields.integer("window", 9, 3, 31)
    if width % 2 == 0:
        raise RequestError("Use an odd median filter window.", "window")
    if box.area > 850_000 and width > 15:
        raise RequestError("Use a smaller inspection box with this median filter window.", "box", 413)
    output = fields.option("output", ("residual", "background"), "residual")
    trend, residual, report = high_frequency(frame, box, width)
    if output == "background":
        trend = np.where(np.isfinite(box.extract(frame.pixels)), trend, np.nan)
    return DerivedPlane(
        residual if output == "residual" else trend,
        box, report, "Local detail" if output == "residual" else "Median surface",
        "filter", (frame.record.key,), {"window": width, "output": output}, (box.x, box.y),
    )


def multiscale_output(frame, fields, box):
    settings = FilterSettings.from_fields(fields)
    plane, report, direction = compute_filter(frame, box, settings)
    if fields.raw("point") is not None:
        report["probe"] = filter_point(frame, box, fields.point("point"), plane, direction)
    display_names = {
        "gaussian": "Gaussian smoothing", "bandpass": "Band-pass structure",
        "laplacian": "Curvature response", "gradient": "Gradient magnitude",
    }
    return DerivedPlane(
        plane, box, report, display_names[settings.operation], "multiscale",
        (frame.record.key,), settings.payload(), (box.x, box.y),
    )


def mask_output(frame, fields, box):
    settings = MaskSettings.from_fields(fields)
    vertices = polygon_points(fields, box) if settings.mode == "polygon" else None
    mask, report = bad_pixel_mask(frame, box, settings, vertices)
    report["edges"] = mask_boundaries(mask, box)
    report["vertices"] = vertices
    if report["new_invalid"] == 0:
        raise RequestError("Mask does not cover any additional usable original pixels.", "mode")
    parameters = {
        "mode": settings.mode, "lower": settings.lower, "upper": settings.upper,
        "dilate": settings.dilate, "vertices": vertices,
    }
    return DerivedPlane(
        masked_pixels(frame, box, mask), frame.box, report, "Masked science frame",
        "mask", (frame.record.key,), {**parameters, "selection": box.payload()}, (0, 0),
    )


def camera_output(store, frame, fields, box):
    dark = optional_frame(store, fields, "dark")
    flat = optional_frame(store, fields, "flat")
    settings = CalibrationSettings.from_fields(fields)
    result, report = camera_correct(frame, dark, flat, box, settings)
    if fields.raw("point") is not None:
        normalization = report["flat_reference"]["normalization"] if flat is not None else 1
        report["probe"] = explain_correction(frame, dark, flat, fields.point("point"), normalization, settings)
    parent_keys = (frame.record.key, *(item.record.key for item in (dark, flat) if item is not None))
    return DerivedPlane(
        result, frame.box, report, "Camera calibrated", "calibrate",
        parent_keys, {
            **settings.payload(), "dark": report["dark"], "flat": report["flat"], "normalization_box": box.payload(),
        }, (0, 0),
    )


def registration_output(store, frame, fields, box):
    partner_key = fields.text("partner", minimum=1, maximum=72)
    partner = store.open(partner_key)
    alignment = Alignment.from_fields(fields)
    blend = Blend.from_fields(fields)
    result, report = registration_result(frame, partner, box, alignment, blend)
    if fields.raw("point") is not None:
        point = fields.point("point")
        report["probe"] = alignment_point(frame, partner, point, box, alignment, blend)
    descriptions = {
        "mean": "Aligned mean", "difference": "Aligned difference",
        "maximum": "Aligned maximum", "reference": "Aligned companion",
    }
    return DerivedPlane(
        result, box, report, descriptions[blend.mode], "register",
        (frame.record.key, partner_key),
        {"alignment": alignment.payload(), "blend": blend.payload(), "partner": partner_key},
        (box.x, box.y),
    )


def master_output(store, frame, fields, box):
    keys = fields.list("references", minimum=1, maximum=7)
    frames = [frame, *(store.open(key) for key in keys)]
    mode = fields.option("mode", ("median", "normalized_flat"), "median")
    result, report = stack_reference_frames(frames, box, mode)
    keys = tuple(item.record.key for item in frames)
    return DerivedPlane(
        result, frame.box, report, "Master flat field" if mode == "normalized_flat" else "Master dark",
        "master", keys, {"mode": mode, "normalization_box": box.payload()}, (0, 0),
    )


def crop_output(frame, fields, box):
    data = box.extract(frame.pixels)
    frame_box = frame.box
    require_output_size(box)
    coverage = int(np.count_nonzero(np.isfinite(data)))
    report = {
        "selected": box.payload(), "parent_dimensions": frame_box.payload(),
        "finite": coverage, "missing": box.area - coverage,
        "origin_in_parent": [box.x, box.y],
    }
    return DerivedPlane(data, box, report, "Image cutout", "crop", (frame.record.key,), {}, (box.x, box.y))


def make_derived(store, frame, operation, fields, box):
    require_output_size(box)
    direct = {
        "sky": background_output, "detail": filter_output, "mask": mask_output,
        "crop": crop_output, "multiscale": multiscale_output,
    }
    with_store = {"calibrate": camera_output, "register": registration_output, "master": master_output}
    if operation in direct:
        return direct[operation](frame, fields, box)
    if operation in with_store:
        return with_store[operation](store, frame, fields, box)
    raise RequestError("Choose a supported derived image operation.", "operation")


def derived_digest(pixels, header, lineage):
    digest = sha256()
    digest.update(encode_header(header).encode("ascii", errors="replace"))
    digest.update(json_text(lineage).encode("utf-8"))
    contiguous = np.ascontiguousarray(pixels, dtype=np.float32)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def retain_derived(store, frame, output, title):
    require_output_size(output.region)
    header = crop_header(frame.header, output.region)
    if output.operation == "master" and output.parameters["mode"] == "normalized_flat":
        header["BUNIT"] = "relative"
    header["ORIGHDU"] = frame.record.hdu
    header["OPERATOR"] = "StarPlate"
    header["DERIVED"] = output.operation
    lineage = {
        "parents": list(output.parents), "operation": output.operation,
        "region_on_parent": output.region.payload(), "origin": list(output.pixel_origin),
        "parameters": output.parameters,
    }
    plane = ImagePlane(output.pixels, header, "fits", 0, str(header.get("BUNIT", frame.record.unit)), ())
    digest = derived_digest(output.pixels, header, lineage)
    name = f"{output.operation}-{frame.record.key[:42]}.fits"
    return store.store(plane, title=title or output.label, original_name=name,
                       origin="derived", digest=digest, lineage=lineage)


def lineage_pixel(store, frame, params):
    x, y = point_in_frame(params.point("point"), frame.shape)
    parents = frame.record.lineage.get("parents", [])
    parent_origin = frame.record.lineage.get("origin", [0, 0])
    result = [{"frame_key": frame.record.key, "point": [x, y]}]
    followed = {frame.record.key}
    current = frame.record
    while parents and len(result) < 12:
        parent_key = parents[0]
        if parent_key in followed:
            break
        followed.add(parent_key)
        x += parent_origin[0]
        y += parent_origin[1]
        parent = store.get_record(parent_key)
        box = Box.full((parent.height, parent.width))
        if not box.covers((x, y)):
            break
        result.append({"frame_key": parent_key, "point": [x, y], "operation": current.lineage.get("operation")})
        current = parent
        parents = current.lineage.get("parents", [])
        parent_origin = current.lineage.get("origin", [0, 0])
    return {"trace": result, "additional_parents": frame.record.lineage.get("parents", [])[1:]}
