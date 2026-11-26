from collections import deque
from hashlib import sha256

import numpy as np

from .errors import RequestError
from .frames import celestial_wcs, encode_header
from .geometry import Box, finite_or_none, index_for_point, point_in_frame
from .registration import Alignment, Blend, alignment_point
from .skycoords import world_at
from .statistics import finite_values, percentile_summary, robust_summary


WALK_LIMIT = 40


def frame_manifest(workspace, starting):
    frames = []
    pending = deque([(starting, 0)])
    visited = set()
    while pending and len(frames) < WALK_LIMIT:
        current, depth = pending.popleft()
        if current.key in visited:
            continue
        visited.add(current.key)
        record = current.payload()
        links = record["lineage"].get("parents", [])
        frames.append({
            "depth": depth,
            "record": record,
            "parents": links,
        })
        for parent in links:
            pending.append((workspace.get_record(parent), depth + 1))
    return {
        "start": starting.key,
        "frames": frames,
        "remaining": bool(pending),
        "pixel_convention": "Zero-based detector pixel centers in file order.",
    }


def frame_fingerprint(frame, box):
    matrix = box.extract(frame.pixels)
    numeric = np.asarray(matrix, dtype="<f4", order="C").copy()
    good = np.isfinite(numeric)
    numeric[~good] = np.nan
    digest = sha256()
    digest.update(f"{box.x},{box.y},{box.width},{box.height}\n".encode("ascii"))
    digest.update(memoryview(numeric).cast("B"))
    heading = encode_header(frame.header).encode("ascii", errors="replace")
    values = finite_values(numeric)
    return {
        "frame_key": frame.record.key,
        "canonical_header_sha256": sha256(heading).hexdigest(),
        "selected_pixels_sha256": digest.hexdigest(),
        "source_sha256": frame.record.digest,
        "encoding": "little-endian float32 rows with canonical NaN",
        "region": box.payload(),
        "valid_pixels": len(values),
        "missing_pixels": box.area - len(values),
        "summary": robust_summary(values).payload(),
        "percentiles": percentile_summary(values, (5, 25, 50, 75, 95)),
    }


def reference_sample(frame, detector_point):
    x, y = index_for_point(detector_point)
    valid = frame.box.covers((x, y))
    level = float(frame.pixels[y, x]) if valid else None
    wcs = celestial_wcs(frame.header)
    sky = world_at(wcs, *detector_point) if valid else None
    return {
        "key": frame.record.key,
        "point": list(detector_point),
        "pixel": [x, y] if valid else None,
        "value": finite_or_none(level),
        "sky": sky,
        "available": bool(valid and level is not None and np.isfinite(level)),
        "unit": frame.record.unit,
    }


def registration_fields(parameters):
    first = parameters["alignment"]
    second = parameters["blend"]
    centers = tuple(tuple(point) for point in first.get("anchors", ()))
    partner_centers = tuple(tuple(point) for point in first.get("moving_anchors", ()))
    alignment = Alignment(first["mode"], first["shift_x"], first["shift_y"], centers, partner_centers)
    blend = Blend(second["mode"], second["interpolation"], second["minimum_weight"],
                  second["alpha"], second["normalize"], second["require_pair"])
    return alignment, blend


def parent_point_report(workspace, parent, point, lineage, index):
    if index == 0:
        return reference_sample(parent, point)
    operation = lineage.get("operation")
    if operation in ("calibrate", "master"):
        return reference_sample(parent, point)
    if operation == "register":
        region = lineage.get("region_on_parent", {})
        box = Box(region["x"], region["y"], region["width"], region["height"])
        settings, blend = registration_fields(lineage["parameters"])
        first = workspace.open(lineage["parents"][0])
        report = alignment_point(first, parent, point, box, settings, blend)["partner"]
        coordinates = report["point"]
        position = None
        if all(value is not None for value in coordinates):
            position = reference_sample(parent, coordinates)
        return {
            "key": parent.record.key,
            "point": coordinates,
            "pixel": position["pixel"] if position else None,
            "value": report["value"],
            "sky": position["sky"] if position else None,
            "available": report["value"] is not None,
            "interpolation_support": report["support"],
            "unit": parent.record.unit,
        }
    return {
        "key": parent.record.key, "point": None, "pixel": None,
        "value": None, "sky": None, "available": False, "unit": parent.record.unit,
    }


def read_parent_samples(workspace, frame, point):
    point_in_frame(point, frame.shape)
    lineage = frame.record.lineage
    x_origin, y_origin = lineage.get("origin", (0, 0))
    parent_point = (point[0] + x_origin, point[1] + y_origin)
    reports = []
    for index, parent_key in enumerate(lineage.get("parents", [])):
        parent = workspace.open(parent_key)
        reports.append(parent_point_report(workspace, parent, parent_point, lineage, index))
    return {
        "result": reference_sample(frame, point),
        "direct_parent_samples": reports,
        "operation": lineage.get("operation"),
    }


def entries_for_assets(archive, folder):
    rows = []
    for item in archive:
        path = folder / item.name
        if path.is_file():
            rows.append({
                "name": item.name,
                "frame_key": item.key,
                "title": item.title,
                "download": item.url,
                "bytes": path.stat().st_size,
                "local": f"/api/archive/{item.key}",
            })
    return rows


def asset_for_key(archive, folder, key):
    for item in archive:
        if item.key != key:
            continue
        path = folder / item.name
        if path.is_file():
            return path
    raise RequestError("The requested archive input is unavailable.", "frame", 404)
