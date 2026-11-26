from csv import DictWriter
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json
import re

from astropy.io import fits
from flask import Blueprint, current_app, request, send_file
import numpy as np

from .errors import RequestError
from .frames import byte_stream_fits, crop_header
from .geometry import Box
from .inputs import Fields
from .rendering import Stretch, encoded_png, make_preview_sheet
from .session import selected_region
from .store import json_text


exports = Blueprint("exports", __name__, url_prefix="/api")


def filename(title):
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", title)
    return safe[:68].strip("-.") or "starplate"


def cutout_hdus(frame, box, include_validity):
    pixels = box.extract(frame.pixels)
    header = crop_header(frame.header, box)
    header["ORIGHDU"] = frame.record.hdu
    header["SRCNAME"] = frame.record.original_name
    header["SRCHASH"] = frame.record.digest
    extra = []
    if include_validity:
        validity = fits.ImageHDU(np.isfinite(pixels).astype(np.uint8), name="VALID")
        validity.header["SRCX"] = box.x, "Original detector x of first mask pixel"
        validity.header["SRCY"] = box.y, "Original detector y of first mask pixel"
        validity.header["HISTORY"] = "1 is finite in primary image; 0 is unavailable"
        extra.append(validity)
    return byte_stream_fits(pixels, header, extra)


def table_for_measurement(measurement):
    result = measurement.result
    operation = measurement.operation
    columns = {}
    chosen = {
        "profile": "samples",
        "stars": "sources",
        "histogram": "bins",
        "histogram-bin": "samples",
        "aperture-curve": "measurements",
        "background": "cells",
        "quality": "cells",
        "source-shape-map": "stars",
        "detector-noise": "cells",
    }
    if operation in chosen:
        rows = result.get(chosen[operation], [])
        if not rows:
            raise RequestError("The measurement does not contain any table rows.", "measurement")
        for row in rows:
            for key, value in row.items():
                if not isinstance(value, (dict, list)):
                    columns.setdefault(key, None)
        normalized = [{key: row.get(key) for key in columns} for row in rows]
    else:
        normalized = [flatten_row(result)]
        columns = dict.fromkeys(normalized[0], None)
    return list(columns), normalized


def flatten_row(row, prefix=""):
    result = {}
    for key, value in row.items():
        target = prefix + key
        if isinstance(value, list):
            if len(value) <= 8 and all(not isinstance(member, (list, dict)) for member in value):
                result[target] = json_text(value)
        elif isinstance(value, dict):
            result.update(flatten_row(value, target + "_"))
        else:
            result[target] = value
    return result


def safe_csv_cell(value):
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def measurement_csv(measurement):
    columns, results = table_for_measurement(measurement)
    out = StringIO(newline="")
    writer = DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        writer.writerow({key: safe_csv_cell(value) for key, value in row.items()})
    return BytesIO(out.getvalue().encode("utf-8"))


def fits_column(key, rows):
    values = [row.get(key) for row in rows]
    nonempty = [value for value in values if value is not None]
    if nonempty and all(type(value) is bool for value in nonempty):
        format_spec = "L"
        array = np.asarray([bool(value) for value in values])
    elif nonempty and all(type(value) in (float, int, bool) for value in nonempty):
        format_spec = "D"
        array = np.asarray([float(value) if value is not None else np.nan for value in values], dtype=float)
    else:
        ascii_values = [str(value or "").encode("ascii", errors="replace")[:180] for value in values]
        width = max(1, *(len(value) for value in ascii_values))
        format_spec = f"{width}A"
        array = np.asarray(ascii_values, dtype=f"S{width}")
    return fits.Column(name=key[:66], format=format_spec, array=array)


def measurement_fits(frame, measurement):
    schema, rows = table_for_measurement(measurement)
    if not schema:
        raise RequestError("The measurement contains no exportable table fields.", "measurement")
    columns = [fits_column(key, rows) for key in schema]
    table = fits.BinTableHDU.from_columns(columns, name="RESULTS")
    table.header["FRAMEKEY"] = frame.record.key
    table.header["MEASOP"] = measurement.operation
    table.header["PARENT"] = frame.record.digest
    table.header["BUNIT"] = frame.record.unit
    output = BytesIO()
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(output, checksum=True)
    output.seek(0)
    return output


def readable_manifest(frame, region, preview, notes, measurements):
    return {
        "format": "StarPlate evidence bundle",
        "schema": 1,
        "frame": frame.record.payload(),
        "original_region": region.payload(),
        "preview_display": preview,
        "coordinate_convention": "zero-based x/y centers in original file order",
        "files": {
            "pixels.fits": "Scientific primary image, VALID extension, original-coordinate metadata",
            "preview.png": "8-bit display preview with notes and display range",
            "notes.json": "Workspace notes that fall within the selected region",
            "measurements.json": "Selected saved measurement data with original parameters",
        },
        "notes": len(notes),
        "measurements": len(measurements),
    }


@exports.get("/frames/<key>/export/<kind>")
def export_image(key, kind):
    fields = Fields(request.args)
    workspace = current_app.config["FRAME_STORE"]
    frame = workspace.open(key)
    box = Box.from_fields(fields, frame.shape)
    stem = filename(frame.record.title)
    if kind == "fits":
        include_mask = fields.flag("validity", True)
        stream = cutout_hdus(frame, box, include_mask)
        return send_file(stream, as_attachment=True, download_name=f"{stem}.fits",
                         mimetype="application/fits")
    stretch = Stretch.from_fields(fields)
    notes = [note for note in workspace.list_notes(key) if box.covers((note.x, note.y))]
    if kind == "png":
        sheet = make_preview_sheet(frame, box, stretch, notes, fields.integer("size", 1450, 300, 2000))
        return send_file(encoded_png(sheet), as_attachment=True, download_name=f"{stem}.png",
                         mimetype="image/png")
    if kind == "bundle":
        return frame_bundle(workspace, frame, box, stretch, notes, fields)
    raise RequestError("Choose FITS, PNG or bundle export.", "format", 404)


def frame_bundle(workspace, frame, box, stretch, notes, fields):
    selected = fields.text("measurements", maximum=1400)
    keys = list(dict.fromkeys(key.strip() for key in selected.split(",") if key.strip()))
    if len(keys) > 12:
        raise RequestError("Select up to 12 saved measurements for one bundle.", "measurements")
    measurements = [workspace.load_measurement(frame.record.key, key) for key in keys]
    if len(json_text([item.result for item in measurements])) > 9_000_000:
        raise RequestError("Selected measurements exceed the archive size limit.", "measurements", 413)
    sheet = make_preview_sheet(frame, box, stretch, notes)
    manifest = readable_manifest(frame, box, stretch.payload(), notes, measurements)
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=2) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("pixels.fits", cutout_hdus(frame, box, include_validity=True).getbuffer())
        archive.writestr("preview.png", encoded_png(sheet).getbuffer())
        archive.writestr("notes.json", json_text([note.payload() for note in notes]))
        archive.writestr("measurements.json", json_text([measure.payload() for measure in measurements]))
    stream.seek(0)
    return send_file(stream, as_attachment=True, download_name=f"{filename(frame.record.title)}-evidence.zip",
                     mimetype="application/zip")


@exports.get("/frames/<key>/saved/<measurement_key>/export/<kind>")
def export_measurement(key, measurement_key, kind):
    workspace = current_app.config["FRAME_STORE"]
    item = workspace.load_measurement(key, measurement_key)
    stem = filename(item.title)
    if kind == "csv":
        stream = measurement_csv(item)
        mimetype = "text/csv"
    elif kind == "fits":
        stream = measurement_fits(workspace.open(key), item)
        mimetype = "application/fits"
    elif kind == "json":
        stream = BytesIO(json.dumps(item.payload(), ensure_ascii=False, indent=2).encode("utf-8"))
        mimetype = "application/json"
    else:
        raise RequestError("Choose CSV, FITS or JSON.", "format", 404)
    return send_file(stream, as_attachment=True, download_name=f"{stem}.{kind}", mimetype=mimetype)
