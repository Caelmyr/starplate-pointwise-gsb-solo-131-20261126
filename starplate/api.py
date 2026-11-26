from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.exceptions import BadRequest

from .demo import ARCHIVE
from .errors import RequestError
from .framecheck import overlap_polyline, route_compare
from .frames import original_file_name, read_plane, safe_header_rows, source_digest
from .geometry import Box
from .inputs import Fields
from .limits import MAX_FILE_BYTES
from .measurements import run_measurement
from .provenance import asset_for_key, entries_for_assets, frame_fingerprint, frame_manifest, read_parent_samples
from .rendering import Stretch, displayed_image, encoded_png
from .session import lineage_pixel, make_derived, retain_derived, selected_region
from .skycoords import celestial_grid, sky_summary


api = Blueprint("science", __name__, url_prefix="/api")


def store():
    return current_app.config["FRAME_STORE"]


def query_fields():
    return Fields(request.args)


def body_fields():
    if not request.is_json:
        raise RequestError("Send this action as JSON.", "", 415)
    try:
        return Fields(request.get_json())
    except BadRequest as error:
        raise RequestError("Request is not valid JSON.") from error


@api.get("/health")
def health():
    return {"healthy": True, "product": "StarPlate", "schema": 1}


@api.get("/frames")
def frames():
    return {"frames": [record.payload() for record in store().list_frames()]}


@api.post("/inspect-import")
def inspect_import():
    uploaded, raw = upload_bytes()
    guessed = read_plane(uploaded.filename, raw)
    return {
        "original_name": original_file_name(uploaded.filename),
        "type": guessed.file_type,
        "candidate_planes": guessed.candidates,
        "selected": guessed.hdu,
        "width": guessed.pixels.shape[1],
        "height": guessed.pixels.shape[0],
    }


def upload_bytes():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        raise RequestError("Choose an image to upload.", "image")
    raw = uploaded.stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise RequestError("Image transfer exceeds the file size limit.", "image", 413)
    return uploaded, raw


@api.post("/frames")
def import_frame():
    form = Fields(request.form)
    uploaded, raw = upload_bytes()
    hdu = form.integer("hdu", minimum=0, maximum=63) if form.raw("hdu") not in (None, "") else None
    name = original_file_name(uploaded.filename)
    requested = form.text("title", maximum=100)
    digest = source_digest(raw)
    if hdu is not None:
        existing = store().existing(digest, hdu, "upload")
        if existing:
            return jsonify(frame=existing.payload(), created=False)
    plane = read_plane(name, raw, hdu)
    title = requested or Path(name).stem[:100]
    record, created = store().store(
        plane, title=title, original_name=name, origin="upload", digest=digest,
    )
    return jsonify(frame=record.payload(), created=created, available_planes=plane.candidates), 201 if created else 200


@api.get("/frames/<key>")
def frame_metadata(key):
    frame = store().open(key)
    return {
        "frame": frame.record.payload(),
        "header": safe_header_rows(frame.header),
        "sky": sky_summary(frame),
    }


@api.patch("/frames/<key>")
def rename_frame(key):
    fields = body_fields()
    title = fields.text("title", minimum=1, maximum=100)
    record = store().rename(key, title)
    return {"frame": record.payload()}


@api.delete("/frames/<key>")
def remove_frame(key):
    store().remove(key)
    return "", 204


@api.get("/frames/<key>/image")
def image_preview(key):
    frame = store().open(key)
    fields = query_fields()
    display = Stretch.from_fields(fields)
    box = Box.from_fields(fields, frame.shape)
    fixed = fields.flag("lock_range", False)
    statistics_box = frame.box if fixed else box
    image, bounds = displayed_image(
        frame, box, statistics_box, display, fields.integer("size", 1536, 64, 2400)
    )
    response = send_file(encoded_png(image), mimetype="image/png", max_age=300)
    response.headers["X-StarPlate-Low"] = f"{bounds[0]:.9g}"
    response.headers["X-StarPlate-High"] = f"{bounds[1]:.9g}"
    return response


@api.get("/frames/<key>/grid")
def grid(key):
    frame = store().open(key)
    count = query_fields().integer("count", 5, 2, 12)
    return celestial_grid(frame, count)


@api.post("/frames/<key>/lineage")
def lineage(key):
    frame = store().open(key)
    return lineage_pixel(store(), frame, body_fields())


@api.post("/frames/<key>/measure/<operation>")
def measure(key, operation):
    frame = store().open(key)
    return run_measurement(frame, operation, body_fields())


@api.post("/frames/<key>/derive/<operation>")
def derive(key, operation):
    workspace = store()
    frame = workspace.open(key)
    fields = body_fields()
    domain = selected_region(frame, fields)
    selected = make_derived(workspace, frame, operation, fields, domain)
    title = fields.text("title", maximum=100)
    result = {"operation": operation, "frame_key": key, "result": selected.reports}
    if fields.flag("retain", False):
        record, created = retain_derived(workspace, frame, selected, title)
        return jsonify(**result, frame=record.payload(), created=created), 201 if created else 200
    return result


@api.get("/frames/<key>/notes")
def list_notes(key):
    return {"notes": [note.payload() for note in store().list_notes(key)]}


@api.post("/frames/<key>/notes")
def add_note(key):
    fields = body_fields()
    title = fields.text("title", minimum=1, maximum=80)
    text = fields.text("text", maximum=600)
    note = store().add_note(key, title, text, fields.point("point"))
    return jsonify(note=note.payload()), 201


@api.delete("/frames/<key>/notes/<note_key>")
def remove_note(key, note_key):
    store().remove_note(key, note_key)
    return "", 204


@api.get("/frames/<key>/saved")
def list_measurements(key):
    return {"measurements": store().list_measurements(key)}


@api.post("/frames/<key>/saved")
def save_measurement(key):
    fields = body_fields()
    workspace = store()
    frame = workspace.open(key)
    operation = fields.text("operation", minimum=1, maximum=30)
    title = fields.text("title", minimum=1, maximum=100)
    parameters = fields.child("parameters")
    result = run_measurement(frame, operation, parameters)
    record = workspace.save_measurement(key, operation, title, dict(parameters.values), result["result"])
    return jsonify(measurement=record.payload()), 201


@api.get("/frames/<key>/saved/<measurement_key>")
def load_measurement(key, measurement_key):
    return {"measurement": store().load_measurement(key, measurement_key).payload()}


@api.delete("/frames/<key>/saved/<measurement_key>")
def remove_measurement(key, measurement_key):
    store().remove_measurement(key, measurement_key)
    return "", 204


@api.post("/frames/<key>/pair")
def pair_statistics(key):
    workspace = store()
    frame = workspace.open(key)
    fields = body_fields()
    return {"result": route_compare(workspace, frame, fields)}


@api.post("/frames/<key>/pair-footprint")
def pair_footprint(key):
    workspace = store()
    frame = workspace.open(key)
    fields = body_fields()
    partner = workspace.open(fields.text("partner", minimum=1, maximum=72))
    return {"result": overlap_polyline(frame, partner, fields)}


@api.get("/frames/<key>/history")
def history(key):
    workspace = store()
    frame = workspace.get_record(key)
    return frame_manifest(workspace, frame)


@api.post("/frames/<key>/source-samples")
def source_samples(key):
    workspace = store()
    frame = workspace.open(key)
    return read_parent_samples(workspace, frame, body_fields().point("point"))


@api.post("/frames/<key>/fingerprint")
def fingerprint(key):
    frame = store().open(key)
    fields = body_fields()
    domain = selected_region(frame, fields, limit=8_500_000)
    return {"result": frame_fingerprint(frame, domain)}


@api.get("/archive")
def archive_inventory():
    folder = Path(current_app.config["STARPLATE_ARCHIVE"])
    return {"assets": entries_for_assets(ARCHIVE, folder)}


@api.get("/archive/<key>")
def archive_original(key):
    folder = Path(current_app.config["STARPLATE_ARCHIVE"])
    path = asset_for_key(ARCHIVE, folder, key)
    return send_file(path, mimetype="application/x-bzip2", as_attachment=True, download_name=path.name)
