from math import isfinite
import os
from pathlib import Path

from flask import Flask, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
import numpy as np
from werkzeug.exceptions import HTTPException

from .api import api
from .color import color_api
from .demo import initialize_archive
from .errors import RequestError
from .exporting import exports
from .limits import MAX_FILE_BYTES
from .store import FrameStore


class ScientificJSON(DefaultJSONProvider):
    @staticmethod
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.floating):
            return float(value) if isfinite(value) else None
        return DefaultJSONProvider.default(value)


class StarPlate(Flask):
    json_provider_class = ScientificJSON


def create_app(config=None):
    project = Path(__file__).resolve().parent.parent
    app = StarPlate(__name__, static_folder=str(project / "static"), static_url_path="/static")
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_FILE_BYTES + 64 * 1024,
        MAX_FORM_MEMORY_SIZE=256 * 1024,
        MAX_FORM_PARTS=12,
        SEND_FILE_MAX_AGE_DEFAULT=120,
        STARPLATE_DATA=os.environ.get("STARPLATE_DATA", str(project / "data")),
        STARPLATE_ARCHIVE=str(project / "samples"),
        SEED_ARCHIVE=True,
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)
    workspace = FrameStore(app.config["STARPLATE_DATA"])
    app.config["FRAME_STORE"] = workspace
    if app.config["SEED_ARCHIVE"]:
        initialize_archive(workspace, Path(app.config["STARPLATE_ARCHIVE"]))
    app.register_blueprint(api)
    app.register_blueprint(exports)
    app.register_blueprint(color_api)

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html", max_age=0)

    @app.errorhandler(RequestError)
    def invalid_request(error):
        return error.payload(), error.status

    @app.errorhandler(HTTPException)
    def http_error(error):
        if not request.path.startswith("/api/"):
            return error
        details = error.description
        if error.code == 413:
            details = "The upload or request exceeds the size limit."
        return {"error": details}, error.code

    @app.after_request
    def response_contract(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if request.path.startswith("/api/") and response.mimetype == "application/json":
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


def main():
    app = create_app()
    host = os.environ.get("STARPLATE_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    app.run(host=host, port=port, threaded=True, use_reloader=False)
