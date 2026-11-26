from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import sqlite3
from uuid import uuid4

import numpy as np

from .errors import RequestError
from .frames import celestial_wcs, decode_header, encode_header, normalized_pixels, original_file_name
from .geometry import point_in_frame
from .limits import MAX_DERIVED_FRAMES, MAX_MEASUREMENTS, MAX_NOTES, MAX_STORE_BYTES
from .models import Frame, FrameRecord, Note, SavedMeasurement


KEY_FORMAT = re.compile(r"[a-z0-9][a-z0-9-]{0,71}")
FRAME_TABLE = """
CREATE TABLE IF NOT EXISTS frames (
    key TEXT PRIMARY KEY, title TEXT NOT NULL, original_name TEXT NOT NULL,
    origin TEXT NOT NULL, source_url TEXT NOT NULL, digest TEXT NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, unit TEXT NOT NULL,
    created_at TEXT NOT NULL, file_type TEXT NOT NULL, hdu INTEGER NOT NULL,
    has_wcs INTEGER NOT NULL, header TEXT NOT NULL, lineage TEXT NOT NULL,
    file_name TEXT NOT NULL UNIQUE, byte_size INTEGER NOT NULL,
    UNIQUE (origin, digest, hdu)
)
"""
MEASUREMENT_TABLE = """
CREATE TABLE IF NOT EXISTS measurements (
    key TEXT PRIMARY KEY, frame_key TEXT NOT NULL REFERENCES frames(key) ON DELETE CASCADE,
    operation TEXT NOT NULL, title TEXT NOT NULL, parameters TEXT NOT NULL,
    result TEXT NOT NULL, created_at TEXT NOT NULL
)
"""
NOTE_TABLE = """
CREATE TABLE IF NOT EXISTS notes (
    key TEXT PRIMARY KEY, frame_key TEXT NOT NULL REFERENCES frames(key) ON DELETE CASCADE,
    title TEXT NOT NULL, text TEXT NOT NULL, x REAL NOT NULL, y REAL NOT NULL,
    created_at TEXT NOT NULL
)
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def validated_key(key):
    if not isinstance(key, str) or not KEY_FORMAT.fullmatch(key):
        raise RequestError("Record identifier is invalid.", "key", 404)
    return key


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def record_from_row(row):
    return FrameRecord(
        key=row["key"],
        title=row["title"],
        original_name=row["original_name"],
        origin=row["origin"],
        source_url=row["source_url"],
        digest=row["digest"],
        width=row["width"],
        height=row["height"],
        unit=row["unit"],
        created_at=row["created_at"],
        file_type=row["file_type"],
        hdu=row["hdu"],
        has_wcs=bool(row["has_wcs"]),
        lineage=json.loads(row["lineage"]),
    )


def note_from_row(row):
    return Note(
        key=row["key"], frame_key=row["frame_key"], title=row["title"], text=row["text"],
        x=row["x"], y=row["y"], created_at=row["created_at"],
    )


def measurement_from_row(row):
    return SavedMeasurement(
        key=row["key"], frame_key=row["frame_key"], operation=row["operation"],
        title=row["title"], parameters=json.loads(row["parameters"]),
        result=json.loads(row["result"]), created_at=row["created_at"],
    )


class FrameStore:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.planes = self.root / "planes"
        self.planes.mkdir(exist_ok=True)
        self.database = self.root / "starplate.sqlite"
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(FRAME_TABLE)
            conn.execute(MEASUREMENT_TABLE)
            conn.execute(NOTE_TABLE)
            conn.execute("CREATE INDEX IF NOT EXISTS notes_frame ON notes(frame_key, created_at)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS measurements_frame ON measurements(frame_key, created_at)"
            )

    @contextmanager
    def connect(self, write=False):
        conn = sqlite3.connect(self.database, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write:
                conn.commit()
        except Exception:
            if write:
                conn.rollback()
            raise
        finally:
            conn.close()

    def existing(self, digest, hdu, origin):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM frames WHERE origin=? AND digest=? AND hdu=?",
                (origin, digest, hdu),
            ).fetchone()
        return record_from_row(row) if row else None

    def list_frames(self):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM frames ORDER BY (origin='archive') DESC, created_at DESC, key"
            ).fetchall()
        return [record_from_row(row) for row in rows]

    def _row(self, conn, key):
        row = conn.execute("SELECT * FROM frames WHERE key=?", (validated_key(key),)).fetchone()
        if row is None:
            raise RequestError("Image is no longer in this workspace.", "frame", 404)
        return row

    def get_record(self, key):
        with self.connect() as conn:
            row = self._row(conn, key)
        return record_from_row(row)

    def open(self, key):
        with self.connect() as conn:
            row = self._row(conn, key)
            path = self.planes / row["file_name"]
            try:
                pixels = np.load(path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError) as error:
                raise RequestError("Stored image is unavailable.", "frame", 422) from error
        header = decode_header(row["header"])
        if pixels.shape != (row["height"], row["width"]):
            raise RequestError("Stored image dimensions do not match its record.", "frame", 422)
        return Frame(record_from_row(row), pixels, header)

    def store(
        self,
        plane,
        *,
        title,
        original_name,
        origin,
        digest,
        key=None,
        source_url="",
        lineage=None,
    ):
        pixels = normalized_pixels(plane.pixels)
        size = pixels.nbytes
        key = validated_key(key or f"plane-{uuid4().hex}")
        name = f"{key}.npy"
        final = self.planes / name
        pending = self.planes / f".pending-{uuid4().hex}"
        installed = False
        try:
            with self.connect(write=True) as conn:
                existing = conn.execute(
                    "SELECT * FROM frames WHERE origin=? AND digest=? AND hdu=?",
                    (origin, digest, plane.hdu),
                ).fetchone()
                if existing is not None:
                    return record_from_row(existing), False
                if final.exists():
                    raise RequestError("This image identifier is already in use.", "frame", 409)
                self._check_budget(conn, size, origin)
                with pending.open("xb") as stream:
                    np.save(stream, pixels, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(pending, final)
                installed = True
                record = self._insert(
                    conn,
                    key=key, title=title, original_name=original_name,
                    origin=origin, digest=digest, source_url=source_url, plane=plane,
                    width=pixels.shape[1], height=pixels.shape[0],
                    lineage=lineage or {}, name=name, size=size,
                )
            return record, True
        except Exception:
            if installed:
                final.unlink(missing_ok=True)
            raise
        finally:
            pending.unlink(missing_ok=True)

    def _check_budget(self, conn, byte_size, origin):
        row = conn.execute(
            "SELECT COALESCE(SUM(byte_size),0) AS used, SUM(origin='derived') AS derived FROM frames"
        ).fetchone()
        if row["used"] + byte_size > MAX_STORE_BYTES:
            raise RequestError("Workspace storage limit reached. Remove a derived image first.", "image", 413)
        if origin == "derived" and (row["derived"] or 0) >= MAX_DERIVED_FRAMES:
            raise RequestError("Remove a derived image before creating another.", "image", 413)

    def _insert(
        self, conn, *, key, title, original_name, origin, digest, source_url,
        plane, width, height, lineage, name, size,
    ):
        conn.execute(
            """INSERT INTO frames (
                key,title,original_name,origin,source_url,digest,width,height,unit,created_at,
                file_type,hdu,has_wcs,header,lineage,file_name,byte_size
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key, title[:100], original_file_name(original_name), origin, source_url, digest,
                width, height, plane.unit, now(), plane.file_type, plane.hdu,
                int(celestial_wcs(plane.header) is not None), encode_header(plane.header),
                json_text(lineage), name, size,
            ),
        )
        return record_from_row(self._row(conn, key))

    def rename(self, key, title):
        with self.connect(write=True) as conn:
            self._row(conn, key)
            conn.execute("UPDATE frames SET title=? WHERE key=?", (title, key))
            return record_from_row(self._row(conn, key))

    def remove(self, key):
        with self.connect(write=True) as conn:
            row = self._row(conn, key)
            if row["origin"] == "archive":
                raise RequestError("Bundled archive images are read-only.", "frame", 409)
            dependent = conn.execute(
                """SELECT candidate.title FROM frames AS candidate
                   JOIN json_each(candidate.lineage, '$.parents') AS parent
                   ON parent.value = ? LIMIT 1""",
                (key,),
            ).fetchone()
            if dependent:
                raise RequestError(
                    f"Retained result “{dependent['title']}” still uses this image.", "frame", 409
                )
            conn.execute("DELETE FROM frames WHERE key=?", (key,))
            path = self.planes / row["file_name"]
        path.unlink(missing_ok=True)

    def list_notes(self, key):
        with self.connect() as conn:
            self._row(conn, key)
            rows = conn.execute(
                "SELECT * FROM notes WHERE frame_key=? ORDER BY created_at DESC, key", (key,)
            ).fetchall()
        return [note_from_row(row) for row in rows]

    def add_note(self, frame_key, title, text, point):
        key = f"note-{uuid4().hex}"
        with self.connect(write=True) as conn:
            frame = self._row(conn, frame_key)
            point_in_frame(point, (frame["height"], frame["width"]))
            count = conn.execute("SELECT COUNT(*) FROM notes WHERE frame_key=?", (frame_key,)).fetchone()[0]
            if count >= MAX_NOTES:
                raise RequestError("This image already holds the maximum number of notes.", "note", 409)
            conn.execute(
                "INSERT INTO notes (key,frame_key,title,text,x,y,created_at) VALUES (?,?,?,?,?,?,?)",
                (key, frame_key, title, text, *point, now()),
            )
            row = conn.execute("SELECT * FROM notes WHERE key=?", (key,)).fetchone()
        return note_from_row(row)

    def remove_note(self, frame_key, note_key):
        with self.connect(write=True) as conn:
            self._row(conn, frame_key)
            cursor = conn.execute(
                "DELETE FROM notes WHERE frame_key=? AND key=?", (frame_key, validated_key(note_key))
            )
            if cursor.rowcount != 1:
                raise RequestError("Note was not found on this frame.", "note", 404)

    def save_measurement(self, frame_key, operation, title, parameters, result):
        key = f"measure-{uuid4().hex}"
        with self.connect(write=True) as conn:
            self._row(conn, frame_key)
            count = conn.execute(
                "SELECT COUNT(*) FROM measurements WHERE frame_key=?", (frame_key,)
            ).fetchone()[0]
            if count >= MAX_MEASUREMENTS:
                raise RequestError("This image already holds the maximum number of measurements.", "measurement", 409)
            content = json_text(result)
            if len(content) > 4_000_000:
                raise RequestError("The result is too large to retain as one measurement.", "measurement", 413)
            conn.execute(
                """INSERT INTO measurements
                   (key,frame_key,operation,title,parameters,result,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (key, frame_key, operation, title, json_text(parameters), content, now()),
            )
            row = conn.execute("SELECT * FROM measurements WHERE key=?", (key,)).fetchone()
        return measurement_from_row(row)

    def list_measurements(self, frame_key):
        with self.connect() as conn:
            self._row(conn, frame_key)
            rows = conn.execute(
                """SELECT key,frame_key,operation,title,created_at FROM measurements
                   WHERE frame_key=? ORDER BY created_at DESC,key""",
                (frame_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_measurement(self, frame_key, measurement_key):
        with self.connect() as conn:
            self._row(conn, frame_key)
            row = conn.execute(
                "SELECT * FROM measurements WHERE frame_key=? AND key=?",
                (frame_key, validated_key(measurement_key)),
            ).fetchone()
        if row is None:
            raise RequestError("Measurement was not found on this frame.", "measurement", 404)
        return measurement_from_row(row)

    def remove_measurement(self, frame_key, measurement_key):
        with self.connect(write=True) as conn:
            self._row(conn, frame_key)
            cursor = conn.execute(
                "DELETE FROM measurements WHERE frame_key=? AND key=?",
                (frame_key, validated_key(measurement_key)),
            )
            if cursor.rowcount != 1:
                raise RequestError("Measurement was not found on this frame.", "measurement", 404)
