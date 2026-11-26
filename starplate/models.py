from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import Box


@dataclass(frozen=True)
class FrameRecord:
    key: str
    title: str
    original_name: str
    origin: str
    source_url: str
    digest: str
    width: int
    height: int
    unit: str
    created_at: str
    file_type: str
    hdu: int
    has_wcs: bool
    lineage: dict[str, Any]

    def payload(self):
        return {
            "key": self.key,
            "title": self.title,
            "original_name": self.original_name,
            "origin": self.origin,
            "source_url": self.source_url,
            "digest": self.digest,
            "width": self.width,
            "height": self.height,
            "unit": self.unit,
            "created_at": self.created_at,
            "file_type": self.file_type,
            "hdu": self.hdu,
            "has_wcs": self.has_wcs,
            "lineage": self.lineage,
        }


@dataclass(frozen=True)
class Frame:
    record: FrameRecord
    pixels: np.ndarray
    header: Any

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def valid(self):
        return np.isfinite(self.pixels)

    @property
    def box(self):
        return Box.full(self.shape)


@dataclass(frozen=True)
class Note:
    key: str
    frame_key: str
    title: str
    text: str
    x: float
    y: float
    created_at: str

    def payload(self):
        return {
            "key": self.key,
            "frame_key": self.frame_key,
            "title": self.title,
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SavedMeasurement:
    key: str
    frame_key: str
    operation: str
    title: str
    parameters: dict[str, Any]
    result: dict[str, Any]
    created_at: str

    def payload(self):
        return {
            "key": self.key,
            "frame_key": self.frame_key,
            "operation": self.operation,
            "title": self.title,
            "parameters": self.parameters,
            "result": self.result,
            "created_at": self.created_at,
        }
