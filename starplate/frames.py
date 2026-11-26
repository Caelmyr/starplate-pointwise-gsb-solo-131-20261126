from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from itertools import islice
import bz2
import gzip
import re
import warnings

from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import RequestError
from .geometry import finite_or_none
from .limits import MAX_DECODED_BYTES, MAX_DIMENSION, MAX_FILE_BYTES, MAX_PIXELS


SCIENCE_TYPES = ("fits", "raster")
FITS_SUFFIXES = (".fit", ".fits", ".fts", ".fit.gz", ".fits.gz", ".fts.gz",
                 ".fit.bz2", ".fits.bz2", ".fts.bz2")
RASTER_TYPES = ("PNG", "TIFF", "JPEG")


@dataclass(frozen=True)
class ImagePlane:
    pixels: np.ndarray
    header: fits.Header
    file_type: str
    hdu: int
    unit: str
    candidates: tuple[dict, ...]


def checked_shape(shape):
    if len(shape) != 2:
        raise RequestError("Choose a two-dimensional image plane.", "image")
    h, w = shape
    if h < 2 or w < 2:
        raise RequestError("Image must be at least 2 by 2 pixels.", "image")
    if max(h, w) > MAX_DIMENSION or h * w > MAX_PIXELS:
        raise RequestError(
            f"Image exceeds the {MAX_DIMENSION:,}-pixel edge or {MAX_PIXELS:,}-pixel area limit.",
            "image",
            413,
        )
    return h, w


def normalized_pixels(array):
    checked_shape(array.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        pixels = np.array(array, dtype=np.float32, order="C", copy=True)
    pixels[~np.isfinite(pixels)] = np.nan
    if not np.isfinite(pixels).any():
        raise RequestError("The selected plane contains no finite pixels.", "image")
    return pixels


def decompress_frame(raw):
    if len(raw) > MAX_FILE_BYTES:
        raise RequestError("Image transfer exceeds the file size limit.", "image", 413)
    if raw.startswith(b"BZh"):
        opener = bz2.BZ2File
    elif raw.startswith(b"\x1f\x8b"):
        opener = gzip.GzipFile
    else:
        return raw
    try:
        with opener(BytesIO(raw)) as stream:
            decoded = stream.read(MAX_DECODED_BYTES + 1)
    except (OSError, EOFError, ValueError) as error:
        raise RequestError("Compressed FITS data could not be decoded.", "image") from error
    if len(decoded) > MAX_DECODED_BYTES:
        raise RequestError("Expanded FITS input exceeds the decode limit.", "image", 413)
    return decoded


def is_fits(name, raw):
    return raw.startswith(b"SIMPLE  =") or name.casefold().endswith(FITS_SUFFIXES)


def list_planes(hdul):
    planes = []
    image_classes = (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)
    for index, hdu in enumerate(hdul):
        if index > 63:
            break
        if not isinstance(hdu, image_classes):
            continue
        if hdu.header.get("NAXIS") != 2 or hdu.shape is None:
            continue
        try:
            height, width = checked_shape(hdu.shape)
        except RequestError:
            continue
        planes.append({
            "hdu": index,
            "name": str(hdu.name),
            "width": width,
            "height": height,
            "unit": str(hdu.header.get("BUNIT", "")).strip(),
        })
    return tuple(planes)


def read_fits(raw, hdu_index):
    raw = decompress_frame(raw)
    try:
        with fits.open(BytesIO(raw), memmap=False, lazy_load_hdus=True) as hdul:
            candidates = list_planes(hdul)
            if not candidates:
                raise RequestError("FITS contains no supported two-dimensional image plane.", "hdu")
            if hdu_index is None:
                candidates_by_area = sorted(
                    candidates, key=lambda row: (row["hdu"] > 0, -row["width"] * row["height"], row["hdu"])
                )
                hdu_index = candidates_by_area[0]["hdu"]
            if hdu_index not in {row["hdu"] for row in candidates}:
                raise RequestError("Selected HDU is not a supported image plane.", "hdu")
            hdu = hdul[hdu_index]
            header = hdu.header.copy()
            pixels = normalized_pixels(hdu.data)
            return ImagePlane(pixels, header, "fits", hdu_index,
                              str(header.get("BUNIT", "")).strip()[:30], candidates)
    except RequestError:
        raise
    except (OSError, ValueError, TypeError, IndexError, OverflowError) as error:
        raise RequestError("FITS headers or pixels are invalid.", "image") from error


def raster_luminance(image):
    if image.mode in ("I", "I;16", "I;16L", "I;16B", "F", "L", "1"):
        return np.asarray(image)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    weights = np.array([.2126, .7152, .0722], dtype=np.float32)
    luma = rgba[:, :, :3] @ weights
    luma[rgba[:, :, 3] < 255] = np.nan
    return luma


def read_raster(raw, index):
    if index not in (None, 0):
        raise RequestError("Raster input provides only plane 0.", "hdu")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in RASTER_TYPES:
                raise RequestError("Choose a FITS, PNG, JPEG or TIFF image.", "image")
            checked_shape((image.height, image.width))
            if getattr(image, "n_frames", 1) != 1:
                raise RequestError("Multi-page raster images must be separated before import.", "image")
            pixels = normalized_pixels(raster_luminance(image))
            header = fits.Header()
            header["ORIGIN"] = "StarPlate raster import"
            header["BUNIT"] = "level"
            header["EXTNAME"] = "IMAGE"
            return ImagePlane(pixels, header, "raster", 0, "level", ())
    except RequestError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as error:
        raise RequestError("Raster image could not be decoded.", "image") from error


def read_plane(name, raw, hdu_index=None):
    if not raw:
        raise RequestError("Select a nonempty image to import.", "image")
    if len(raw) > MAX_FILE_BYTES:
        raise RequestError("Image transfer exceeds the file size limit.", "image", 413)
    return read_fits(raw, hdu_index) if is_fits(name, raw) else read_raster(raw, hdu_index)


def original_file_name(name):
    safe = re.sub(r"[^a-zA-Z0-9_. +()-]", "_", str(name).replace("\\", "/").rsplit("/", 1)[-1])
    return safe[:110].strip(" .") or "image"


def source_digest(raw):
    return sha256(raw).hexdigest()


def celestial_wcs(header):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FITSFixedWarning)
        try:
            wcs = WCS(header, relax=True)
            if wcs.has_celestial:
                return wcs.celestial
        except (ValueError, TypeError, MemoryError):
            pass
    return None


def safe_header_rows(header, maximum=160):
    rows = []
    for card in islice(header.cards, maximum):
        value = card.value
        if isinstance(value, float):
            value = finite_or_none(value)
        elif not isinstance(value, (bool, int, str, type(None))):
            value = str(value)
        rows.append({
            "keyword": card.keyword,
            "value": value,
            "comment": str(card.comment),
        })
    return rows


def encode_header(header):
    return header.tostring(sep="\n", endcard=False, padding=False)


def decode_header(serialized):
    return fits.Header.fromstring(serialized, sep="\n")


def byte_stream_fits(pixels, header=None, extra=None):
    primary = fits.PrimaryHDU(np.asarray(pixels, dtype=np.float32), header=header)
    stream = BytesIO()
    hdus = fits.HDUList([primary, *(extra or [])])
    hdus.writeto(stream, output_verify="silentfix", checksum=True)
    stream.seek(0)
    return stream


def crop_header(original, box):
    header = original.copy()
    if "CRPIX1" in header:
        header["CRPIX1"] = float(header["CRPIX1"]) - box.x
    if "CRPIX2" in header:
        header["CRPIX2"] = float(header["CRPIX2"]) - box.y
    if "LTV1" in header:
        header["LTV1"] = float(header["LTV1"]) - box.x
    if "LTV2" in header:
        header["LTV2"] = float(header["LTV2"]) - box.y
    header["SRCX"] = box.x, "Original detector column of left pixel"
    header["SRCY"] = box.y, "Original detector row of first array pixel"
    header["ORIGHDU"] = (header.get("ORIGHDU", 0), "Input FITS image plane")
    header["HISTORY"] = "StarPlate pixel-preserving subset"
    return header
