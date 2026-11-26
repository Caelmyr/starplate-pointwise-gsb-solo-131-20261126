from dataclasses import dataclass
from math import acos, atan2, cos, degrees, hypot, isfinite, radians, sin, sqrt

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import deg
import numpy as np

from .errors import RequestError
from .frames import celestial_wcs
from .geometry import Box, finite_or_none, point_in_frame
from .skycoords import angular_step, world_at


MEASURE_DEGREE = 3600.0


@dataclass(frozen=True)
class PixelSegment:
    start: tuple[float, float]
    end: tuple[float, float]

    @classmethod
    def from_fields(cls, frame, fields):
        start = point_in_frame(fields.point("start"), frame.shape, "start")
        end = point_in_frame(fields.point("end"), frame.shape, "end")
        if start == end:
            raise RequestError("Choose two different detector positions.", "end")
        return cls(start, end)

    @property
    def vector(self):
        return self.end[0] - self.start[0], self.end[1] - self.start[1]

    @property
    def length(self):
        return hypot(*self.vector)

    @property
    def angle(self):
        return degrees(atan2(self.vector[1], self.vector[0])) % 360

    @property
    def midpoint(self):
        return (self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2


def spherical_segment(wcs, segment):
    a = world_at(wcs, *segment.start)
    b = world_at(wcs, *segment.end)
    if a is None or b is None:
        return None
    first = SkyCoord(a["ra_deg"], a["dec_deg"], unit=(deg, deg), frame="icrs")
    second = SkyCoord(b["ra_deg"], b["dec_deg"], unit=(deg, deg), frame="icrs")
    east, north = first.spherical_offsets_to(second)
    return {
        "start": a,
        "end": b,
        "arcseconds": float(first.separation(second).arcsec),
        "position_angle": float(first.position_angle(second).degree),
        "east": float(east.arcsec),
        "north": float(north.arcsec),
    }


def measure_segment(frame, fields):
    segment = PixelSegment.from_fields(frame, fields)
    wcs = celestial_wcs(frame.header)
    angular = spherical_segment(wcs, segment)
    placement = world_at(wcs, *segment.midpoint)
    return {
        "start": list(segment.start),
        "end": list(segment.end),
        "midpoint": list(segment.midpoint),
        "pixels": segment.length,
        "display_angle_clockwise": segment.angle,
        "vector": list(segment.vector),
        "celestial": angular,
        "midpoint_world": placement,
        "midpoint_scale": angular_step(wcs, *segment.midpoint) if wcs is not None else None,
    }


def arc_side(value):
    return (value + 180) % 360 - 180


def local_wcs_error(frame, count_x, count_y):
    wcs = celestial_wcs(frame.header)
    if wcs is None:
        raise RequestError("Scale diagnostics require a usable FITS WCS.", "frame")
    width = frame.record.width
    height = frame.record.height
    xs = np.linspace(0, width - 1, count_x)
    ys = np.linspace(0, height - 1, count_y)
    scales = []
    for sy in ys:
        for sx in xs:
            metrics = angular_step(wcs, float(sx), float(sy))
            if metrics is None:
                continue
            scales.append({"x": float(sx), "y": float(sy), **metrics})
    center = angular_step(wcs, *frame.box.center)
    deviations = []
    for sample in scales:
        orientation = arc_side(sample["bearing_x_deg"] - center["bearing_x_deg"]) if center else None
        fraction = (sample["area_arcsec2"] / center["area_arcsec2"] - 1)             if center and center["area_arcsec2"] else None
        deviations.append({**sample, "bearing_difference": orientation, "area_difference": fraction})
    ranges = {}
    for key in ("scale_x_arcsec", "scale_y_arcsec", "bearing_difference", "area_difference"):
        values = [sample[key] for sample in deviations if sample[key] is not None]
        ranges[key] = {"min": min(values), "max": max(values)} if values else None
    return {
        "width": width, "height": height,
        "center": center, "samples": deviations, "ranges": ranges,
    }


def perimeter_points(box, samples):
    right = box.right - 1
    bottom = box.bottom - 1
    edges = [(box.x, box.y), (right, box.y), (right, bottom), (box.x, bottom), (box.x, box.y)]
    for first, end in zip(edges, edges[1:]):
        for t in np.linspace(0, 1, samples, endpoint=False):
            yield float(first[0] + t * (end[0] - first[0])), float(first[1] + t * (end[1] - first[1]))


def spherical_area(vertices):
    if len(vertices) < 3:
        return None
    coordinates = SkyCoord(
        [point["ra_deg"] for point in vertices],
        [point["dec_deg"] for point in vertices],
        unit=(deg, deg),
        frame="icrs",
    )
    vectors = coordinates.cartesian.xyz.value.T
    center = np.sum(vectors, axis=0)
    scale = np.linalg.norm(center)
    if scale < 1e-9:
        return None
    center /= scale
    area = 0.0
    for a, b in zip(vectors, np.roll(vectors, -1, axis=0)):
        scalar = np.dot(center, np.cross(a, b))
        denominator = 1 + np.dot(center, a) + np.dot(a, b) + np.dot(b, center)
        area += 2 * atan2(float(scalar), float(denominator))
    return float(abs(area) * (180 / np.pi) ** 2)


def sky_footprint(frame, box, samples):
    wcs = celestial_wcs(frame.header)
    if wcs is None:
        raise RequestError("Footprints require a usable FITS WCS.", "frame")
    boundary = []
    worlds = []
    for x, y in perimeter_points(box, samples):
        sky = world_at(wcs, x, y)
        if sky is not None:
            boundary.append({"x": x, "y": y, "world": sky})
            worlds.append(sky)
    center = world_at(wcs, *box.center)
    surface = spherical_area(worlds) if len(worlds) == samples * 4 else None
    return {
        "region": box.payload(),
        "center": center,
        "boundary": boundary,
        "square_degrees": surface,
        "square_arcminutes": surface * 3600 if surface is not None else None,
    }


def exposure_instant(header):
    keys = ("DATE-OBS", "DATEOBS", "MJD-OBS", "MJD")
    for key in keys:
        raw = header.get(key)
        if raw is None:
            continue
        try:
            if key.startswith("MJD"):
                instant = Time(float(raw), format="mjd", scale="utc")
            else:
                instant = Time(str(raw).strip().replace("Z", ""), scale="utc")
        except (ValueError, TypeError):
            continue
        return {"source": key, "utc": instant.utc.isot, "mjd": float(instant.mjd)}
    return None


def header_number(header, keys):
    for key in keys:
        try:
            value = float(header.get(key))
        except (ValueError, TypeError):
            continue
        if isfinite(value):
            return {"source": key, "value": value}
    return None


def observation_info(frame):
    header = frame.header
    name_keys = ("OBJECT", "FILTER", "TELESCOP", "INSTRUME", "OBSERVER", "ORIGIN", "TIMESYS", "PROJID")
    text_fields = {key.casefold(): str(header[key]).strip() for key in name_keys if key in header}
    instant = exposure_instant(header)
    exposure = header_number(header, ("EXPTIME", "EXPOSURE", "TELAPSE"))
    airmass = header_number(header, ("AIRMASS", "AMSTART", "AMEND"))
    camera_gain = header_number(header, ("GAIN", "EGAIN", "CCDGAIN"))
    read_noise = header_number(header, ("RDNOISE", "READNOIS", "RON"))
    temperature = header_number(header, ("CCD-TEMP", "CCDTEMP", "SENSOR-T"))
    quote = header_number(header, ("SEEING", "FWHM", "PSF_FWHM"))
    return {
        "metadata": text_fields,
        "start": instant,
        "exposure_seconds": exposure,
        "airmass": airmass,
        "camera_gain": camera_gain,
        "read_noise": read_noise,
        "temperature": temperature,
        "header_seeing": quote,
        "hdu": frame.record.hdu,
        "stored_unit": frame.record.unit,
    }
