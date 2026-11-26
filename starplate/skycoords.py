from math import atan2, degrees, hypot, isfinite, radians
import re

from astropy.coordinates import Angle, SkyCoord
from astropy.units import UnitConversionError, deg, hourangle
from astropy.wcs import NoConvergence
import numpy as np

from .errors import RequestError
from .frames import celestial_wcs
from .geometry import Box, finite_or_none


def world_at(wcs, x, y):
    if wcs is None:
        return None
    try:
        world = wcs.pixel_to_world(float(x), float(y))
        equatorial = world.icrs
        ra = float(equatorial.ra.to_value(deg) % 360)
        dec = float(equatorial.dec.to_value(deg))
    except (AttributeError, ValueError, ArithmeticError, TypeError):
        return None
    if not isfinite(ra) or not isfinite(dec):
        return None
    return {"ra_deg": ra, "dec_deg": dec}


def parse_angle(value, is_ra=False):
    field = "ra" if is_ra else "dec"
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise RequestError("Enter an angle in decimal degrees or sexagesimal.", field)
    text = str(value).strip()
    if not text or len(text) > 64:
        raise RequestError("Enter an angle in decimal degrees or sexagesimal.", field)
    sexagesimal = bool(re.search(r"[:hmsd°\s]", text, flags=re.IGNORECASE))
    units = hourangle if is_ra and sexagesimal and not re.search(r"[d°]", text) else deg
    try:
        angle = Angle(text, unit=units)
        numeric = float(angle.to_value(deg))
    except (ValueError, TypeError, UnitConversionError) as error:
        raise RequestError("Angle could not be parsed.", field) from error
    if not isfinite(numeric):
        raise RequestError("Enter a finite angle.", field)
    if is_ra:
        return numeric % 360
    if not -90 <= numeric <= 90:
        raise RequestError("Declination must lie between -90 and +90 degrees.", field)
    return numeric


def detector_at(wcs, right_ascension, declination):
    if wcs is None:
        return None
    try:
        world = SkyCoord(ra=right_ascension * deg, dec=declination * deg, frame="icrs")
        x, y = wcs.world_to_pixel(world)
        point = (float(x), float(y))
    except (ValueError, TypeError, ArithmeticError, NoConvergence):
        return None
    if not all(isfinite(index) for index in point):
        return None
    return point


def angular_step(wcs, x, y, step=.5):
    try:
        origin = wcs.pixel_to_world(x, y).icrs
        world_x = wcs.pixel_to_world(x + step, y).icrs
        world_y = wcs.pixel_to_world(x, y + step).icrs
        east_x, north_x = origin.spherical_offsets_to(world_x)
        east_y, north_y = origin.spherical_offsets_to(world_y)
        x_components = east_x.arcsec / step, north_x.arcsec / step
        y_components = east_y.arcsec / step, north_y.arcsec / step
        jacobian = np.array((x_components, y_components))
        if not np.isfinite(jacobian).all():
            return None
    except (AttributeError, ValueError, ArithmeticError, TypeError):
        return None
    return {
        "scale_x_arcsec": finite_or_none(hypot(*x_components)),
        "scale_y_arcsec": finite_or_none(hypot(*y_components)),
        "bearing_y_deg": float(degrees(atan2(*y_components)) % 360),
        "bearing_x_deg": float(degrees(atan2(*x_components)) % 360),
        "area_arcsec2": finite_or_none(abs(np.linalg.det(jacobian))),
    }


def sky_summary(frame):
    wcs = celestial_wcs(frame.header)
    if wcs is None:
        return {"available": False, "corners": [], "center": None, "scale": None}
    box = frame.box
    corners = [(box.x, box.y), (box.right - 1, box.y),
               (box.right - 1, box.bottom - 1), (box.x, box.bottom - 1)]
    return {
        "available": True,
        "corners": [
            {"x": x, "y": y, "world": world_at(wcs, x, y)} for x, y in corners
        ],
        "center": world_at(wcs, *box.center),
        "scale": angular_step(wcs, *box.center),
    }


def lookup_world(frame, fields):
    wcs = celestial_wcs(frame.header)
    if wcs is None:
        raise RequestError("This frame has no usable celestial WCS.", "frame")
    ra = parse_angle(fields.raw("ra"), is_ra=True)
    dec = parse_angle(fields.raw("dec"))
    point = detector_at(wcs, ra, dec)
    if point is None:
        return {"in_frame": False, "x": None, "y": None, "ra_deg": ra, "dec_deg": dec}
    world = world_at(wcs, *point)
    match = angular_difference((ra, dec), world)
    return {
        "in_frame": frame.box.covers(point),
        "x": point[0],
        "y": point[1],
        "ra_deg": ra,
        "dec_deg": dec,
        "round_trip_arcsec": match,
    }


def angular_difference(point, world):
    if world is None:
        return None
    sample = SkyCoord(ra=point[0] * deg, dec=point[1] * deg, frame="icrs")
    measured = SkyCoord(ra=world["ra_deg"] * deg, dec=world["dec_deg"] * deg, frame="icrs")
    return finite_or_none(sample.separation(measured).arcsec)


def celestial_grid(frame, count=5):
    wcs = celestial_wcs(frame.header)
    if wcs is None:
        return {"lines": [], "available": False}
    h, w = frame.shape
    center = world_at(wcs, *frame.box.center)
    footprints = [world_at(wcs, x, y) for x, y in
                  [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), frame.box.center]]
    footprints = [value for value in footprints if value is not None]
    if center is None or len(footprints) < 3:
        return {"lines": [], "available": False}
    unwrapped = unwrap_ra([point["ra_deg"] for point in footprints], center["ra_deg"])
    declinations = [point["dec_deg"] for point in footprints]
    bounds = (float(min(unwrapped)), float(max(unwrapped)), min(declinations), max(declinations))
    ra_ticks = np.linspace(bounds[0], bounds[1], count + 2)[1:-1]
    dec_ticks = np.linspace(bounds[2], bounds[3], count + 2)[1:-1]
    curves = []
    for ra in ra_ticks:
        segments = visible_grid_segments(
            wcs, frame.box, np.full(100, ra), np.linspace(bounds[2], bounds[3], 100)
        )
        if segments:
            curves.append({"kind": "ra", "value": float(ra % 360), "segments": segments})
    for dec in dec_ticks:
        segments = visible_grid_segments(
            wcs, frame.box, np.linspace(bounds[0], bounds[1], 100), np.full(100, dec)
        )
        if segments:
            curves.append({"kind": "dec", "value": float(dec), "segments": segments})
    return {"available": True, "lines": curves}


def unwrap_ra(values, center):
    values = np.asarray(values, dtype=float)
    return (values - center + 180) % 360 + center - 180


def visible_grid_segments(wcs, box: Box, ra_values, dec_values):
    try:
        world = SkyCoord(ra=(ra_values % 360) * deg, dec=dec_values * deg, frame="icrs")
        xs, ys = wcs.world_to_pixel(world)
    except (ValueError, TypeError, ArithmeticError, NoConvergence):
        return []
    lines, pending = [], []
    for x, y in zip(xs, ys):
        point = float(x), float(y)
        if all(isfinite(value) for value in point) and box.covers(point):
            pending.append(point)
            continue
        if len(pending) >= 2:
            lines.append(pending)
        pending = []
    if len(pending) >= 2:
        lines.append(pending)
    return lines
