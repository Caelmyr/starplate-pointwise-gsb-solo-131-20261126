from dataclasses import dataclass
from math import ceil, floor, hypot, isfinite

import numpy as np

from .errors import RequestError
from .inputs import Fields


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def full(cls, shape):
        return cls(0, 0, shape[1], shape[0])

    @classmethod
    def from_fields(cls, fields, shape):
        if not isinstance(fields, Fields):
            fields = Fields(fields, "box")
        x = fields.integer("x", 0, 0, shape[1] - 1)
        y = fields.integer("y", 0, 0, shape[0] - 1)
        width = fields.integer("width", shape[1] - x, 1, shape[1] - x)
        height = fields.integer("height", shape[0] - y, 1, shape[0] - y)
        return cls(x, y, width, height)

    @classmethod
    def around(cls, center, radius):
        x, y = center
        left = floor(x - radius)
        top = floor(y - radius)
        right = ceil(x + radius) + 1
        bottom = ceil(y + radius) + 1
        return cls(left, top, right - left, bottom - top)

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (self.x + (self.width - 1) / 2, self.y + (self.height - 1) / 2)

    @property
    def slices(self):
        return slice(self.y, self.bottom), slice(self.x, self.right)

    def extract(self, image):
        return image[self.slices]

    def intersect(self, other):
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Box(left, top, right - left, bottom - top)

    def offset(self, dx, dy):
        return Box(self.x + dx, self.y + dy, self.width, self.height)

    def covers(self, point):
        x, y = point
        return self.x - .5 <= x < self.right - .5 and self.y - .5 <= y < self.bottom - .5

    def covers_box(self, other):
        return (
            self.x <= other.x and self.y <= other.y
            and self.right >= other.right and self.bottom >= other.bottom
        )

    def centers(self):
        y, x = np.ogrid[self.y:self.bottom, self.x:self.right]
        return x, y

    def payload(self):
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def point_in_frame(point, shape, field="point"):
    if not Box.full(shape).covers(point):
        raise RequestError("Point lies outside the original image.", field)
    return point


def index_for_point(point):
    return floor(point[0] + .5), floor(point[1] + .5)


def disk_support(domain, center, radius):
    bounding = Box.around(center, radius)
    box = domain.intersect(bounding)
    if box is None:
        return None, None
    x, y = box.centers()
    support = np.square(x - center[0]) + np.square(y - center[1]) <= radius ** 2
    return box, support


def sample_segment(start, end, spacing=1.0, max_points=8192):
    length = hypot(end[0] - start[0], end[1] - start[1])
    count = max(1, ceil(length / spacing)) + 1
    if count > max_points:
        raise RequestError(f"The segment would need more than {max_points} samples.", "segment")
    distances = np.linspace(0, length, count)
    x = np.linspace(start[0], end[0], count)
    y = np.linspace(start[1], end[1], count)
    return distances, x, y


def segment_projection(point, start, end):
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    squared = vx * vx + vy * vy
    if squared == 0:
        return 0.0, hypot(point[0] - start[0], point[1] - start[1])
    t = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / squared
    use_t = min(1.0, max(0.0, t))
    distance = hypot(point[0] - start[0] - use_t * vx, point[1] - start[1] - use_t * vy)
    return t, distance


def angle_normalize(degrees, period=360.0):
    return float(degrees % period)


def finite_or_none(value):
    if value is None or not isfinite(value):
        return None
    return float(value)


def contiguous_ranges(flags):
    visible = np.asarray(flags, dtype=bool)
    if visible.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(visible, prepend=False, append=False))
    return [(int(a), int(b)) for a, b in zip(breaks[::2], breaks[1::2])]


def downsample_shape(shape, longest):
    factor = min(1.0, longest / max(shape))
    return tuple(max(1, round(n * factor)) for n in shape)


def scaled_box(box, factor):
    left = floor(box.x * factor)
    top = floor(box.y * factor)
    right = ceil(box.right * factor)
    bottom = ceil(box.bottom * factor)
    return Box(left, top, right - left, bottom - top)
