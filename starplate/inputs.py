from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

from .errors import RequestError


class Fields:
    def __init__(self, values=None, prefix=""):
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise RequestError("Expected a set of named fields.", prefix)
        self.values = values
        self.prefix = prefix

    def path(self, key):
        if self.prefix and key:
            return f"{self.prefix}.{key}"
        return key or self.prefix

    def raw(self, key, default=None):
        return self.values.get(key, default)

    def _number(self, key, default):
        value = self.raw(key, default)
        if isinstance(value, bool) or value is None:
            raise RequestError("Enter a number.", self.path(key))
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as error:
                raise RequestError("Enter a number.", self.path(key)) from error
        if not isinstance(value, Real) or not isfinite(value):
            raise RequestError("Enter a finite number.", self.path(key))
        return value

    def _bounds(self, key, value, minimum, maximum):
        if minimum is not None and value < minimum:
            raise RequestError(f"Enter at least {minimum:g}.", self.path(key))
        if maximum is not None and value > maximum:
            raise RequestError(f"Enter at most {maximum:g}.", self.path(key))
        return value

    def number(self, key, default=None, minimum=None, maximum=None):
        value = float(self._number(key, default))
        return self._bounds(key, value, minimum, maximum)

    def integer(self, key, default=None, minimum=None, maximum=None):
        value = self._number(key, default)
        if not isinstance(value, Integral) and value != int(value):
            raise RequestError("Enter a whole number.", self.path(key))
        return self._bounds(key, int(value), minimum, maximum)

    def text(self, key, default="", minimum=0, maximum=150):
        value = self.raw(key, default)
        if not isinstance(value, str):
            raise RequestError("Enter text.", self.path(key))
        value = value.strip()
        if len(value) < minimum:
            raise RequestError(f"Enter at least {minimum} characters.", self.path(key))
        if len(value) > maximum:
            raise RequestError(f"Use no more than {maximum} characters.", self.path(key))
        return value

    def flag(self, key, default=False):
        value = self.raw(key, default)
        if isinstance(value, bool):
            return value
        if value in ("true", "1"):
            return True
        if value in ("false", "0"):
            return False
        raise RequestError("Choose yes or no.", self.path(key))

    def option(self, key, choices, default=None):
        value = self.raw(key, default)
        if value not in choices:
            raise RequestError("Choose one of the available options.", self.path(key))
        return value

    def list(self, key, default=(), minimum=0, maximum=100):
        value = self.raw(key, default)
        if not isinstance(value, (list, tuple)):
            raise RequestError("Provide a list.", self.path(key))
        if len(value) < minimum:
            raise RequestError(f"Provide at least {minimum} entries.", self.path(key))
        if len(value) > maximum:
            raise RequestError(f"Provide no more than {maximum} entries.", self.path(key))
        return value

    def child(self, key):
        return Fields(self.raw(key), self.path(key))

    def point(self, key, default=None):
        point = self.raw(key, default)
        path = self.path(key)
        if point is None:
            raise RequestError("Mark a point on the image.", path)
        if isinstance(point, (list, tuple)):
            if len(point) != 2:
                raise RequestError("Provide an x and y coordinate.", path)
            point = {"x": point[0], "y": point[1]}
        fields = Fields(point, path)
        return (fields.number("x"), fields.number("y"))
