"""Compatibility shims required before importing chumpy/FLAME."""

import inspect
from collections import namedtuple

import numpy as np


for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec
