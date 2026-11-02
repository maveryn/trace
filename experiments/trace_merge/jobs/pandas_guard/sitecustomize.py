"""Guard against a segfault in pandas 2.3.3's numeric inference (loaded via PYTHONPATH by jobs/eval_subset.sh).

``pandas._libs.lib.maybe_convert_numeric`` (used by ``read_excel``/``read_csv``'s python parser type inference and by
``to_numeric``) overflows its exponent accumulator on strings that start like a float with an exponent of 11 or more
digits, e.g. the SHA-256 request hash ``7649e84210128867284d…``, and the process dies with SIGSEGV. The pinned scorers
load prediction workbooks that carry such hashes in a ``request_hash`` column, so a fraction of about 1 in 2,000 rows
crashes the score worker deterministically (which hashes occur depends on the served model name).

The guard replaces, in a private copy passed to the parser only, an exponent of 10 or more significant digits by
``9999``. A string with characters after the exponent is still not numeric (the column stays object dtype with its
original values), and a purely numeric string with such an exponent gets the result pandas itself gives for an
exponent that size (an overflow is reported as unparsable, an underflow as 0.0). Exponents with leading zeros are not
large and are left alone. Nothing else is touched, and processes that never import pandas never load the hook's
target.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import re
import sys

_BAD = re.compile(r"^(\s*[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?0*)([1-9]\d{9,})")  # 10+ significant digits
_TARGET = "pandas._libs.lib"


def _neutral(s: str) -> str:
    m = _BAD.match(s)
    return m.group(1) + "9999" + s[m.end(2):] if m else s


def _guard(orig):
    def maybe_convert_numeric(values, *args, **kwargs):
        try:
            if getattr(values, "dtype", None) is not None and values.dtype == object:
                hits = [i for i, v in enumerate(values) if type(v) is str and _BAD.match(v)]
                if hits:
                    values = values.copy()
                    for i in hits:
                        values[i] = _neutral(values[i])
        except Exception:  # never let the guard itself change behaviour
            pass
        return orig(values, *args, **kwargs)

    maybe_convert_numeric.__wrapped__ = orig
    maybe_convert_numeric.__doc__ = getattr(orig, "__doc__", None)
    return maybe_convert_numeric


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            fn = getattr(module, "maybe_convert_numeric", None)
            if fn is not None and not hasattr(fn, "__wrapped__"):
                module.maybe_convert_numeric = _guard(fn)

        loader.exec_module = exec_and_patch
        return spec


if _TARGET not in sys.modules:
    sys.meta_path.insert(0, _Finder())
