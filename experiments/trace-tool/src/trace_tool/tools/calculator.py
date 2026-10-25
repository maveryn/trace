"""Calculator tool: a whitelisted arithmetic evaluator.

SECURITY: expressions come from model output and must be treated as untrusted
input. Two layers reject everything that is not plain arithmetic:

1. Before parsing, every character must be in a small ASCII whitelist and every
   identifier must be one of ``_FUNCTIONS`` / ``_CONSTANTS``. CPython 3.12+
   interns parsed identifiers as immortal strings (model-chosen names would grow
   memory without bound), and the parser writes SyntaxWarnings to stderr for
   string, escape and keyword-after-number input; neither can happen once these
   checks pass. NUL bytes and lone surrogates are rejected here too.
2. The text is parsed with ``ast.parse(mode="eval")`` and walked against a
   whitelist of node types; ``eval``/``exec`` are never used. Attribute access,
   subscripts, comprehensions, lambdas and every name outside ``_FUNCTIONS`` /
   ``_CONSTANTS`` are rejected.

Big-integer work runs in C while holding the GIL, so a slow expression would
freeze the whole asyncio harness and no timeout could interrupt it. Instead,
every operation is bounded *before* it runs:

* Size: no integer anywhere in an evaluation exceeds MAX_INT_DIGITS digits.
  ``**``, ``*``, ``<<``, ``comb``, ``perm`` and ``lcm`` check the result size
  first; ``round`` caps ``ndigits``; mean/stdev run on floats; every
  intermediate int is re-checked.
* Time: each evaluate() call gets a work budget of MAX_WORK_US, held in a
  ContextVar so concurrent evaluations never share one. Every super-linear
  big-int operation (``*``, ``//``, ``%``, ``**``/``pow``, ``<<``, factorial,
  comb, perm, gcd, lcm) charges its estimated cost before it runs. Linear work
  (``+``, ``-``, comparisons, sum, min/max) is bounded by MAX_AST_NODES.
* Output: format_result refuses results longer than MAX_RESULT_CHARS before it
  converts any int to text (int -> str is quadratic).

Review changes to this file with that threat model in mind.

Numerics. The model reads results as text, and TRACE scores its answers by exact
match against values computed with Python floats and Python's round().

* A float's value is its displayed value: 12 significant digits, 15 when
  |x| >= 1e10 (``_canon``). floor/ceil/trunc, comparisons (also inside lists),
  integer arguments and float ``//`` / ``%`` use that value, so they never
  disagree with what is shown: 0.166666666667 * 6 shows 1 and ceil() gives 1.
  From 2**53 up the displayed value is an integer that the float may not hold
  exactly; those operations use the displayed integer (``_shown``), so
  floor(2 * 6.022e23) is 1204400000000000000000000, not the float's
  1204400000000000054525952.
* a // b is the floor of the displayed a / b, computed from the unrounded
  operands: 1 // (1/6) is 6 and (pi/2) // (pi/4) is 2. When that quotient is
  displayed as an integer the remainder is a zero with the divisor's sign.
* There are no absolute thresholds. 1.6e-19 or 6.6e-34 keep full relative
  precision, display as themselves and compare correctly.
* Cancellation is removed where it happens (``_cancel``): in a float sum or
  difference whose terms cancel (also sum(), mean(), stdev's deviations from
  the mean and a float remainder). When every term stands for a decimal number
  (``_decimal_value``: an int, a literal no longer than the display, or a
  product or power of literals within a few ulps of a short decimal, as 0.8**2
  stands for 0.64), the result is their exact decimal sum: 786.3 - 786.2 is
  0.1, 77401.7 - 77400.4 == 1.3 and 87.4 * 460.8 - 40273.9194 is 0.0006. A term
  with real binary digits (5/3, 1/0.96, pi) keeps Python's float result, so
  (5/3 - 1) * 9e13 is 6e13 as in Python; rounding such a difference would move
  its digits. Then a result at most 1e-12 (1e-15 from 1e10 up, matching the
  display) of the largest term is exactly 0.0, so 0.1 * 3 - 0.3 is 0 and the
  square root of a cancelled negative radicand is 0. Sums whose terms do not
  cancel are Python's.
* round(x, n) is Python's built-in round, and the arithmetic in its first
  argument is evaluated without the decimal sums above (``_RAW``), so
  round(a - b, n) gives exactly what Python gives for the raw floats: exact ties
  go to the even digit, and round(x) of a float from 2**53 up is the float's
  exact binary integer. Operations that use the displayed value (floor, ceil,
  trunc, //, %, comparisons, integer arguments) give the same result inside
  round() as outside it: round(floor(E)) == floor(E).
* Radian trig is plain IEEE (cos(radians(90)) is 6.1e-17, tan(pi/2) is a huge
  finite number). Degree trig takes the angle as displayed (``_shown``), reduces
  it exactly and is exact at multiples of 30 and 45 degrees, also for computed
  angles: 180 - atand(1/2) - atand(2) shows 90, so its cosd is 0 and its tand is
  undefined, as for a literal 90.
* asin/acos and asind/acosd take an argument displayed as +-1 as exactly +-1,
  from either side: the cosine of two parallel vectors, 0.9999999999999998,
  gives 0 degrees, not 1.2e-06. Other arguments are used as they are.

Model-facing errors. Implicit multiplication (2(3+4), (a+b)(c+d), 2pi),
percent signs used as percent (80 * 15% + 80, which Python reads as a
remainder and silently evaluates to 0), thousands separators (max(1,200, 950),
which Python reads as max(1, 200, 950)), module prefixes (math.sqrt(2)),
method calls ([3, 5, 7].mean()), keyword arguments (round(x, ndigits=2), which
would otherwise read as an unknown constant) and factorial notation (5!,
6!/(2!*4!)) are rejected with a hint, never evaluated. A digit group of 1-3
digits followed by ",ddd" with no space is always rejected, so a compact list
such as [100,200,300] needs a space after each comma. Likewise "!=" right after
a digit or ")" is rejected (4!=24, which Python reads as the inequality
4 != 24), so an inequality needs a space before "!=". gcd, lcm and hypot take
one list or several numbers, like min and max.
"""

from __future__ import annotations

import ast
import decimal
import math
import operator
import re
import statistics
import string
from contextvars import ContextVar
from typing import Any, Callable

from .base import Tool, ToolContext, ToolError, ToolResult

MAX_EXPRESSION_CHARS = 1000
MAX_AST_NODES = 400
MAX_INT_DIGITS = 4000  # also below Python's default int<->str conversion limit
MAX_FACTORIAL_ARG = 1000
MAX_COMB_ARG = 100000
MAX_ROUND_DIGITS = 100
MAX_SEQUENCE_LEN = 1000
MAX_RESULT_CHARS = 2000
_LOG10_2 = math.log10(2)
_MAX_INT_BITS = int(MAX_INT_DIGITS / _LOG10_2)
_LN2 = math.log(2)
_LN10 = math.log(10)


def _too_large() -> ToolError:
    return ToolError("Error: result is too large")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --- numerics (see the module docstring) ----------------------------------------

# Set while the arithmetic in round()'s first argument is evaluated: cancelling
# sums and differences then skip the exact decimal sum, so round() sees Python's
# raw floats. _eval clears it again for operations on displayed values
# (_USES_SHOWN_VALUES), so those never depend on whether they sit inside round().
_RAW: ContextVar[bool] = ContextVar("calculator_raw", default=False)
_EXACT_FLOAT_INT_LIMIT = 2.0**53  # from here up a float's displayed integer may differ from its binary value


def _float_text(x: float) -> str:
    """The displayed text of a finite float: 12 significant digits, 15 from 1e10 up."""
    if x == 0:
        return "0"  # also for -0.0
    return format(x, ".12g" if abs(x) < 1e10 else ".15g")


def _canon(x: Any) -> Any:
    """The value a finite float is displayed as; every other value is returned unchanged."""
    if isinstance(x, float) and math.isfinite(x):
        shown = float(_float_text(x))
        # within ~1e294 of DBL_MAX the 15-digit text rounds above the largest float
        return shown if math.isfinite(shown) else x
    return x


def _shown(x: Any) -> Any:
    """The displayed value of x as an exact number, for comparisons and integer conversion.

    Like _canon, except that a float displayed from 2**53 up becomes the integer
    its text shows: 2 * 6.022e23 is displayed as 1.2044e+24 and is
    1204400000000000000000000 here, not the float's 1204400000000000054525952.
    Such a float has at most 309 digits, well inside MAX_INT_DIGITS.
    """
    x = _canon(x)
    if isinstance(x, float) and math.isfinite(x) and abs(x) >= _EXACT_FLOAT_INT_LIMIT:
        return int(decimal.Decimal(_float_text(x)))
    return x


# (significant digits, ulps): a float whose shortest repr is longer than the display
# stands for the decimal with at most that many digits within that many ulps of it.
# The tolerance shrinks as the decimal grid gets finer, so that a float with real
# binary digits (5/3, 311/38) almost never lands within it by chance. Finer levels
# (14 digits, or any repr of up to 15 digits below 1e10) mistook such floats for
# decimals in about 1 case in 1000 and then gave a result worse than both Python's
# and the exact one; these levels did not in 16000 sampled cases.
_DECIMAL_SNAP = ((12, 4), (13, 2))
# Exact for any sum of float-range decimals and ints below 2**1024 (at most ~650 digits from
# the largest to the smallest); arithmetic costs the digits used, not the precision.
_EXACT = decimal.Context(prec=1000)


def _decimal_value(x: Any) -> decimal.Decimal | None:
    """The decimal number x stands for, or None when its binary digits are real.

    An int stands for itself. A float stands for its shortest repr when that is
    no longer than the display (12 significant digits, 15 from 1e10 up): every
    literal of that length, including a result pasted back from the display,
    reads back as itself. A longer float stands for a decimal of at most 12
    significant digits within 4 ulps, or of at most 13 within 2 ulps: the
    rounding noise of a product, quotient or power of literals (0.8**2 is
    0.6400000000000001 and stands for 0.64). Other floats, such as 5/3, 1/0.96
    or pi, stand for no decimal.
    """
    if _is_int(x):
        return decimal.Decimal(x) if x.bit_length() <= 1024 else None
    if not isinstance(x, float) or not math.isfinite(x):
        return None
    shortest = decimal.Decimal(repr(x))
    if len(shortest.as_tuple().digits) <= (12 if abs(x) < 1e10 else 15):
        return shortest
    for digits, ulps in _DECIMAL_SNAP:
        candidate = decimal.Decimal(format(x, f".{digits}g"))
        if abs(float(candidate) - x) <= ulps * math.ulp(x):
            return candidate
    return None


def _decimal_sum(terms: list[Any]) -> float | None:
    """The exact sum of the decimals the terms stand for, as the nearest float; None if one stands for none."""
    values = []
    for term in terms:
        value = _decimal_value(term)
        if value is None:
            return None
        values.append(value)
    with decimal.localcontext(_EXACT):  # a copy: concurrent evaluations never share its flags
        return float(sum(values, decimal.Decimal(0)))


def _drop_noise(r: float, scale: float) -> float:
    """A result at the display resolution of its terms (1e-12 of scale, 1e-15 from 1e10 up) is exactly 0.0."""
    if r == 0 or abs(r) <= scale * (1e-12 if scale < 1e10 else 1e-15):
        return 0.0
    return r


def _cancel(terms: list[Any], r: Any) -> Any:
    """Remove cancellation noise from r, the float sum of `terms` (see the module docstring).

    When the terms cancel (|r| below the largest term) and each stands for a
    decimal, r is their exact decimal sum: 786.3 - 786.2 is 0.1. Otherwise r
    stays Python's float result: 5/3 - 1 is 0.6666666666666667, so
    (5/3 - 1) * 9e13 is 6e13 as in Python. Then a result at the display
    resolution of the terms is exactly 0.0. Inside round()'s argument nothing
    is changed, so round() sees exactly Python's floats.
    """
    if not isinstance(r, float) or not math.isfinite(r):
        return r
    try:
        scale = float(max(abs(t) for t in terms))
    except OverflowError:  # sum([10**400, -10**400, 0.5]): no float noise at that scale
        return r
    if _RAW.get():  # round()'s argument is Python's raw float arithmetic, noise included
        return r
    if abs(r) < scale:
        exact = _decimal_sum(terms)
        if exact is not None:
            r = exact
    return _drop_noise(r, scale)


def _add(a: Any, b: Any) -> Any:
    r = a + b
    return _cancel([a, b], r) if isinstance(r, float) else r


def _sub(a: Any, b: Any) -> Any:
    r = a - b
    return _cancel([a, -b], r) if isinstance(r, float) else r


# --- work budget ---------------------------------------------------------------
# Costs are estimated microseconds of C big-int work, measured with CPython 3.12
# on an Apple M5 and rounded up to upper bounds; la/lb are operand bit lengths,
# r the result's bit length and k the number of factors (every int is at most
# _MAX_INT_BITS ~ 13288 bits). Largest single operations the size cap allows,
# measured there (estimate in brackets):
#   comb(13290, 6645)           7 ms     [13 ms]
#   gcd of two 4000-digit ints  0.33 ms  [0.56 ms]
#   4000-digit // 2000-digit    0.13 ms  [0.29 ms]
#   2000 x 2000-digit product   0.07 ms  [0.19 ms]
#   perm(1450)                  0.18 ms  [0.48 ms]
#   factorial(1000)             0.06 ms  [0.22 ms]
#   3**8382                     0.07 ms  [0.12 ms]
# The budget fits one of those combs but not two, so on that machine any
# expression finishes in about 20 ms of big-int work plus at most a few ms of
# interpreter overhead, leaving headroom for slower or busy hosts.
MAX_WORK_US = 20_000.0
_WORK: ContextVar[list[float] | None] = ContextVar("calculator_work", default=None)


def _too_expensive() -> ToolError:
    return ToolError("Error: expression needs too much computation; simplify it or split it into several calls")


def _charge(cost_us: float) -> None:
    """Spend estimated work before an operation runs; raise once the budget is used up."""
    budget = _WORK.get()
    if budget is None:  # called outside evaluate(): still bound this one operation
        if cost_us > MAX_WORK_US:
            raise _too_expensive()
        return
    budget[0] -= cost_us
    if budget[0] < 0:
        raise _too_expensive()


def _linear_cost(*bits: float) -> float:
    return 2.0 + 1e-3 * sum(bits)


def _mul_cost(la: int, lb: int) -> float:
    return _linear_cost(la, lb) + 4e-6 * la * lb


def _div_cost(la: int, lb: int) -> float:
    return _linear_cost(la, lb) + 6e-6 * max(la - lb + 1, 0) * lb


def _gcd_cost(la: int, lb: int) -> float:
    # Lehmer gcd is quadratic, and unbalanced operands first pay for a long division
    return _div_cost(max(la, lb), min(la, lb)) + 3e-6 * la * lb


def _pow_cost(r: float) -> float:
    return _linear_cost(r) + 6e-7 * r * r


def _perm_cost(k: int, r: float) -> float:
    return 2.0 + 2.5e-5 * k * r


def _comb_cost(k: int, r: float) -> float:
    return 2.0 + 1.5e-4 * k * r


# --- bounded integer operations ---------------------------------------------------


def _int_digits_upper_bound(base: int, exponent: int) -> float:
    if base in (0, 1, -1):
        return 1.0
    if exponent.bit_length() > 64:
        return math.inf
    return exponent * math.log10(abs(base))  # math.log10 accepts arbitrarily large ints


def _safe_pow(base: Any, exponent: Any) -> Any:
    if _is_int(base) and _is_int(exponent) and exponent > 0:
        if base in (0, 1, -1):  # CPython would still square once per exponent bit
            return -1 if base == -1 and exponent & 1 else abs(base)
        digits = _int_digits_upper_bound(base, exponent)
        if digits > MAX_INT_DIGITS:
            raise _too_large()
        _charge(_pow_cost(digits / _LOG10_2))
    try:
        result = operator.pow(base, exponent)
    except OverflowError:
        raise _too_large() from None
    if isinstance(result, complex):
        raise ToolError("Error: result is not a real number")
    return result


def _int_too_large(value: int) -> bool:
    return value.bit_length() > _MAX_INT_BITS


def _checked(value: Any) -> Any:
    """Enforce the size invariant on every intermediate value."""
    if _is_int(value) and _int_too_large(value):
        raise _too_large()
    if isinstance(value, list):
        for item in value:
            _checked(item)
    return value


def _safe_mult(left: Any, right: Any) -> Any:
    if _is_int(left) and _is_int(right):
        la, lb = left.bit_length(), right.bit_length()
        if la + lb > _MAX_INT_BITS:
            raise _too_large()
        _charge(_mul_cost(la, lb))
    return operator.mul(left, right)


def _safe_lshift(left: Any, right: Any) -> Any:
    if not _is_int(left) or not _is_int(right):
        raise ToolError("Error: << needs integers")
    if right < 0:
        raise ToolError("Error: negative shift count")
    if left.bit_length() + right > _MAX_INT_BITS:
        raise _too_large()
    _charge(_linear_cost(left.bit_length() + right))
    return operator.lshift(left, right)


def _float_divmod(a: Any, b: Any) -> tuple[Any, Any]:
    """divmod with a float operand, on the displayed quotient: 0.6 // 0.2 == 3, 0.6 % 0.2 == 0.

    a // b is the floor of the displayed a / b, and a / b is computed from the
    unrounded operands. Rounding the operands to their displayed digits first
    would move the quotient by up to ~1e-11 of its value, which the quotient's
    own display does not hide: 1 // (1/6) would be 5 while 1 / (1/6) shows 6.
    When a / b displays as an integer the remainder is a zero with the divisor's
    sign, as Python's own % returns it. Quotients from 1e14 up display no
    fractional digits, so there the float divmod of the displayed operands is
    kept (1e20 % 3.0 is 1).
    """
    quotient, remainder = divmod(a, b)  # raises ZeroDivisionError / OverflowError first
    shown = _canon(a / b)
    if abs(shown) < 1e14:  # False for inf and nan
        if shown.is_integer():
            return shown, math.copysign(0.0, b)
        floored = float(math.floor(shown))
        if floored != quotient:  # not expected: a non-integer display has the same floor
            quotient, remainder = floored, a - floored * b
        # a and b are finite here (an infinite one gives an inf, nan or 0 quotient). The
        # remainder a - floored*b cancels, so it is computed as _cancel computes a difference:
        # 77401.7 % 1 is 0.7. Never under _RAW: // and % use displayed values (_uses_shown_values).
        da, db = _decimal_value(a), _decimal_value(b)
        if da is not None and db is not None:
            with decimal.localcontext(_EXACT):
                remainder = float(da - decimal.Decimal(floored) * db)
        remainder = _drop_noise(remainder, float(max(abs(a), abs(b)))) or math.copysign(0.0, b)
        return quotient, remainder
    return divmod(_canon(a), _canon(b))


def _floordiv(a: Any, b: Any) -> Any:
    if isinstance(a, float) or isinstance(b, float):
        return _float_divmod(a, b)[0]
    _charge(_div_cost(a.bit_length(), b.bit_length()))
    return a // b


def _mod(a: Any, b: Any) -> Any:
    if isinstance(a, float) or isinstance(b, float):
        return _float_divmod(a, b)[1]
    _charge(_div_cost(a.bit_length(), b.bit_length()))
    return a % b


def _bounded_int(name: str, limit: int) -> Callable[[Any], int]:
    def check(value: Any) -> int:
        value = _shown(value)  # 5.0000000000001 is shown as 5
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not _is_int(value):
            raise ToolError(f"Error: {name} needs integer arguments")
        if abs(value) > limit:
            raise ToolError(f"Error: {name} argument too large (limit {limit})")
        return value

    return check


_FACT_ARG = _bounded_int("factorial", MAX_FACTORIAL_ARG)
_COMB_ARG = _bounded_int("comb/perm", MAX_COMB_ARG)
_ROUND_ARG = _bounded_int("round ndigits", MAX_ROUND_DIGITS)
_GCD_ARG = _bounded_int("gcd", 10**MAX_INT_DIGITS)
_LCM_ARG = _bounded_int("lcm", 10**MAX_INT_DIGITS)


def _factorial(n: Any) -> int:
    n = _FACT_ARG(n)
    if n < 0:
        raise ToolError("Error: factorial needs a non-negative integer")
    _charge(_perm_cost(n, math.lgamma(n + 1) / _LN2))
    return math.factorial(n)


def _log10_falling(n: int, k: int) -> float:
    """log10 of n * (n-1) * ... * (n-k+1), i.e. perm(n, k)."""
    return (math.lgamma(n + 1) - math.lgamma(n - k + 1)) / _LN10


def _nk_args(n: Any, k: Any) -> tuple[int, int]:
    n, k = _COMB_ARG(n), _COMB_ARG(k)
    if n < 0 or k < 0:
        raise ToolError("Error: comb/perm need non-negative integers")
    return n, k


def _comb(n: Any, k: Any) -> int:
    n, k = _nk_args(n, k)
    if k > n:
        return 0
    k = min(k, n - k)
    digits = _log10_falling(n, k) - math.lgamma(k + 1) / _LN10
    if digits > MAX_INT_DIGITS:
        raise _too_large()
    _charge(_comb_cost(k, digits / _LOG10_2))
    return math.comb(n, k)


def _perm(n: Any, k: Any = None) -> int:
    n, k = _nk_args(n, n if k is None else k)
    if k > n:
        return 0
    digits = _log10_falling(n, k)
    if digits > MAX_INT_DIGITS:
        raise _too_large()
    _charge(_perm_cost(k, digits / _LOG10_2))
    return math.perm(n, k)


# gcd and lcm take their numbers from _sequence_args: gcd([12, 18]) == gcd(12, 18), gcd() == 0, lcm() == 1.


def _gcd(numbers: list[Any]) -> int:
    values = [_GCD_ARG(a) for a in numbers]
    result = abs(values[0]) if values else 0
    for value in values[1:]:
        _charge(_gcd_cost(result.bit_length(), value.bit_length()))
        result = math.gcd(result, value)
    return result


def _lcm(numbers: list[Any]) -> int:
    values = [_LCM_ARG(a) for a in numbers] or [1]
    result = values[0]
    for value in values[1:]:
        la, lb = result.bit_length(), value.bit_length()
        if la + lb > _MAX_INT_BITS:
            raise _too_large()
        # math.lcm(a, b) computes a // gcd(a, b) * b, and the gcd has at most min(la, lb) bits
        _charge(_gcd_cost(la, lb) + _div_cost(la, min(lb, (la + 1) // 2)) + _mul_cost(la, lb))
        result = math.lcm(result, value)
    return abs(result)


# --- rounding -------------------------------------------------------------------


def _round(x: Any, ndigits: Any = None) -> Any:
    # Python's round() on the raw value, with no snapping: TRACE computes its
    # answers with it, so exact ties go to the even digit and round(x) of a
    # float from 2**53 up is its exact binary integer. The caller evaluates the
    # arithmetic in x with _RAW set (see _eval_node).
    if ndigits is None:
        return round(x)
    return round(x, _ROUND_ARG(ndigits))


# --- sequences ------------------------------------------------------------------


def _floats(name: str, values: list[Any]) -> list[float]:
    try:
        return [float(v) for v in values]
    except OverflowError:
        raise ToolError(f"Error: {name} values are too large") from None


def _sum(values: list[Any]) -> Any:
    total = sum(values)
    return _cancel(values, total) if isinstance(total, float) else total


def _mean(values: list[Any]) -> float:
    xs = _floats("mean", values)
    return _cancel(xs, math.fsum(xs)) / len(xs)


def _stdev(values: list[Any], *, population: bool) -> float:
    xs = _floats("stdev", values)
    if len(xs) < (1 if population else 2):
        raise ToolError("Error: pstdev needs a value" if population else "Error: stdev needs at least two values")
    mean = _mean(xs)
    deviations = [_cancel([x, -mean], x - mean) for x in xs]
    return math.sqrt(math.fsum(d * d for d in deviations) / (len(xs) - (0 if population else 1)))


def _sequence_args(name: str, args: tuple[Any, ...], *, empty_ok: bool = False) -> list[Any]:
    """Either one list or several numbers: f([1, 2, 3]) or f(1, 2, 3).

    The list was built by a List node, so it already holds at most MAX_SEQUENCE_LEN numbers.
    """
    if any(isinstance(a, list) for a in args):
        if len(args) != 1:
            raise ToolError(
                f"Error: {name} takes one list or several numbers, e.g. {name}([1, 2, 3]) or {name}(1, 2, 3), "
                "not a mix of lists and numbers or several lists"
            )
        values = list(args[0])
    else:
        values = list(args)
    if not values and not empty_ok:
        raise ToolError(f"Error: {name} needs at least one value")
    if not all(_is_number(v) for v in values):
        raise ToolError(f"Error: {name} needs numbers")
    return values


def _seq_fn(name: str, fn: Callable[[list[Any]], Any], *, empty_ok: bool = False) -> Callable[..., Any]:
    def call(*args: Any) -> Any:
        return fn(_sequence_args(name, args, empty_ok=empty_ok))

    return call


# --- real functions -------------------------------------------------------------


def _unit_arg(x: Any) -> Any:
    """An asin/acos argument displayed as +-1 is exactly +-1, on either side of it.

    0.9999999999999998 (10 / (sqrt(5) * sqrt(20))) and 1.0000000000000002 are both
    shown as 1 and compare equal to 1, so acos gives 0 for both. asin/acos are
    ill-conditioned there: one ulp below 1 would otherwise be ~1e-6 degrees.
    Every other argument is used as is, so small arguments keep their precision.
    """
    if isinstance(x, float) and abs(_canon(x)) == 1:
        return math.copysign(1.0, x)
    return x


def _snap_degrees(d: float) -> float:
    """asind(0.5) is 30, not 30.000000000000004. Never snaps to 0: asind(1e-15) keeps its value."""
    r = round(d)
    return float(r) if r and abs(d - r) <= 1e-9 else d


def _unsigned_zero(v: Any) -> Any:
    """-0.0 -> 0.0, so atan2 gives the same angle for float and int coordinates."""
    return 0.0 if _is_number(v) and v == 0 else v


def _asin(x: Any) -> float:
    return math.asin(_unit_arg(x))


def _acos(x: Any) -> float:
    return math.acos(_unit_arg(x))


def _atan2(y: Any, x: Any) -> float:
    return math.atan2(_unsigned_zero(y), _unsigned_zero(x))


_SIN_DEGREES = {0: 0.0, 30: 0.5, 45: math.sqrt(2) / 2, 60: math.sqrt(3) / 2, 90: 1.0}


def _sin_first_quadrant(r: Any) -> float:
    """sin of r degrees, 0 <= r <= 90: exact at 0/30/45/60/90, full relative precision near 0."""
    exact = _SIN_DEGREES.get(r)
    return math.sin(math.radians(r)) if exact is None else exact


def _cos_first_quadrant(r: Any) -> float:
    if r >= 45:
        return _sin_first_quadrant(90 - r)  # exact subtraction (Sterbenz), keeps precision near 90
    if r == 0 or r == 30:
        return _SIN_DEGREES[90 - r]
    return math.cos(math.radians(r))


def _degree_quadrant(d: Any) -> tuple[Any, float, float]:
    """For 0 <= d < 360: (r, sin sign, cos sign) with r in [0, 90]. Every subtraction is exact."""
    if d <= 90:
        return d, 1.0, 1.0
    if d <= 180:
        return 180 - d, 1.0, -1.0
    if d <= 270:
        return d - 180, -1.0, -1.0
    return 360 - d, -1.0, 1.0


def _degrees_arg(x: Any) -> Any:
    """The angle as displayed (_shown): 180 - atand(1/2) - atand(2) is 90.00000000000001 and shows 90.

    So an angle shown as 90 hits the exact table: cosd is 0 and tand is undefined,
    as for a literal 90. _shown keeps 12 significant digits, so tiny angles keep
    their relative precision, and from 2**53 up it is the displayed integer.
    """
    if not _is_number(x):
        raise ToolError("Error: sind/cosd/tand need a number")
    if isinstance(x, float) and math.isinf(x):  # as math.sin(inf); inf % 360 would quietly give nan
        raise ToolError("Error: math domain error")
    return _shown(x)


def _sind(x: Any) -> float:
    x = _degrees_arg(x)
    sign = -1.0 if x < 0 else 1.0
    r, s, _ = _degree_quadrant(abs(x) % 360)  # abs first: -1e-20 % 360 would round to 360.0
    return sign * s * _sin_first_quadrant(r) + 0.0  # + 0.0 turns -0.0 into 0.0


def _cosd(x: Any) -> float:
    r, _, c = _degree_quadrant(abs(_degrees_arg(x)) % 360)
    return c * _cos_first_quadrant(r) + 0.0


def _tand(x: float) -> float:
    s, c = _sind(x), _cosd(x)
    if c == 0.0:
        raise ToolError("Error: tan is undefined at this angle")
    return s / c


def _asind(x: Any) -> float:
    return _snap_degrees(math.degrees(_asin(x)))


def _acosd(x: Any) -> float:
    return _snap_degrees(math.degrees(_acos(x)))


def _atand(x: float) -> float:
    return _snap_degrees(math.degrees(math.atan(x)))


def _atan2d(y: Any, x: Any) -> float:
    return _snap_degrees(math.degrees(_atan2(y, x)))


def _log(x: float, base: float | None = None) -> float:
    if base is None:
        return math.log(x)
    if base == 10:
        return math.log10(x)
    if base == 2:
        return math.log2(x)
    return math.log(x, base)


def _cbrt(x: float) -> float:
    if hasattr(math, "cbrt"):  # Python 3.11+, exact for perfect cubes
        return math.cbrt(x)
    c = math.copysign(abs(x) ** (1.0 / 3.0), x)
    r = round(c)
    return float(r) if r**3 == x else c


# On the displayed value; from 2**53 up that is already the displayed integer (_shown).


def _floor(x: Any) -> int:
    return math.floor(_shown(x))


def _ceil(x: Any) -> int:
    return math.ceil(_shown(x))


def _trunc(x: Any) -> int:
    return math.trunc(_shown(x))


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": _round,
    "floor": _floor,
    "ceil": _ceil,
    "trunc": _trunc,
    "sqrt": math.sqrt,
    "cbrt": _cbrt,
    "exp": math.exp,
    "log": _log,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "pow": _safe_pow,
    "hypot": _seq_fn("hypot", lambda v: math.hypot(*v)),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": _asin,
    "acos": _acos,
    "atan": math.atan,
    "atan2": _atan2,
    "sind": _sind,
    "cosd": _cosd,
    "tand": _tand,
    "asind": _asind,
    "acosd": _acosd,
    "atand": _atand,
    "atan2d": _atan2d,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": _factorial,
    "comb": _comb,
    "perm": _perm,
    "gcd": _seq_fn("gcd", _gcd, empty_ok=True),
    "lcm": _seq_fn("lcm", _lcm, empty_ok=True),
    "min": _seq_fn("min", min),
    "max": _seq_fn("max", max),
    "sum": _seq_fn("sum", _sum),
    "mean": _seq_fn("mean", _mean),
    "median": _seq_fn("median", statistics.median),
    "stdev": _seq_fn("stdev", lambda v: _stdev(v, population=False)),
    "pstdev": _seq_fn("pstdev", lambda v: _stdev(v, population=True)),
}

# (min, max) number of arguments, checked before a call so that a wrong count is
# reported with the name the model used, not an internal one such as <lambda>.
# None: any number (the sequence functions report an empty call themselves).
_ARITY: dict[str, tuple[int, int | None]] = {
    **dict.fromkeys(
        ["abs", "floor", "ceil", "trunc", "sqrt", "cbrt", "exp", "ln", "log10", "log2", "sin", "cos", "tan"],
        (1, 1),
    ),
    **dict.fromkeys(
        ["asin", "acos", "atan", "sind", "cosd", "tand", "asind", "acosd", "atand", "degrees", "radians", "factorial"],
        (1, 1),
    ),
    **dict.fromkeys(["round", "log", "perm"], (1, 2)),
    **dict.fromkeys(["pow", "atan2", "atan2d", "comb"], (2, 2)),
    "hypot": (1, None),
    **dict.fromkeys(["gcd", "lcm", "min", "max", "sum", "mean", "median", "stdev", "pstdev"], (0, None)),
}
if _ARITY.keys() != _FUNCTIONS.keys():  # a KeyError in _eval_node would escape the ToolError contract
    raise RuntimeError(f"calculator: no argument count for {sorted(_ARITY.keys() ^ _FUNCTIONS.keys())}")


def _arity_error(name: str, given: int) -> ToolError:
    low, high = _ARITY[name]
    if high == low:
        wanted = f"{low} argument{'' if low == 1 else 's'}"
    elif high is None:
        wanted = f"at least {low} argument{'' if low == 1 else 's'}"
    else:
        wanted = f"{low} or {high} arguments"
    hint = f"; for n decimals use {name}(x * 10**n) / 10**n" if name in ("floor", "ceil", "trunc") else ""
    return ToolError(f"Error: {name} takes {wanted}, got {given}{hint}")


_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: _add,
    ast.Sub: _sub,
    ast.Mult: _safe_mult,
    ast.Div: operator.truediv,
    ast.FloorDiv: _floordiv,
    ast.Mod: _mod,
    ast.Pow: _safe_pow,
    ast.LShift: _safe_lshift,
}

_UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# --- comparisons ----------------------------------------------------------------


# Every comparison is on the displayed values (_shown): ints exactly, floats at
# 12 (15) significant digits, so 0.1 + 0.2 == 0.3, 1e-19 < 1.6e-19 and
# 6.022e23 == 602200000000000000000000.


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
            and all(_equal(x, y) for x, y in zip(a, b))
        )
    return _shown(a) == _shown(b)


def _ordering(compare: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
    def check(a: Any, b: Any) -> bool:
        if isinstance(a, list) or isinstance(b, list):
            raise ToolError("Error: lists can only be compared with == or !=")
        return compare(_shown(a), _shown(b))

    return check


_CMPOPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Lt: _ordering(operator.lt),
    ast.LtE: _ordering(operator.le),
    ast.Gt: _ordering(operator.gt),
    ast.GtE: _ordering(operator.ge),
    ast.Eq: _equal,
    ast.NotEq: lambda a, b: not _equal(a, b),
}


# --- evaluation -----------------------------------------------------------------


# Operations on displayed values: inside round()'s argument they still see
# cancelled values, so wrapping an expression in round() never changes them.
_SHOWN_VALUE_FUNCTIONS = frozenset({"floor", "ceil", "trunc", "factorial", "comb", "perm", "gcd", "lcm"})


def _uses_shown_values(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, (ast.FloorDiv, ast.Mod))
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SHOWN_VALUE_FUNCTIONS


def _eval(node: ast.AST) -> Any:
    if _RAW.get() and _uses_shown_values(node):
        token = _RAW.set(False)
        try:
            return _checked(_eval_node(node))
        finally:
            _RAW.reset(token)
    return _checked(_eval_node(node))


_IMPLICIT_MULTIPLICATION = (
    "Error: implicit multiplication is not supported; write an explicit *, "
    "e.g. 2*(3+4), (5+3)*(2), 2*pi or 2*sqrt(3)"
)
_KEYWORD_ARGUMENTS = (
    "Error: keyword arguments are not supported; pass arguments by position, e.g. round(x, 2) or log(x, 2)"
)
_MAX_HINT_NODES = 30  # larger receivers get a generic example (also bounds ast.unparse's recursion)
_MAX_HINT_CHARS = 120


def _method_call_error(node: ast.Call, func: ast.Attribute) -> ToolError:
    """A numpy/pandas-style method call such as [3, 5, 7].mean() or max(a, b).round(1); func is node.func.

    The pre-parse check only lets an attribute through when it is a function or
    constant name, and a receiver that is an unknown name is reported there as a
    module prefix; what is left is a list, literal, bracket or call before the dot.
    """
    name = func.attr
    if name not in _FUNCTIONS:  # [1, 2].pi()
        example = "write mean([3, 5, 7]), not [3, 5, 7].mean()"
    else:
        example = f"write {name}(x, ...), not x.{name}(...)"
        if sum(1 for _ in ast.walk(node)) <= _MAX_HINT_NODES:  # ast.walk is iterative
            call = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[func.value, *node.args], keywords=[])
            try:
                written, method = ast.unparse(call), ast.unparse(node)
            except (ValueError, RecursionError):  # a long int literal past a lowered int->str limit
                written = method = ""
            if written and len(written) + len(method) <= _MAX_HINT_CHARS:
                example = f"write {written}, not {method}"
    return ToolError(f"Error: method calls are not supported; pass the value as the first argument: {example}")


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if _is_number(node.value):
            return node.value
        raise ToolError(f"Error: unsupported literal {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ToolError(f"Error: unsupported operator {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if not (_is_number(left) and _is_number(right)):
            raise ToolError("Error: arithmetic operands must be numbers")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ToolError(f"Error: unsupported operator {type(node.op).__name__}")
        operand = _eval(node.operand)
        if not _is_number(operand):
            raise ToolError("Error: arithmetic operands must be numbers")
        return op(operand)
    if isinstance(node, ast.Compare):
        left = _eval(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMPOPS.get(type(op_node))
            if op is None:
                raise ToolError(f"Error: unsupported comparison {type(op_node).__name__}")
            right = _eval(comparator)
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > MAX_SEQUENCE_LEN:
            raise ToolError("Error: list is too long")
        values = [_eval(elt) for elt in node.elts]
        if not all(_is_number(v) for v in values):
            raise ToolError("Error: lists may only contain numbers")
        return values
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"Error: unknown name '{node.id}'")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):  # [3, 5, 7].mean(), max(a, b).round(1)
            raise _method_call_error(node, node.func)
        if not isinstance(node.func, ast.Name):  # (5+3)(2), sqrt(2)(3), [1, 2](3): never a function name
            raise ToolError(_IMPLICIT_MULTIPLICATION)
        name = node.func.id
        if name in _CONSTANTS:  # pi(2)
            raise ToolError(
                f"Error: {name} is a constant, not a function; to multiply write an explicit *, e.g. {name}*(2)"
            )
        if name not in _FUNCTIONS:
            raise ToolError(f"Error: unknown function '{name}'. Available: {', '.join(sorted(_FUNCTIONS))}")
        if node.keywords:  # a keyword spelled like a known name (round(2.5, e=1)); others fail in _validate_text
            raise ToolError(_KEYWORD_ARGUMENTS)
        low, high = _ARITY[name]
        if len(node.args) < low or (high is not None and len(node.args) > high):
            raise _arity_error(name, len(node.args))
        if name == "round":  # round() sees Python's raw floats (see _RAW)
            token = _RAW.set(True)
            try:
                args = [_eval(node.args[0])]
            finally:
                _RAW.reset(token)
            token = _RAW.set(False)  # ndigits is an integer argument: always the displayed value
            try:
                args += [_eval(arg) for arg in node.args[1:]]
            finally:
                _RAW.reset(token)
        else:
            args = [_eval(arg) for arg in node.args]
        return _FUNCTIONS[name](*args)
    raise ToolError(f"Error: unsupported syntax {type(node).__name__}")


_ALLOWED_CHARS = frozenset(string.ascii_letters + string.digits + "_ \t\n.,+-*/%()[]<>=!")
_DIGITS = r"[0-9](?:_?[0-9])*"
# Python numeric literals (no imaginary suffix), so the letters inside 1e5 or
# 0x1F are not mistaken for identifiers while `1if` still yields `if`.
_NUMBER = (
    r"0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+"
    rf"|(?:{_DIGITS}(?:\.(?:{_DIGITS})?)?|\.{_DIGITS})(?:[eE][-+]?{_DIGITS})?"
)
_TOKEN_RE = re.compile(rf"(?P<number>{_NUMBER})|(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
# After the identifier check only whitelisted names are left, and none of them is
# a keyword, so a number followed by a name or "(" (2pi, 2 sqrt(3), 2(3+4)) and a
# ")" followed by a number or name ((3+4)2, (a+b) pi) are never valid Python:
# they are implicit multiplication. ")(" is left to _eval_node, as "(sqrt)(2)" is a call.
_AFTER_NUMBER_RE = re.compile(r"[ \t\n]*[A-Za-z_(]")
_AFTER_PAREN_RE = re.compile(r"\)[ \t\n]*(?:[0-9A-Za-z_]|\.[0-9])")
# % followed by a sign, an operator, a closing bracket or the end: "80 * 15% + 80"
# is (80*15) % (+80) == 0 in Python, a silent wrong answer, and the other forms
# are syntax errors. A remainder by a negative number is written x % (-3).
_PERCENT_RE = re.compile(r"%(?=[ \t\n]*(?:[-+*/%)\],<>=!]|$))")
_PERCENT_HINT = (
    "Error: % is the remainder (modulo), not percent; write 15% of 80 as 0.15*80 or 80*15/100 "
    "(for a remainder by a negative number write 7 % (-3))"
)
# An unknown name followed by "." and a name is a module prefix: math.sqrt(2), np.mean([1, 2]), math.pi.
_MODULE_ATTRIBUTE_RE = re.compile(r"[ \t\n]*\.[ \t\n]*([A-Za-z_][A-Za-z0-9_]*)")
# 1-3 digits, then comma-digit-digit-digit groups, with no space after any comma: 1,200, 2,500,000 or
# 1,200.50. Inside a list or call Python reads it as several items (max(1,200, 950) is 950, not 1200),
# so it is rejected everywhere, also in a compact list such as [100,200,300] or a call such as
# pow(2,100), which need a space after each comma. The leading group may not continue a name, a
# longer number or a decimal (x2,500 / 1500,200 / 0.5,100 are not matched).
_THOUSANDS_RE = re.compile(r"(?<![\w.])[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])")
# An unknown name right after "(" or "," and followed by a single "=" is a keyword argument:
# round(3.14159, ndigits=2), log(8, base=2). "name == x" is a comparison and is not matched.
_KEYWORD_EQUALS_RE = re.compile(r"[ \t\n]*=(?!=)")
# "!" is never valid on its own (only in "!="), so 5! and 6!/(2!*4!) are factorial notation.
# "!=" directly after a digit or ")" is rejected too: Python reads 4!=24 as 4 != 24 and silently
# returns True, while factorial notation means factorial(4) == 24. An inequality needs a space: 4 != 24.
_BANG_RE = re.compile(r"!(?!=)|(?<=[0-9)])!=")
_NUMBER_AFTER_RE = re.compile(r"[ \t\n]*([0-9]{1,15}(?:\.[0-9]{1,15})?)(?![\w.])")


def _int_before(text: str, end: int) -> str | None:
    """The integer literal of at most 6 digits that ends at text[end], if it is a whole literal."""
    start = end
    while start > 0 and end - start <= 6 and text[start - 1] in string.digits:
        start -= 1
    if start == end or end - start > 6 or (start > 0 and (text[start - 1].isalnum() or text[start - 1] in "._")):
        return None
    return text[start:end]


def _factorial_error(text: str, bang: int) -> ToolError:
    n = _int_before(text, bang)
    if text.startswith("!=", bang):  # 4!=24: factorial check or compact inequality
        after = _NUMBER_AFTER_RE.match(text, bang + 2)
        m = after.group(1) if after is not None else None
        if n is not None and m is not None:
            return ToolError(
                f"Error: {n}!={m} is ambiguous because ! is not factorial here: write factorial({n}) == {m} "
                f"to check a factorial, or {n} != {m} (with a space before !=) for an inequality"
            )
        return ToolError(
            "Error: != right after a number or ) is ambiguous because ! is not factorial here: write "
            "factorial(n) == m to check a factorial, or a != b (with a space before !=) for an inequality"
        )
    n = n or "n"
    return ToolError(
        f"Error: ! is not factorial here; write factorial({n}), not {n}!, and comb(n, k) for n!/(k!*(n-k)!) "
        "or perm(n, k) for n!/(n-k)!"
    )


def _module_prefix_error(module: str, attribute: str) -> ToolError:
    module, attribute = module[:40], attribute[:40]
    if attribute in _FUNCTIONS:
        example = f"write {attribute}(...), not {module}.{attribute}(...)"
    elif attribute in _CONSTANTS:
        example = f"write {attribute}, not {module}.{attribute}"
    else:
        example = f"write sqrt(2) or pi, not {module}.sqrt(2) or {module}.pi"
    return ToolError(f"Error: module prefixes are not supported; call functions and constants by name alone: {example}")


def _thousands_error(number: str) -> ToolError:
    number = number[:40]  # digits and commas only
    return ToolError(
        f"Error: {number} is ambiguous because commas are not thousands separators: write "
        f"{number.replace(',', '')} for one number, or {number.replace(',', ', ')} (a space after each comma) "
        "for separate values"
    )


def _validate_text(text: str) -> None:
    """First layer, before ast.parse: ASCII whitelist and known identifiers only.

    It also turns module prefixes, keyword arguments, factorial notation,
    implicit multiplication, percent signs and thousands separators into errors
    with a hint, instead of a bare syntax error, a misleading "unknown name", a
    silently wrong remainder or inequality, or silently extra list items.
    """
    for ch in text:
        if ch not in _ALLOWED_CHARS:
            shown = repr(ch) if ch.isprintable() else ascii(ch)
            raise ToolError(
                f"Error: unsupported character {shown}; use only ASCII digits, operators, "
                "brackets and the listed function names"
            )
    for match in _TOKEN_RE.finditer(text):
        name = match.group("name")
        if name is None or name in _FUNCTIONS or name in _CONSTANTS:
            continue
        attribute = _MODULE_ATTRIBUTE_RE.match(text, match.end())
        if attribute is not None:
            raise _module_prefix_error(name, attribute.group(1))
        if _KEYWORD_EQUALS_RE.match(text, match.end()) and text[: match.start()].rstrip(" \t\n").endswith(("(", ",")):
            raise ToolError(_KEYWORD_ARGUMENTS)
        if text[match.end() :].lstrip().startswith("("):
            raise ToolError(f"Error: unknown function '{name[:40]}'. Available: {', '.join(sorted(_FUNCTIONS))}")
        raise ToolError(f"Error: unknown name '{name[:40]}'; the constants are {', '.join(sorted(_CONSTANTS))}")
    bang = _BANG_RE.search(text)
    if bang is not None:
        raise _factorial_error(text, bang.start())
    for match in _TOKEN_RE.finditer(text):
        if match.group("number") is not None and _AFTER_NUMBER_RE.match(text, match.end()):
            raise ToolError(_IMPLICIT_MULTIPLICATION)
    if _AFTER_PAREN_RE.search(text):
        raise ToolError(_IMPLICIT_MULTIPLICATION)
    if _PERCENT_RE.search(text):
        raise ToolError(_PERCENT_HINT)
    thousands = _THOUSANDS_RE.search(text)
    if thousands is not None:
        raise _thousands_error(thousands.group())


def evaluate(expression: str) -> Any:
    """Evaluate one arithmetic expression; raises ToolError on anything else."""
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("Error: expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError(f"Error: expression longer than {MAX_EXPRESSION_CHARS} characters")
    text = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    _validate_text(text)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Error: invalid expression ({exc.msg})") from None
    except (ValueError, RecursionError, MemoryError):  # absurd nesting
        raise ToolError("Error: invalid expression (unsupported characters or nesting)") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ToolError("Error: expression is too complex")
    if isinstance(tree.body, ast.Tuple):
        raise ToolError(
            "Error: a bare comma-separated sequence is not one expression; commas are not thousands "
            "separators (write 1234567) and lists need brackets, e.g. mean([1, 2, 3])"
        )
    token = _WORK.set([MAX_WORK_US])
    try:
        return _eval(tree)
    except ToolError:
        raise
    except ZeroDivisionError:
        raise ToolError("Error: division by zero") from None
    except (ValueError, TypeError, OverflowError, RecursionError, MemoryError, statistics.StatisticsError) as exc:
        raise ToolError(f"Error: {exc}") from None
    finally:
        _WORK.reset(token)


def _too_long() -> ToolError:
    return ToolError(f"Error: result is longer than {MAX_RESULT_CHARS} characters")


def format_result(value: Any) -> str:
    pieces: list[str] = []
    room = MAX_RESULT_CHARS

    def add(text: str) -> None:
        nonlocal room
        room -= len(text)
        if room < 0:
            raise _too_long()
        pieces.append(text)

    def walk(v: Any) -> None:
        if isinstance(v, bool):
            add(str(v))
        elif isinstance(v, int):
            if _int_too_large(v):
                raise _too_large()
            # refuse before the quadratic int -> str conversion; the bound undercounts by at most one
            if int((v.bit_length() - 1) * _LOG10_2) > room:
                raise _too_long()
            add(str(v))
        elif isinstance(v, float):
            if not math.isfinite(v):
                raise ToolError("Error: result is not finite")
            add(_float_text(v))
        elif isinstance(v, list):
            add("[")
            for i, item in enumerate(v):
                if i:
                    add(", ")
                walk(item)
            add("]")
        else:
            raise ToolError("Error: unsupported result type")

    try:
        walk(value)
    except ValueError:  # int -> str beyond sys.get_int_max_str_digits(), if a host lowered it
        raise _too_large() from None
    except RecursionError:  # deeply nested lists passed in by a caller other than evaluate()
        raise ToolError("Error: unsupported result type") from None
    return "".join(pieces)


class CalculatorTool(Tool):
    name = "calculator"
    # Kept short: it is part of every prompt in the calculator/all conditions. The
    # finer rules (module docstring) reach the model through targeted error hints.
    description = (
        "Evaluate an arithmetic expression and return the result (floats shown with 12 significant digits; "
        "comparisons, floor/ceil/trunc, // and % use the value as shown). Supports + - * / // % ** (or ^), "
        "parentheses, comparisons, lists like [1, 2, 3], the constants pi, e, tau, and the functions "
        + ", ".join(sorted(_FUNCTIONS))
        + ". Write every multiplication with * (2*pi, not 2pi) and call functions by name (sqrt(2), not "
        "math.sqrt(2)). % is the remainder, not percent. log and ln are natural logs. sin/cos/tan use radians; "
        "sind/cosd/tand/asind/acosd/atand/atan2d use degrees and are exact at special angles. "
        "round is Python's round: exact ties go to the even digit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression to evaluate, e.g. \"(12.5 + 7) * 3\" or \"sqrt(3**2 + 4**2)\"",
            },
        },
        "required": ["expression"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        expression = args.get("expression")
        value = evaluate(expression)
        return ToolResult(text=format_result(value), meta={"expression": expression})
