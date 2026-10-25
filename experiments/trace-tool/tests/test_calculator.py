import ast
import math
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

from trace_tool.tools import calculator
from trace_tool.tools.base import ToolContext, ToolError
from trace_tool.tools.calculator import CalculatorTool, evaluate, format_result


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1 + 2 * 3", 7),
        ("(12.5 + 7) * 3", 58.5),
        ("2 ^ 10", 1024),
        ("17 // 5", 3),
        ("17 % 5", 2),
        ("sqrt(3**2 + 4**2)", 5.0),
        ("hypot(3, 4)", 5.0),
        ("round(2/3, 3)", 0.667),
        ("sind(30)", pytest.approx(0.5)),
        ("degrees(atan2(1, 1))", pytest.approx(45.0)),
        ("max(3, 9, 4)", 9),
        ("mean([1, 2, 3, 4])", 2.5),
        ("median([5, 1, 3])", 3),
        ("factorial(5)", 120),
        ("comb(6, 2)", 15),
        ("pi * 2", math.tau),
        ("log(8, 2)", pytest.approx(3.0)),
        ("3 > 2", True),
        ("-(-4)", 4),
        ("1.5e3 / 3", 500.0),
    ],
)
def test_evaluates(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd').read()",
        "exec('1')",
        "eval('1+1')",
        "lambda: 1",
        "[x for x in range(10)]",
        "math.sqrt(4)",
        "sqrt.__globals__",
        "'a' * 10",
        "x + 1",
        "9 ** 9 ** 9",
        "10 ** 100000",
        "factorial(100000)",
        "comb(100000, 50000)",
        "2 << 100000",
        "1 / 0",
        "sqrt(-1)",
        "(-8) ** 0.5",
        "1" * 2000,
        "max()",
        "round(1.5, ndigits=1)",
    ],
)
def test_rejects(expr):
    with pytest.raises(ToolError):
        format_result(evaluate(expr))


def test_tool_wrapper_returns_text():
    tool = CalculatorTool()
    ctx = ToolContext([])
    assert tool({"expression": "0.1 + 0.2"}, ctx).text == "0.3"
    assert tool({"expression": "10 / 4"}, ctx).text == "2.5"
    bad = tool({"expression": "import os"}, ctx)
    assert not bad.ok and bad.text.startswith("Error")
    assert not tool({}, ctx).ok


@pytest.mark.parametrize(
    "expr",
    [
        "round(7, -10**7)",
        "round(7, -10**8)",
        "round(7, -1e300)",
        "perm(100000)",
        "perm(100000, 50000)",
        "comb(100000, 50000)",
        "lcm(perm(100000)+1, perm(99999)+1, perm(99998)+1, perm(99997)+1, perm(99996)+1, perm(99995)+1)",
        "lcm(10**3000+1, 10**3000+3)",
        "factorial(1000) * factorial(1000)",
        "factorial(1000) << 10000",
        "sum([10**3999] * 1000) * 10",
        "stdev([10**3999, 1])",
        "mean([10**3999, 1])",
    ],
)
def test_expensive_inputs_fail_fast(expr):
    """These froze the event loop (GIL held in C) before the size invariant."""
    import time

    started = time.thread_time()
    with pytest.raises(ToolError):
        format_result(evaluate(expr))
    assert time.thread_time() - started < 0.5, expr


def test_bounded_functions_still_work():
    assert evaluate("comb(100000, 3)") == 166661666700000
    assert evaluate("perm(10, 3)") == 720
    assert evaluate("perm(5)") == 120
    assert evaluate("comb(3, 5)") == 0
    assert evaluate("lcm(4, 6, 10)") == 60
    assert evaluate("round(1234, -2)") == 1200
    assert evaluate("stdev([2, 4, 4, 4, 5, 5, 7, 9])") == pytest.approx(2.138089935)
    assert evaluate("pstdev([2, 4, 4, 4, 5, 5, 7, 9])") == pytest.approx(2.0)
    assert evaluate("mean([1, 2])") == 1.5


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("(10**400)**2 == 10**800", "True"),
        ("factorial(200)**2 == factorial(200) * factorial(200)", "True"),
        ("cbrt(64)", "4"),
        ("floor(cbrt(64))", "4"),
        ("floor(cbrt(1000))", "10"),
        ("ceil(cbrt(-64))", "-4"),
        ("floor(log(1000, 10))", "3"),
        ("cosd(90)", "0"),
        ("10*cosd(90)", "0"),
        ("sind(180)", "0"),
        ("floor(10*sind(30))", "5"),
        ("floor(tand(45))", "1"),
        ("floor(tand(60)**2)", "3"),
        ("sind(30) == 0.5", "True"),
        ("asind(0.5)", "30"),
        ("0.1 + 0.2 == 0.3", "True"),
        ("0.1 + 0.2 > 0.3", "False"),
        ("3 * 1.2", "3.6"),
        ("0.1 * 3 - 0.3", "0"),
        ("2**53 + 0.0", "9.00719925474099e+15"),  # 15 significant digits from 1e10 up
        ("123456789012.5", "123456789012.5"),
        ("round(2.5)", "2"),
        ("round(1234, -2)", "1200"),
    ],
)
def test_display_precision_semantics(expr, expected):
    assert CalculatorTool()({"expression": expr}, ToolContext([])).text == expected


@pytest.mark.parametrize("expr", ["tand(90)", "tand(-270)", "1,234", "1, 2, 3", "1+\ud800", "1\x00"])
def test_more_rejects(expr):
    with pytest.raises(ToolError):
        format_result(evaluate(expr))


# --- regression tests for the review findings (numbered as in the review) -------------


def _text(expr):
    return CalculatorTool()({"expression": expr}, ToolContext([])).text


def _fill(unit, sep="+", prefix="", suffix=""):
    """Longest prefix + unit sep unit ... + suffix within the length and AST-node limits."""

    def fits(candidate):
        nodes = sum(1 for _ in ast.walk(ast.parse(candidate.replace("^", "**"), mode="eval")))
        return len(candidate) <= calculator.MAX_EXPRESSION_CHARS and nodes <= calculator.MAX_AST_NODES

    parts = [unit]
    while fits(prefix + sep.join(parts + [unit]) + suffix):
        parts.append(unit)
    return prefix + sep.join(parts) + suffix


# 1. CPU budget


@pytest.mark.parametrize(
    "unit, sep, prefix, suffix",
    [
        ("comb(13290,6645)", "+", "", ""),
        ("comb(8000,4000)%2", "+", "", ""),
        ("comb(100000,1850)%2", "+", "", ""),
        ("perm(1450)%2", "+", "", ""),
        ("factorial(1000)%2", "+", "", ""),
        ("gcd(2**13286-1,3**8382-1)", "+", "", ""),
        ("2**13286-1", ",", "gcd(3**8382-1,", ")"),
        ("lcm(2**6643-1,3**4191-1)%2", "+", "", ""),
        ("7", ",", "lcm(3**4191,", ")"),
        ("3**8382%2", "+", "", ""),
        ("1**(10**3999)", "+", "", ""),
        ("3**8382//3**4191%2", "+", "", ""),
        ("3**8382%(2**6643-1)%2", "+", "", ""),
        ("3**4191*3**4191%2", "+", "", ""),
        ("(1<<13286)%2", "+", "", ""),
        ("9**4190", ",", "[", "]"),
        ("3**8382", ",", "median([", "])"),
        ("sind(3**8382)", "+", "", ""),
    ],
)
def test_worst_case_expressions_finish_well_under_budget(unit, sep, prefix, suffix):
    expr = _fill(unit, sep, prefix, suffix)
    started = time.thread_time()
    try:
        format_result(evaluate(expr))
    except ToolError:
        pass
    assert time.thread_time() - started < 0.2, expr[:60]


def test_work_budget_stops_repeated_expensive_calls_but_allows_single_ones():
    assert evaluate("comb(13290, 6645) % 1000") == math.comb(13290, 6645) % 1000
    with pytest.raises(ToolError, match="too much computation"):
        evaluate("comb(13290, 6645) % 2 + comb(13290, 6644) % 2 + comb(13290, 6643) % 2")
    assert evaluate("comb(100000, 3)") == 166661666700000
    assert evaluate("factorial(1000)") == math.factorial(1000)
    assert evaluate("2**10000") == 2**10000
    assert evaluate("2^10000 % 7") == 2


def test_work_budget_is_per_evaluation_and_per_thread():
    one = calculator._perm_cost(1450, calculator._log10_falling(1450, 1450) / calculator._LOG10_2)
    count = int(0.6 * calculator.MAX_WORK_US / one)  # about 60% of one budget
    expr = "+".join(["perm(1450)%2"] * count)
    assert evaluate(expr) == 0
    assert evaluate(expr) == 0  # a fresh budget for every call
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(evaluate, [expr] * 12)) == [0] * 12
    assert evaluate("comb(13290, 6645) % 2") == 0
    with pytest.raises(ToolError, match="too much computation"):  # the costs add up within one call
        evaluate(expr + "+comb(13290, 6645) % 2")


# 2. Pre-parse validation


@pytest.mark.parametrize(
    "expr",
    [
        "1 if 1 else 2",
        "1if 1else 2",
        "0x1for 1",
        "1 is 1",
        "not 1",
        "True + 1",
        "None",
        "abc_def + 1",
        "sqrt(4).real",
        "1 .real",
        "'1'",
        '"1"',
        "b'1'",
        "f'{1}'",
        "1 # comment",
        "1 \\\n + 2",
        "~1",
        "1 & 1",
        "1 | 1",
        "{1: 2}",
        "1 @ 1",
        "(x := 1)",
        "√2",
        "π",
        "1\r+1",
        "1j",
        "1_",
    ],
)
def test_pre_parse_validation_rejects(expr):
    with pytest.raises(ToolError):
        evaluate(expr)


def test_pre_parse_validation_runs_before_the_parser(monkeypatch):
    class NoParser:
        def __getattr__(self, name):
            raise AssertionError("ast was used for an expression the text checks should reject")

    monkeypatch.setattr(calculator, "ast", NoParser())
    for expr in ["model_chosen_name_123 + 1", "1if 1else 2", "'abc'", "1+\ud800", "1\x00", "sqrt(4).imag"]:
        with pytest.raises(ToolError):
            evaluate(expr)


def test_pre_parse_validation_avoids_parser_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for expr in ["1if 1else 2", "0x1for 1", "'\\d'", "1 is 1", "2pi", "1e", "0b1e"]:
            with pytest.raises(ToolError):
                evaluate(expr)
    assert not caught


def test_pre_parse_validation_messages_and_numeric_literals():
    assert "unknown function 'foo'" in _text("foo(1)") and "sqrt" in _text("foo(1)")
    assert "unknown name 'x'" in _text("x + 1")
    assert "'\\x00'" in _text("1\x00") and "'√'" in _text("√2")
    literals = [("1e5", "100000"), ("2.5E-1", "0.25"), ("0x1F + 0b11 + 0o7", "41"), ("1_000 * e / e", "1000")]
    for expr, expected in literals + [(".5 + 1.", "1.5")]:
        assert _text(expr) == expected, expr


# 3. Float // and %


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("0.6 // 0.2", 3),
        ("0.6 % 0.2", 0),
        ("0.3 // 0.1", 3),
        ("1 % 0.1", 0),
        ("-0.6 // 0.2", -3),
        ("-0.6 // -0.2", 3),
        ("0.6 // -0.2", -3),
        ("7.5 // 2", 3),
        ("7.5 % 2", 1.5),
        ("-7.5 // 2", -4),
        ("-7.5 % 2", 0.5),
        ("1e20 % 3.0", 1),
        ("17 // 5", 3),
        ("-17 % 5", 3),
    ],
)
def test_float_floordiv_and_mod_snap(expr, expected):
    assert evaluate(expr) == expected


def test_float_mod_keeps_python_sign_convention():
    # a negative divisor needs parentheses: "% -" is rejected as a likely percent sign (T5)
    assert math.copysign(1.0, evaluate("0.6 % (-0.2)")) == -1.0
    assert math.copysign(1.0, evaluate("-0.6 % 0.2")) == 1.0
    assert evaluate("0.7 % 0.2") == pytest.approx(0.1)
    with pytest.raises(ToolError):
        evaluate("1.5 % 0")


# 4. Display


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1000.3 - 1000.1", "0.2"),
        ("1/3", "0.333333333333"),
        ("2/3 * 1e9", "666666666.667"),
        ("123456789012.5", "123456789012.5"),
        ("12345678901.23456", "12345678901.2346"),
        ("2**53 + 0.0", "9.00719925474099e+15"),
        ("1e15 + 0.0", "1e+15"),
        ("999999999999999.0", "999999999999999"),
        ("1e-13", "1e-13"),  # no absolute threshold: tiny values keep their digits
        ("-1e-13", "-1e-13"),
        ("1e-11", "1e-11"),
        ("-0.0", "0"),
    ],
)
def test_display_precision(expr, expected):
    assert _text(expr) == expected


def test_result_length_is_capped_before_int_to_text():
    assert format_result(10**1999) == "1" + "0" * 1999
    assert format_result([1, 2.5]) == "[1, 2.5]"
    for value in [-(10**1999), [10**1000, 10**1000], [9**4190] * 100, math.factorial(1000)]:
        with pytest.raises(ToolError, match="longer than 2000"):
            format_result(value)
    started = time.thread_time()
    assert _text(_fill("9**4190", ",", "[", "]")).startswith("Error")
    assert time.thread_time() - started < 0.2


# 5. asin/acos snap their argument


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("asin(1.0000000000000002)", math.pi / 2),
        ("acos(-1.0000000000000002)", math.pi),
        ("asind(1.0000000000000002)", 90),
        ("acosd(-1.0000000000000002)", 180),
        ("asind(0.5000000000001)", 30),
        ("acosd(sqrt(2)/2 + 1e-13)", 45),
        ("acosd(sqrt(3)/2 - 1e-13)", 30),
        ("asin(-1e-13)", -1e-13),  # inside [-1, 1] the argument is used as is
        ("asin(1.000000000001)", math.pi / 2),  # displayed as 1
    ],
)
def test_inverse_trig_snaps_argument(expr, expected):
    assert evaluate(expr) == expected


def test_inverse_trig_still_rejects_real_domain_errors():
    with pytest.raises(ToolError):
        evaluate("asin(1.000001)")
    with pytest.raises(ToolError):
        evaluate("asin(1.00000000001)")  # displayed as 1.00000000001


# 6. Tolerances: floor/ceil and comparisons use the displayed value


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("ceil(5e11 + 0.4)", "500000000001"),
        ("1e9 + 0.99999", "1000000001"),  # 12 significant digits below 1e10 ...
        ("floor(1e9 + 0.99999)", "1000000001"),  # ... so floor() agrees with the display
        ("floor(1e10 + 0.9999)", "10000000000"),  # 15 digits from 1e10 up
        ("floor(3 * 1.1 / 1.1)", "3"),
        ("5e11 + 0.4 == 5e11", "False"),
        ("0.1 + 0.2 == 0.3", "True"),
        ("10**400 == 1.5", "False"),
        ("10**400 != 0.5", "True"),
        ("10**400 >= 1e308", "True"),
        ("1.5 < 10**400", "True"),
    ],
)
def test_tolerances_follow_the_display_and_big_int_comparisons(expr, expected):
    assert _text(expr) == expected


# 7. Sequence functions


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("max(7)", 7),
        ("min([4, 2, 8])", 2),
        ("sum(1, 2, 3)", 6),
        ("mean(1, 2)", 1.5),
        ("median(5, 1, 3)", 3),
        ("pstdev(1, 1)", 0),
        ("stdev(2, 4, 4, 4, 5, 5, 7, 9)", pytest.approx(2.138089935)),
    ],
)
def test_sequence_functions_take_one_list_or_numbers(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize(
    "expr", ["max([1, 2], [3])", "max([1, 2], 3)", "sum([1], [2])", "mean(1, [2])", "sum()", "median([])"]
)
def test_sequence_functions_reject_mixed_arguments(expr):
    with pytest.raises(ToolError):
        evaluate(expr)


# 8. atan2 with -0.0


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("atan2d(-0.0, -1)", "180"),
        ("atan2d(0, -1)", "180"),
        ("atan2d(0.0 * -1, -5)", "180"),
        ("atan2(-0.0, -1) == atan2(0, -1)", "True"),
        ("atan2d(-0.0, -0.0)", "0"),
        ("atan2d(-1, -1)", "-135"),
    ],
)
def test_atan2_ignores_the_sign_of_zero(expr, expected):
    assert _text(expr) == expected


# 9. sqrt of cancellation noise


def test_sqrt_of_cancelled_radicand_is_zero_but_tiny_values_are_kept():
    assert evaluate("0.3 - 0.1 * 3") == 0.0  # cancellation is removed where it happens
    assert _text("sqrt(0.3 - 0.1 * 3)") == "0"
    assert _text("sqrt(1e-13)") == format(math.sqrt(1e-13), ".12g")
    for expr in ["sqrt(-1e-11)", "sqrt(-1e-13)", "sqrt(-1e-300)"]:  # real negatives, not cancellation noise
        with pytest.raises(ToolError):
            evaluate(expr)


# 10. Integer-argument functions


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("factorial(5.0000000000001)", 120),
        ("comb(6.000000000001, 2)", 15),
        ("perm(5.0, 2.0)", 20),
        ("gcd(12.0, 18)", 6),
        ("gcd(4.0000000000001, 6)", 2),
        ("lcm(4.0000000000001, 6)", 12),
        ("round(2.675, 2.0000000000001)", 2.67),
        ("gcd()", 0),
        ("gcd(-4)", 4),
    ],
)
def test_integer_arguments_snap_and_accept_integer_floats(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize("expr", ["gcd(1.5, 3)", "factorial(4.5)", "gcd(1 < 2, 2)", "round(1, 0.5)"])
def test_integer_arguments_reject_non_integers(expr):
    with pytest.raises(ToolError):
        evaluate(expr)


# 11. round() is Python's round on the raw float (TRACE computes its answers that way)


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("round(1.15 * 25, 1)", "28.7"),  # 28.749999999999996, as Python rounds it
        ("round(0.3 * 1.5, 1)", "0.4"),  # 0.44999999999999996
        ("round(28.75, 1)", "28.8"),
        ("round(2.5)", "2"),
        ("round(3.5)", "4"),
        ("round(-2.5)", "-2"),
        ("round(0.125, 2)", "0.12"),
        ("round(2.675, 2)", "2.67"),
        ("round(1234, -2)", "1200"),
        ("round(1000.3 - 1000.1, 1)", "0.2"),
    ],
)
def test_round_is_pythons_round_on_the_raw_float(expr, expected):
    assert _text(expr) == expected
    assert evaluate(expr) == eval(expr)  # noqa: S307 - trusted test literals only


# 12. Comparisons with lists


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("[0.1 + 0.2, 1] == [0.3, 1]", True),
        ("[1, 2] == [1, 2, 3]", False),
        ("[1, 2] != [2, 1]", True),
        ("[1, 2] != [1, 2]", False),
        ("[1] == 1", False),
    ],
)
def test_list_equality_is_elementwise(expr, expected):
    assert evaluate(expr) is expected


@pytest.mark.parametrize("expr", ["[1] < [2]", "[1, 2] <= [1, 2]", "1 > [0]", "[1] >= 1"])
def test_list_ordering_is_rejected(expr):
    with pytest.raises(ToolError, match="=="):
        evaluate(expr)


def test_description_states_precision_and_rounding():
    description = CalculatorTool.description
    assert "12 significant digits" in description and "as shown" in description
    assert "round is Python's round: exact ties go to the even digit" in description
    assert "sind/cosd/tand" in description and "radians" in description
    assert "1e-12" not in description  # there is no absolute threshold any more
    # it is part of every calculator/all prompt; details reach the model via error hints
    assert len(description) < 1000


# --- regression tests for the second red-team review (numbered as in that review) -----

# R1. A wrong number of arguments names the function the model called


@pytest.mark.parametrize(
    "expr, name",
    [
        ("floor(2.57, 1)", "floor"),
        ("ceil()", "ceil"),
        ("trunc(1, 2)", "trunc"),
        ("pow(2, 3, 5)", "pow"),
        ("round(1.25, 1, 2)", "round"),
        ("sind()", "sind"),
        ("log(8, 2, 3)", "log"),
        ("atan2(1)", "atan2"),
        ("hypot()", "hypot"),
        ("sqrt(floor(2.5, 1))", "floor"),  # the inner call is the one reported
    ],
)
def test_wrong_argument_count_names_the_called_function(expr, name):
    text = _text(expr)
    assert text.startswith(f"Error: {name} takes "), text
    assert "<lambda>" not in text and "_safe_pow" not in text and "_round" not in text and "_sind" not in text


def test_floor_with_two_arguments_suggests_scaling():
    assert _text("floor(2.57, 1)") == (
        "Error: floor takes 1 argument, got 2; for n decimals use floor(x * 10**n) / 10**n"
    )


@pytest.mark.parametrize("name", sorted(calculator._FUNCTIONS))
def test_every_function_reports_its_own_name_for_any_argument_count(name):
    low, high = calculator._ARITY[name]
    for count in range(4):
        text = _text(f"{name}({', '.join(['2'] * count)})")
        assert "<lambda>" not in text and "positional argument" not in text, (name, count, text)
        if count < low or (high is not None and count > high):
            assert text.startswith(f"Error: {name} takes "), (name, count, text)


# R2 / R7. No absolute thresholds: tiny SI quantities keep full relative precision


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1.6e-19 * 2", "3.2e-19"),
        ("6.63e-34 * 3e8 / 500e-9", "3.978e-19"),
        ("2e-6 * 3e-7", "6e-13"),
        ("8.99e9 * 2e-6 * 3e-7 / 0.5**2", format(8.99e9 * 2e-6 * 3e-7 / 0.5**2, ".12g")),
        ("1e-19 < 1.6e-19", "True"),
        ("2e-13 == 5e-13", "False"),
        ("1.602e-19 > 9.109e-31", "True"),
        ("sqrt(1e-14)", "1e-07"),
        ("1e-14**0.5", "1e-07"),
        ("cbrt(1e-15)", "1e-05"),
        (  # de Broglie wavelength of a 100 eV electron, about 1.23e-10 m
            "6.626e-34 / sqrt(2 * 9.109e-31 * 100 * 1.602e-19)",
            format(6.626e-34 / math.sqrt(2 * 9.109e-31 * 100 * 1.602e-19), ".12g"),
        ),
        ("0.1 * 3 - 0.3", "0"),  # cancellation still gives exactly 0
        ("0.1 * 3 - 0.3 == 0", "True"),
    ],
)
def test_tiny_values_keep_relative_precision(expr, expected):
    assert _text(expr) == expected


# R3. Cancellation in sums and differences is removed where it happens


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("786.3 - 786.2", "0.1"),
        ("4470.9 - 4470.3", "0.6"),
        ("77401.7 - 77400.4", "1.3"),
        ("77401.7 - 77400.4 == 1.3", "True"),
        ("77401.7 - 77400.4 > 1.3", "False"),
        ("77401.7 - 77400.4 == 54321.2 - 54319.9", "True"),
        ("(37906.0 - 37905.8) // 0.1", "2"),
        ("floor((37906.0 - 37905.8) / 0.1)", "2"),
        ("5026.0 - 5025.9", "0.1"),
        ("ceil((5026.0 - 5025.9) * 10)", "1"),
        ("sqrt((24.9 + 745.3)**2 - 770.2**2)", "0"),  # degenerate leg: cancelled radicand
        ("1000.3 - 1000.1", "0.2"),
        ("pi - 3", "0.14159265359"),
        ("sum([786.3, -786.2])", "0.1"),
        ("sum(0.1, 0.2, -0.3)", "0"),
        ("mean([0.1, 0.2, -0.3])", "0"),
        ("stdev([0.1 + 0.2, 0.3])", "0"),
        ("77401.7 % 1", "0.7"),
        ("atan2d(0.1 * 3 - 0.3, -1)", "180"),  # a cancelled y is +0.0
        ("1e10 + 0.001 - 1e10", "0.001"),  # 15 displayed digits from 1e10 up
        ("9e9 + 0.001 - 9e9", "0"),  # 9000000000.001 is displayed as 9000000000
    ],
)
def test_cancellation_is_removed_where_it_happens(expr, expected):
    assert _text(expr) == expected


def test_cancelled_differences_match_the_exact_decimal_answer():
    from fractions import Fraction

    rng = random.Random(4)
    for digits in (3, 4, 5, 6, 7):
        for decimals in (1, 2):
            for _ in range(150):
                a = Fraction(rng.randint(10 ** (digits + decimals - 1), 10 ** (digits + decimals) - 1), 10**decimals)
                d = Fraction(rng.randint(1, 50 * 10**decimals), 10**decimals)
                expr = f"{float(a)!r} - {float(a - d)!r}"
                assert Fraction(_text(expr)) == d, expr
                assert evaluate(f"{expr} == {float(d)!r}") is True, expr
                assert evaluate(f"floor(({expr}) * {10**decimals})") == d * 10**decimals, expr
                assert evaluate(f"({expr}) // {1 / 10**decimals!r}") == d * 10**decimals, expr


# R4. floor/ceil/trunc and == use the displayed value, also for pasted-back results


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1/6", "0.166666666667"),
        ("0.166666666667 * 6", "1"),
        ("ceil(0.166666666667 * 6)", "1"),
        ("0.166666666667 * 6 == 1", "True"),
        ("floor(0.285714285714 * 7)", "2"),
        ("ceil(1.73205080757**2)", "3"),
        ("trunc(-0.285714285714 * 7)", "-2"),
        ("perm(5, 0.285714285714 * 7)", "20"),  # integer arguments use the displayed 2
        ("0.166666666667 * 30", "5.00000000001"),  # really not an integer, as displayed
        ("0.166666666667 * 6 // 1", "1"),
        ("[0.166666666667 * 6, 2] == [1, 2]", "True"),
    ],
)
def test_pasted_back_values_agree_with_the_display(expr, expected):
    assert _text(expr) == expected


def test_floor_ceil_and_equality_never_disagree_with_the_displayed_text():
    from fractions import Fraction

    rng = random.Random(7)
    for _ in range(1500):
        expr = f"{rng.randint(1, 10**6)} / {rng.randint(1, 999)} * {rng.choice([1, 3, 7, 0.1, 0.3, 1.7])}"
        shown = Fraction(_text(expr))
        assert evaluate(f"floor({expr})") == math.floor(shown), expr
        assert evaluate(f"ceil({expr})") == math.ceil(shown), expr
        assert evaluate(f"{expr} == {_text(expr)}") is True, expr


# R5. Large magnitudes: comparisons and floor/ceil use 15 significant digits


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("3.22 * 4900000000", "15778000000"),
        ("3.22 * 4900000000 == 15778000000", "True"),
        ("3.22 * 4900000000 > 15778000000", "False"),
        ("ceil(3.22 * 4900000000)", "15778000000"),
        ("9.95 * 790000000000", "7860500000000"),
        ("floor(9.95 * 790000000000)", "7860500000000"),
        ("9.95 * 790000000000 >= 7860500000000", "True"),
        ("73.9 * 94000000000000", "6.9466e+15"),  # not 6946600000000001
        ("73.9 * 94000000000000 == 6946600000000000", "True"),
        ("5e11 + 0.4 == 5e11", "False"),
        ("1e20 % 3.0", "1"),
    ],
)
def test_large_magnitudes_use_fifteen_significant_digits(expr, expected):
    assert _text(expr) == expected


# R6. round() keeps every input digit: Python's round on the raw float


@pytest.mark.parametrize(
    "expr", ["round(1234567890.149, 1)", "round(2548889221.649, 1)", "round(123456789.0449, 2)", "round(9.995, 2)"]
)
def test_round_keeps_input_digits(expr):
    assert evaluate(expr) == eval(expr)  # noqa: S307 - trusted test literals only


def test_round_sees_the_raw_difference_not_the_displayed_one():
    # 971.66 - 779.51 is 192.14999999999998 as a float and is displayed as 192.15;
    # TRACE's ground truth, Python's round on the raw float, is 192.1
    assert _text("971.66 - 779.51") == "192.15"
    assert _text("round(971.66 - 779.51, 1)") == "192.1" == str(round(971.66 - 779.51, 1))
    assert _text("round(sum([971.66, -779.51]), 1)") == "192.1"
    rng = random.Random(5)
    for _ in range(3000):
        a, b = rng.randint(1, 10**6) / 100, rng.randint(1, 10**6) / 100
        n = rng.choice([0, 1, 2])
        assert evaluate(f"round({a!r} - {b!r}, {n})") == round(a - b, n), (a, b, n)
        assert evaluate(f"round({a!r} + {b!r}, {n})") == round(a + b, n), (a, b, n)


# Radian trig is plain IEEE; degree trig is exact at special angles


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("cos(radians(90))", format(math.cos(math.radians(90)), ".12g")),
        ("tan(pi / 2)", format(math.tan(math.pi / 2), ".15g")),
        ("cosd(90)", "0"),
        ("cosd(-270)", "0"),
        ("sind(-180)", "0"),
        ("sind(150)", "0.5"),
        ("cosd(-60)", "0.5"),
        ("sind(225) == -sqrt(2) / 2", "True"),
        ("tand(135)", "-1"),
        ("tand(-45)", "-1"),
        ("sind(1e-14)", format(math.sin(math.radians(1e-14)), ".12g")),
        ("asind(-1e-15)", format(math.degrees(math.asin(-1e-15)), ".12g")),  # not snapped to 0
        ("sind(720 + 30)", "0.5"),
        ("sind([30])", "Error: sind/cosd/tand need a number"),
        ("cosd(1e999)", "Error: math domain error"),
        ("sind(3**200) == sind(3**200 % 360)", "True"),
    ],
)
def test_radian_trig_is_raw_and_degree_trig_is_exact(expr, expected):
    assert _text(expr) == expected


def test_degree_trig_is_exact_at_multiples_of_30_and_45():
    exact = {0.0, 0.5, math.sqrt(2) / 2, math.sqrt(3) / 2, 1.0}
    for angle in list(range(-720, 721, 15)):
        if angle % 30 and angle % 45:
            continue
        for value in (calculator._sind(angle), calculator._cosd(angle), calculator._sind(float(angle))):
            assert abs(value) in exact, (angle, value)
        assert calculator._sind(angle) == pytest.approx(math.sin(math.radians(angle)), abs=1e-15)
        assert calculator._cosd(angle) == pytest.approx(math.cos(math.radians(angle)), abs=1e-15)


# format_result raises nothing but ToolError


def test_format_result_turns_int_to_str_limits_into_tool_errors():
    import sys

    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        with pytest.raises(ToolError, match="too large"):
            format_result(10**700)
        assert CalculatorTool()({"expression": "10**700"}, ToolContext([])).text == "Error: result is too large"
    finally:
        sys.set_int_max_str_digits(previous)


# --- regression tests for the third red-team review (numbered as in that review) ------

# T1 / T6. Implicit multiplication asks for an explicit * and never names an AST class


@pytest.mark.parametrize(
    "expr",
    [
        "2(3+4)",
        "0.5(10)(4)",
        "(5+3)(2)",
        "(2+3)(4+5)",
        "1/2 (3)",
        "[1,2](3)",
        "sqrt(2)(3)",
        "2pi",
        "2 pi",
        "2 sqrt(3)",
        "(3+4)2",
        "(1+2) pi",
        "(1+2) sqrt(3)",
        "(1+2) .5",
    ],
)
def test_implicit_multiplication_asks_for_an_explicit_star(expr):
    text = _text(expr)
    assert text.startswith("Error: implicit multiplication is not supported; write an explicit *"), text
    for internal in ("Constant", "BinOp", "Call", "List", "unknown function", "invalid syntax", "decimal literal"):
        assert internal not in text, (expr, text)


def test_constant_called_as_a_function_asks_for_an_explicit_star():
    assert _text("pi(2)") == "Error: pi is a constant, not a function; to multiply write an explicit *, e.g. pi*(2)"
    assert _text("e (3)").startswith("Error: e is a constant, not a function")


def test_implicit_multiplication_check_keeps_other_messages_and_valid_calls():
    assert _text("2x") == "Error: unknown name 'x'; the constants are e, pi, tau"  # unknown names come first
    assert "unknown name 'if'" in _text("1if 1else 2")
    assert _text("2*(3+4)") == "14" and _text("(5+3)*(2)") == "16" and _text("2*pi") == _text("tau")
    assert _text("(sqrt)(4)") == "2"  # ")(" is a call here, so it is left to the evaluator
    assert _text("log10(100) + atan2(0, 1) + log2(8)") == "5"  # digits inside names are not numbers


def test_implicit_multiplication_and_percent_are_rejected_before_the_parser(monkeypatch):
    class NoParser:
        def __getattr__(self, name):
            raise AssertionError("ast was used for an expression the text checks should reject")

    monkeypatch.setattr(calculator, "ast", NoParser())
    for expr in ["2pi", "2(3+4)", "0.5(10)(4)", "(3+4)2", "80 * 15% + 80", "120 * 45%"]:
        with pytest.raises(ToolError):
            evaluate(expr)


# T2. Float // and % take the quotient from the unrounded operands


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1 // (1/6)", "6"),
        ("1 / (1/6)", "6"),
        ("1 % (1/6)", "0"),
        ("(pi/2) // (pi/4)", "2"),
        ("(pi/2) % (pi/4)", "0"),
        ("pi // (pi/3)", "3"),
        ("pi % (pi/2)", "0"),
        ("5 // (5/30)", "30"),
        ("5 // (5/3)", "3"),
        ("(10*pi/3) % (2*pi)", "4.18879020479"),
        ("(10*pi/3) % (2*pi) == 4*pi/3", "True"),
        # documented behaviour that must not change
        ("0.6 // 0.2", "3"),
        ("0.6 % 0.2", "0"),
        ("72 // 3.6", "20"),
        ("(0.1*3) // 0.1", "3"),
        ("1e20 % 3.0", "1"),
        ("0.166666666667 * 6 // 1", "1"),
        ("1 // 0.166666666667", "5"),  # 1 / 0.166666666667 shows 5.99999999999
    ],
)
def test_float_floordiv_and_mod_use_the_unrounded_operands(expr, expected):
    assert _text(expr) == expected


def test_float_floordiv_and_mod_of_whole_multiples_sweep():
    for n in range(1, 20):
        for r in range(1, 12):
            assert evaluate(f"({n}*2*pi*{r}) // (2*pi*{r})") == n, (n, r)
            assert evaluate(f"({n}*2*pi*{r}) % (2*pi*{r})") == 0, (n, r)
    for b in range(5, 181, 5):
        for k in range(1, 13):
            assert evaluate(f"radians({b * k}) // radians({b})") == k, (b, k)
    for n in range(1, 39):
        for k in range(1, 12):
            assert evaluate(f"{n} // ({n}/{k})") == k, (n, k)
            assert evaluate(f"{n} % ({n}/{k})") == 0, (n, k)


# T3. Operations on displayed values give the same result inside round()


@pytest.mark.parametrize(
    "inner",
    [
        "floor((5757.9 - 5757.3) * 10)",
        "(786.3 - 786.2) // 0.1",
        "ceil((5026.0 - 5025.9) * 10)",
        "trunc((5757.9 - 5757.3) * 10)",
        "77401.7 % 1",
        "factorial((5757.9 - 5757.3) * 10)",
        "comb((5757.9 - 5757.3) * 10, 2)",
        "786.3 - 786.2 == 0.1",
        "786.3 - 786.2 >= 0.1",
        "[786.3 - 786.2] == [0.1]",
    ],
)
def test_round_does_not_change_display_based_operations(inner):
    assert evaluate(f"round({inner})") == round(evaluate(inner)), inner
    assert evaluate(f"round({inner}, 1)") == round(evaluate(inner), 1), inner


def test_round_of_floor_matches_floor_sweep():
    from fractions import Fraction

    rng = random.Random(11)
    for _ in range(2000):
        a, d = rng.randint(1000, 100000), rng.randint(1, 50)
        diff = f"({float(Fraction(a, 10))!r} - {float(Fraction(a - d, 10))!r})"
        assert evaluate(f"round(floor({diff} * 10))") == evaluate(f"floor({diff} * 10)") == d, diff
        assert evaluate(f"round({diff} // 0.1)") == evaluate(f"{diff} // 0.1") == d, diff


def test_round_still_sees_raw_arithmetic_and_shown_ndigits():
    assert _text("round(971.66 - 779.51, 1)") == "192.1"  # unchanged: Python's round on the raw float
    assert _text("round(floor(971.66 - 779.51) + (971.66 - 779.51) % 1, 1)") == "192.2"  # % is as shown: 0.15
    assert evaluate("round(2.675, (5757.9 - 5757.3) * 10 - 4)") == 2.67  # ndigits uses the displayed 2


# T4. floor/ceil/trunc of floats from 2**53 up return the displayed integer


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2 * 6.022e23", "1.2044e+24"),
        ("floor(2 * 6.022e23)", "1204400000000000000000000"),
        ("ceil(2 * 6.022e23)", "1204400000000000000000000"),
        ("trunc(-2 * 6.022e23)", "-1204400000000000000000000"),
        ("floor(6.022e23) == 602200000000000000000000", "True"),
        ("6.022e23 == 602200000000000000000000", "True"),
        ("6.022e23 < 602200000000000000000001", "True"),
        ("[6.022e23] == [602200000000000000000000]", "True"),
        ("floor(1e300) == 10**300", "True"),
        ("gcd(2 * 6.022e23, 1000)", "1000"),
        ("floor(2**53 + 0.0)", "9007199254740990"),  # displayed as 9.00719925474099e+15
        ("round(2 * 6.022e23)", str(round(2 * 6.022e23))),  # round stays Python's round
        ("round(2 * 6.022e23, 0)", "1.2044e+24"),
    ],
)
def test_huge_floats_floor_to_the_displayed_integer(expr, expected):
    assert _text(expr) == expected


def test_floor_ceil_trunc_of_huge_floats_match_the_display_sweep():
    from fractions import Fraction

    rng = random.Random(13)
    for _ in range(500):
        x = rng.uniform(1, 10) * 10 ** rng.randint(16, 300) * rng.choice([1, -1])
        shown = int(Fraction(format_result(x)))
        for name in ("floor", "ceil", "trunc"):
            assert evaluate(f"{name}({x!r})") == shown, (name, x)
        assert evaluate(f"{x!r} == {shown}") is True, x


# T5. % is the remainder; percent notation gets a hint instead of a silent 0


@pytest.mark.parametrize(
    "expr", ["80 * 15% + 80", "1500 * 8% + 1500", "200*15%-5", "45% * 120", "120 * 45%", "15%", "(80*15%)+80", "7 % -3"]
)
def test_percent_sign_is_rejected_with_a_hint(expr):
    text = _text(expr)
    assert text.startswith("Error: % is the remainder (modulo), not percent; write 15% of 80 as 0.15*80"), text
    assert "% (-3)" in text


@pytest.mark.parametrize(
    "expr, expected",
    [("17 % 5", "2"), ("7 % (-3)", "-2"), ("-7 % 3", "2"), ("7.5 % 2", "1.5"), ("0.15 * 80 + 80", "92")],
)
def test_remainder_still_works(expr, expected):
    assert _text(expr) == expected


def test_description_states_multiplication_percent_and_huge_rounding():
    description = CalculatorTool.description
    assert "Write every multiplication with * (2*pi, not 2pi)" in description
    assert "% is the remainder, not percent" in description
    assert "log and ln are natural logs" in description
    assert "1e-12" not in description


# --- regression tests for the fourth red-team review (numbered as in that review) -----

# U1. A module prefix gets a hint to drop it, not "unknown name 'math'"


@pytest.mark.parametrize(
    "expr, hint",
    [
        ("math.sqrt(3**2 + 4**2)", "write sqrt(...), not math.sqrt(...)"),
        ("math.sqrt(2)", "write sqrt(...), not math.sqrt(...)"),
        ("math.pi * 2**2", "write pi, not math.pi"),
        ("math.pi*2", "write pi, not math.pi"),
        ("math.log(8, 2)", "write log(...), not math.log(...)"),
        ("math . radians(30)", "write radians(...), not math.radians(...)"),
        ("np.mean([1, 2])", "write mean(...), not np.mean(...)"),
        ("statistics.mean([3, 4, 5])", "write mean(...), not statistics.mean(...)"),
        ("Math.sqrt(2)", "write sqrt(...), not Math.sqrt(...)"),
        ("np.array([1, 2])", "write sqrt(2) or pi, not np.sqrt(2) or np.pi"),
    ],
)
def test_module_prefix_asks_to_drop_the_prefix(expr, hint):
    text = _text(expr)
    assert text.startswith(
        "Error: module prefixes are not supported; call functions and constants by name alone: "
    ), text
    assert hint in text and "unknown name" not in text, text


def test_module_prefix_hint_keeps_the_other_name_messages():
    assert _text("sqrt(4).real") == "Error: unknown name 'real'; the constants are e, pi, tau"
    assert _text("x + 1") == "Error: unknown name 'x'; the constants are e, pi, tau"
    assert _text("x.5 + 1") == "Error: unknown name 'x'; the constants are e, pi, tau"  # .5 is not a name
    assert _text("foo(1)").startswith("Error: unknown function 'foo'")


# U2 / U6. The description states the log base and the stdev convention


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("log(100) == ln(100)", "True"),  # log is the natural log, as the description says
        ("log(100, 10)", "2"),
        ("log10(100)", "2"),
        ("10*log10(2)", format(10 * math.log10(2), ".12g")),
        ("stdev([2, 4, 4, 4, 5, 5, 7, 9])", format(math.sqrt(32 / 7), ".12g")),  # n - 1
        ("pstdev([2, 4, 4, 4, 5, 5, 7, 9])", "2"),  # n
    ],
)
def test_log_and_stdev_behave_as_described(expr, expected):
    assert _text(expr) == expected


# U3. gcd, lcm and hypot take one list or several numbers, like min and max


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("gcd([12, 18])", "6"),
        ("gcd(12, 18)", "6"),
        ("lcm([4, 6, 10])", "60"),
        ("lcm(4, 6, 10)", "60"),
        ("hypot([3, 4])", "5"),
        ("hypot(3, 4)", "5"),
        ("hypot([1, 2, 2])", "3"),
        ("gcd([])", "0"),
        ("gcd()", "0"),
        ("lcm([])", "1"),
        ("lcm()", "1"),
        ("gcd([12.0, 18])", "6"),
        ("lcm([-4, 6])", "12"),
        ("gcd([2 * 6.022e23, 1000])", "1000"),
    ],
)
def test_gcd_lcm_hypot_take_one_list_or_several_numbers(expr, expected):
    assert _text(expr) == expected


@pytest.mark.parametrize(
    "expr, message",
    [
        ("gcd([1.5, 3])", "Error: gcd needs integer arguments"),
        ("lcm(4, 6.5)", "Error: lcm needs integer arguments"),
        ("lcm([4], 6)", "Error: lcm takes one list or several numbers, e.g. lcm([1, 2, 3]) or lcm(1, 2, 3)"),
        ("gcd([12], [18])", "Error: gcd takes one list or several numbers"),
        ("hypot([3], 4)", "Error: hypot takes one list or several numbers"),
        ("hypot([])", "Error: hypot needs at least one value"),
        ("hypot()", "Error: hypot takes at least 1 argument, got 0"),
    ],
)
def test_gcd_lcm_hypot_errors_name_the_function(expr, message):
    assert _text(expr).startswith(message), _text(expr)


@pytest.mark.parametrize(
    "unit, sep, prefix, suffix",
    [("2**13286-1", ",", "gcd([3**8382-1,", "])"), ("7", ",", "lcm([3**4191,", "])")],
)
def test_gcd_lcm_of_a_list_stay_within_the_work_budget(unit, sep, prefix, suffix):
    expr = _fill(unit, sep, prefix, suffix)
    started = time.thread_time()
    try:
        format_result(evaluate(expr))
    except ToolError:
        pass
    assert time.thread_time() - started < 0.2, expr[:60]
    with pytest.raises(ToolError, match="too large"):
        evaluate("lcm([10**3000+1, 10**3000+3])")


# U4. Thousands separators are rejected inside lists and calls too, never read as extra items


@pytest.mark.parametrize(
    "expr, number",
    [
        ("1,200 + 300", "1,200"),
        ("1,234", "1,234"),
        ("mean([2,500, 1,800, 3,100])", "2,500"),
        ("mean([2,500, 1,800])", "2,500"),
        ("max(1,200, 950)", "1,200"),
        ("hypot(1,200, 900)", "1,200"),
        ("sum([1,200,950])", "1,200,950"),
        ("[100,200,300]", "100,200,300"),
        ("pow(2,100)", "2,100"),
        ("-1,200.50 * 2", "1,200"),
        ("sqrt(2,500)", "2,500"),
        ("[1, 2,500]", "2,500"),
    ],
)
def test_thousands_separators_are_rejected_everywhere(expr, number):
    text = _text(expr)
    assert text.startswith(f"Error: {number} is ambiguous because commas are not thousands separators: "), text
    assert f"write {number.replace(',', '')} for one number" in text, text
    assert f"or {number.replace(',', ', ')} (a space after each comma) for separate values" in text, text


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("max(1, 200, 950)", "950"),
        ("max(1200, 950)", "1200"),
        ("mean([2500, 1800])", "2150"),
        ("[100, 200, 300]", "[100, 200, 300]"),
        ("pow(2,10)", "1024"),
        ("log(1000,10)", "3"),
        ("comb(1500,2)", "1124250"),
        ("[1.5,200]", "[1.5, 200]"),
        ("[0.5,100]", "[0.5, 100]"),
        ("[1,20,3000]", "[1, 20, 3000]"),
        ("atan2(3,4) == atan2(3, 4)", "True"),
    ],
)
def test_commas_that_cannot_be_thousands_separators_still_separate(expr, expected):
    assert _text(expr) == expected


def test_module_prefixes_and_thousands_are_rejected_before_the_parser(monkeypatch):
    class NoParser:
        def __getattr__(self, name):
            raise AssertionError("ast was used for an expression the text checks should reject")

    monkeypatch.setattr(calculator, "ast", NoParser())
    for expr in ["math.sqrt(2)", "np.mean([1, 2])", "max(1,200, 950)", "mean([2,500, 1,800])", "1,200 + 300"]:
        with pytest.raises(ToolError):
            evaluate(expr)


# U5. Cancellation keeps the real digits of computed operands


def _python_text(expr):
    return format_result(eval(expr.replace("sqrt", "math.sqrt")))  # noqa: S307 - trusted test literals only


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("(1/sqrt(1 - 0.8**2) - 1) * 1 * (3e8)**2", "6e+16"),
        ("(5/3 - 1) * 9e13", "60000000000000"),
        ("(5/3 - 1) * 9e13 == 6e13", "True"),
        ("(1 - 1/3) * 3e14", "200000000000000"),
        ("floor((1 - 1/3) * 3e14)", "200000000000000"),
        ("(1 - 1/3) * 3e14 // 1", "200000000000000"),
        ("(1 - (152.1e9 - 147.1e9)/(152.1e9 + 147.1e9)) * 149.6e9", "147100000000"),
        ("(9 - 25/3) * 9e13", _python_text("(9 - 25/3) * 9e13")),  # as in Python, not moved further
        # literal noise is still removed exactly
        ("786.3 - 786.2", "0.1"),
        ("1000.3 - 1000.1", "0.2"),
        ("0.1*3 - 0.3", "0"),
        ("77401.7 - 77400.4 == 1.3", "True"),
        ("12345678901.2346 - 12345678900.1", "1.1346"),  # a pasted-back 15-digit display
        # products, quotients and powers of literals stand for their decimal value
        ("87.4 * 460.8 - 40273.9194", "0.0006"),
        ("1.15 * 25 - 28.749", "0.001"),
        ("1 - 0.8**2", "0.36"),
        ("0.8**2 - 0.64", "0"),
        ("0.3 / 0.1 - 2.999", "0.001"),
        ("100 % 0.3", "0.1"),
        ("mean([1000.1, -999.9])", "0.1"),
        ("sum([1000.1, -999.9, 0.3])", "0.5"),
    ],
)
def test_cancellation_keeps_the_real_digits_of_computed_operands(expr, expected):
    assert _text(expr) == expected


def test_cancellation_of_computed_operands_is_never_worse_than_python_sweep():
    from fractions import Fraction

    rng = random.Random(17)
    exact_hits = python_hits = 0
    for _ in range(1500):
        q = rng.randint(3, 40)
        p = rng.randint(q + 1, 20 * q)
        c = math.ceil(p / q) + rng.choice([0, 1])
        m = rng.choice([1e10, 9e13, 3e14, 2.5e12, 6e11, 100.0, 7.5e16])
        expr = rng.choice([f"({c} - {p}/{q}) * {m!r}", f"({p}/{q} - {c}) * {m!r}"])
        exact = (Fraction(c) - Fraction(p, q)) * Fraction(m) * (1 if expr.startswith(f"({c} ") else -1)
        got, want, python = _text(expr), format_result(float(exact)), _python_text(expr)
        assert got in (want, python), (expr, got, want, python)
        exact_hits += got == want
        python_hits += python == want
    assert exact_hits >= python_hits


def test_cancelled_products_of_literals_match_the_exact_decimal_answer():
    from fractions import Fraction

    rng = random.Random(19)
    for _ in range(1500):
        x = Fraction(rng.randint(1, 99999), 10 ** rng.randint(0, 3))
        y = Fraction(rng.randint(1, 9999), 10 ** rng.randint(0, 2))
        d = Fraction(rng.randint(1, 99), 10 ** rng.randint(1, 3))
        if d <= x * y / 10**9:  # within 1e-12 of the larger term: exactly 0 by design
            continue
        expr = f"{float(x)!r} * {float(y)!r} - {float(x * y - d)!r}"
        assert Fraction(_text(expr)) == d, expr


def test_round_still_sees_the_raw_float_of_a_cancelling_difference():
    for expr in ["round((5/3 - 1) * 9e13)", "round(87.4 * 460.8 - 40273.9194, 6)", "round(1 - 0.8**2, 16)"]:
        assert evaluate(expr) == eval(expr), expr  # noqa: S307 - trusted test literals only


@pytest.mark.parametrize(
    "unit, sep, prefix, suffix",
    [
        ("1/3", "-", "", ""),
        ("0.1*3", "-", "", ""),
        ("1/3", ", ", "stdev([", "])"),
        ("1/7", ", ", "mean([", "])"),
        ("pi % 0.1", "+", "", ""),
        ("1e-300", ", ", "sum([1e300, -1e300, ", "])"),
    ],
)
def test_decimal_cancellation_work_is_bounded(unit, sep, prefix, suffix):
    expr = _fill(unit, sep, prefix, suffix)
    started = time.thread_time()
    format_result(evaluate(expr))
    assert time.thread_time() - started < 0.1, expr[:60]


def test_decimal_cancellation_is_safe_across_threads():
    exprs = ["786.3 - 786.2", "(5/3 - 1) * 9e13", "sum([0.1, 0.2, -0.3])", "100 % 0.3", "1 - 0.8**2"] * 40
    expected = [_text(e) for e in exprs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(_text, exprs)) == expected


# --- regression tests for the fifth red-team review (numbered as in that review) ------

# V1. Keyword arguments get the keyword message, not "unknown name"

_KEYWORD_MESSAGE = (
    "Error: keyword arguments are not supported; pass arguments by position, e.g. round(x, 2) or log(x, 2)"
)


@pytest.mark.parametrize(
    "expr",
    [
        "round(3.14159, ndigits=2)",
        "log(8, base=2)",
        "perm(5, k=2)",
        "round(2.5, e=1)",  # a keyword spelled like a constant reaches the evaluator's check
        "round(2.5, n = 1)",
        "round(number=2.5)",
        "round(1.5, ndigits=1)",
        "max([1, 2], key=abs)",
        "round(2.5, **e)",
    ],
)
def test_keyword_arguments_get_the_keyword_message(expr):
    assert _text(expr) == _KEYWORD_MESSAGE


def test_keyword_check_keeps_comparisons_and_positional_calls():
    assert _text("x = 5") == "Error: unknown name 'x'; the constants are e, pi, tau"  # not inside a call
    assert _text("max(x == 1)") == "Error: unknown name 'x'; the constants are e, pi, tau"  # == is a comparison
    assert _text("[x=1]") == "Error: unknown name 'x'; the constants are e, pi, tau"
    assert _text("round(3.14159, 2)") == "3.14" and _text("log(8, 2)") == "3" and _text("perm(5, 2)") == "20"
    assert _text("round(2.5, 1) == 2.5") == "True"


# V2. Method-call syntax gets a method-call hint, not the implicit-multiplication one


@pytest.mark.parametrize(
    "expr, example",
    [
        ("[3, 5, 7].mean()", "write mean([3, 5, 7]), not [3, 5, 7].mean()"),
        ("max([1, 2]).round(2)", "write round(max([1, 2]), 2), not max([1, 2]).round(2)"),
        ("(1+2).round(1)", "write round(1 + 2, 1), not (1 + 2).round(1)"),
        ("[12, 15, 9].sum()", "write sum([12, 15, 9]), not [12, 15, 9].sum()"),
        ("2.5 .round()", "write round(2.5), not 2.5.round()"),
        ("[1, 2].pi()", "write mean([3, 5, 7]), not [3, 5, 7].mean()"),
        ("mean([" + ", ".join(["1"] * 40) + "]).round(1)", "write round(x, ...), not x.round(...)"),
    ],
)
def test_method_calls_get_a_method_call_hint(expr, example):
    text = _text(expr)
    assert text == f"Error: method calls are not supported; pass the value as the first argument: {example}", text
    assert "implicit multiplication" not in text


def test_method_call_hint_keeps_the_other_messages():
    assert _text("(5+3)(2)").startswith("Error: implicit multiplication is not supported")
    assert _text("sqrt(4).real") == "Error: unknown name 'real'; the constants are e, pi, tau"
    assert _text("np.mean([1, 2])").startswith("Error: module prefixes are not supported")
    assert _text("[1, 2].mean(ndigits=2)") == _KEYWORD_MESSAGE


# V3. Factorial notation gets a hint; the compact n!=m is not silently an inequality


@pytest.mark.parametrize(
    "expr, hint",
    [
        ("5!", "write factorial(5), not 5!"),
        ("6!/(2!*4!)", "write factorial(6), not 6!"),
        ("5! = 120", "write factorial(5), not 5!"),
        ("(5-1)!", "write factorial(n), not n!"),
        ("10! / 8!", "write factorial(10), not 10!"),
    ],
)
def test_factorial_notation_gets_a_hint(expr, hint):
    text = _text(expr)
    assert text.startswith(f"Error: ! is not factorial here; {hint}, and comb(n, k) for n!/(k!*(n-k)!)"), text
    assert "invalid syntax" not in text


@pytest.mark.parametrize(
    "expr, message",
    [
        (
            "4!=24",
            "Error: 4!=24 is ambiguous because ! is not factorial here: write factorial(4) == 24 to check a "
            "factorial, or 4 != 24 (with a space before !=) for an inequality",
        ),
        (
            "3!=6",
            "Error: 3!=6 is ambiguous because ! is not factorial here: write factorial(3) == 6 to check a "
            "factorial, or 3 != 6 (with a space before !=) for an inequality",
        ),
        (
            "(5-1)!=24",
            "Error: != right after a number or ) is ambiguous because ! is not factorial here: write "
            "factorial(n) == m to check a factorial, or a != b (with a space before !=) for an inequality",
        ),
    ],
)
def test_compact_factorial_check_is_not_silently_an_inequality(expr, message):
    assert _text(expr) == message


@pytest.mark.parametrize(
    "expr, expected",
    [("4 != 24", "True"), ("4 !=24", "True"), ("4 != 4", "False"), ("pi!=3", "True"), ("[1, 2]!=[2, 1]", "True")],
)
def test_spaced_inequality_still_works(expr, expected):
    assert _text(expr) == expected


def test_keywords_method_calls_and_factorials_are_rejected_with_the_text_checks(monkeypatch):
    class NoParser:
        def __getattr__(self, name):
            raise AssertionError("ast was used for an expression the text checks should reject")

    monkeypatch.setattr(calculator, "ast", NoParser())
    for expr in ["round(3.14159, ndigits=2)", "log(8, base=2)", "5!", "6!/(2!*4!)", "4!=24"]:
        with pytest.raises(ToolError):
            evaluate(expr)


# V4. Degree trig uses the angle as displayed


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("180 - atand(1/2) - atand(2)", "90"),
        ("(180 - atand(1/2) - atand(2)) == 90", "True"),
        ("cosd(180 - atand(1/2) - atand(2))", "0"),
        ("sind(180 - atand(1/2) - atand(2))", "1"),
        ("tand(180 - atand(1/2) - atand(2))", "Error: tan is undefined at this angle"),
        ("atand(2) - atand(-0.5)", "90"),
        ("cosd(atand(2) - atand(-0.5))", "0"),
        ("tand(atand(2) - atand(-0.5))", "Error: tan is undefined at this angle"),
        ("10 * tand(atand(2) - atand(-0.5))", "Error: tan is undefined at this angle"),
        ("sind(2*(atand(1/2) + atand(2)))", "0"),
        ("sind(degrees(pi/6))", "0.5"),  # degrees(pi/6) is 29.999999999999996 and shows 30
        ("tand(degrees(pi/4))", "1"),
        ("sind(1e-14)", format(math.sin(math.radians(1e-14)), ".12g")),  # tiny angles keep their precision
        ("sind(33.123456789)", format(math.sin(math.radians(33.123456789)), ".12g")),
        ("sind(2 * 6.022e23) == sind(1204400000000000000000000)", "True"),  # the displayed integer
    ],
)
def test_degree_trig_uses_the_displayed_angle(expr, expected):
    assert _text(expr) == expected


def test_computed_right_angles_are_exact_sweep():
    for a in range(1, 21):
        for b in range(1, 21):
            for angle in (
                f"180 - atand({a}/{b}) - atand({b}/{a})",
                f"180 - acosd({a}/hypot({a}, {b})) - asind({a}/hypot({a}, {b}))",
                f"180 - acosd({a}/hypot({a}, {b})) - acosd({b}/hypot({a}, {b}))",
                f"atand({a}/{b}) - atand(-{b}/{a})",
            ):
                assert _text(angle) == "90", angle
                assert _text(f"cosd({angle})") == "0", angle
                assert _text(f"sind({angle})") == "1", angle
                assert _text(f"sind(2*({angle}))") == "0", angle
                assert _text(f"tand({angle})") == "Error: tan is undefined at this angle", angle


# V5. asin/acos snap an argument displayed as +-1 from either side


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("10/(sqrt(5)*sqrt(20))", "1"),
        ("10/(sqrt(5)*sqrt(20)) == 1", "True"),
        ("acosd(10/(sqrt(5)*sqrt(20)))", "0"),
        ("acosd(-10/(sqrt(5)*sqrt(20)))", "180"),
        ("floor(acosd(-10/(sqrt(5)*sqrt(20))))", "180"),
        ("acosd(-10/(sqrt(5)*sqrt(20))) == 180", "True"),
        ("acosd((1*2 + 1*2)/(sqrt(1**2 + 1**2)*sqrt(2**2 + 2**2)))", "0"),
        ("asind(1 - 1e-13)", "90"),
        ("acosd(0.9999999999999999)", "0"),
        ("acosd(1.0000000000000002)", "0"),
        ("acos(0.9999999999999999)", "0"),
        ("asin(-0.9999999999999999) == -pi/2", "True"),
        ("asind(0.5000000000001)", "30"),
        ("acosd(0.99999999999)", format(math.degrees(math.acos(0.99999999999)), ".12g")),  # shown as itself
        ("asin(1.00000000001)", "Error: math domain error"),
    ],
)
def test_inverse_trig_snaps_an_argument_shown_as_one(expr, expected):
    assert _text(expr) == expected


def test_parallel_vectors_give_zero_or_180_degrees_sweep():
    for ux in range(-4, 5):
        for uy in range(-4, 5):
            if ux == uy == 0:
                continue
            for k in (-3, -2, -1, 2, 3, 5, 7):
                vx, vy = k * ux, k * uy
                cosine = f"(({ux})*({vx}) + ({uy})*({vy}))/(sqrt(({ux})**2 + ({uy})**2)*sqrt(({vx})**2 + ({vy})**2))"
                assert _text(f"acosd({cosine})") == ("0" if k > 0 else "180"), cosine




def test_values_near_float_max_and_raw_round_contract():
    from trace_tool.tools.calculator import evaluate

    big = "1.7976931348623157e308"
    import decimal

    # finite (not inf); like every float from 2**53 up, floor gives the integer as displayed
    assert evaluate(f"floor({big})") == int(decimal.Decimal("1.79769313486232e308"))
    assert math.isfinite(evaluate(f"{big} // 1e300"))
    # round() sees Python's raw floats: tiny differences are not zeroed inside it
    assert evaluate("round((0.1*3 - 0.3) * 1e20, 1)") == round((0.1 * 3 - 0.3) * 1e20, 1)
    assert evaluate("0.1*3 - 0.3") == 0  # outside round() the cancellation is removed
