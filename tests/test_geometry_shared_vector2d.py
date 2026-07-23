import math

from trace_tasks.tasks.geometry.shared.vector2d import (
    add,
    add_scaled,
    distance,
    dot,
    mid,
    mul,
    perp,
    point_to_list,
    sub,
    unit,
)


def test_vector2d_basic_operations() -> None:
    assert add((1, 2), (3, 4)) == (4.0, 6.0)
    assert add_scaled((1, 2), (3, 4)) == (4.0, 6.0)
    assert add_scaled((1, 2), (3, 4), scale=0.5) == (2.5, 4.0)
    assert sub((5, 7), (2, 3)) == (3.0, 4.0)
    assert mul((2, -3), 2.5) == (5.0, -7.5)
    assert mid((2, 4), (6, 10)) == (4.0, 7.0)
    assert dot((2, 3), (4, -5)) == -7.0
    assert distance((1, 1), (4, 5)) == 5.0
    assert perp((2, 5)) == (-5.0, 2.0)


def test_vector2d_unit_and_zero_fallback() -> None:
    ux, uy = unit((3, 4))
    assert math.isclose(ux, 0.6)
    assert math.isclose(uy, 0.8)

    assert unit((0, 0)) == (1.0, 0.0)


def test_point_to_list_rounding() -> None:
    assert point_to_list((1.23456, 9.87654)) == [1.235, 9.877]
    assert point_to_list((1.23456, 9.87654), ndigits=1) == [1.2, 9.9]
