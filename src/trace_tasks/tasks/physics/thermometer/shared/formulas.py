"""Formula helpers for thermometer temperature conversion."""

from __future__ import annotations


def convert_temperature(source_unit: str, target_unit: str, source_temperature: int) -> int:
    """Convert between supported thermometer units with integer answers."""

    source = int(source_temperature)
    source_text = str(source_unit)
    target_text = str(target_unit)
    if source_text == "C" and target_text == "F":
        if source % 5 != 0:
            raise ValueError("Celsius source temperature must be a multiple of 5")
        return int(source * 9 // 5 + 32)
    if source_text == "F" and target_text == "C":
        numerator = int(source - 32) * 5
        if numerator % 9 != 0:
            raise ValueError("Fahrenheit source temperature must convert to an integer Celsius answer")
        return int(numerator // 9)
    raise ValueError(f"unsupported temperature conversion: {source_text} to {target_text}")


def source_temperature_from_target(source_unit: str, target_unit: str, target_temperature: int) -> int:
    """Return the supported source temperature for one target answer."""

    target = int(target_temperature)
    source_text = str(source_unit)
    target_text = str(target_unit)
    if source_text == "C" and target_text == "F":
        numerator = int(target - 32) * 5
        if numerator % 9 != 0:
            raise ValueError("target Fahrenheit answer does not map to an integer Celsius source")
        return int(numerator // 9)
    if source_text == "F" and target_text == "C":
        if target % 5 != 0:
            raise ValueError("target Celsius answer must be a multiple of 5 for supported Fahrenheit sources")
        return int(target * 9 // 5 + 32)
    raise ValueError(f"unsupported temperature conversion: {source_text} to {target_text}")


__all__ = ["convert_temperature", "source_temperature_from_target"]
