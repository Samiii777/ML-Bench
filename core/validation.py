"""Correctness checks attached to a benchmark result. Any failed check => FAIL."""

import math
from dataclasses import dataclass, asdict
from typing import Any, List, Tuple


@dataclass
class ValidationCheck:
    name: str
    check_type: str
    expected: Any
    actual: Any = None
    passed: bool = False
    message: str = ""


class ResultValidator:
    def __init__(self):
        self.checks: List[ValidationCheck] = []

    def _add(self, name, check_type, expected, actual, passed, message):
        self.checks.append(ValidationCheck(name, check_type, expected, actual, bool(passed),
                                           "" if passed else message))

    def expect_equals(self, name: str, actual: Any, expected: Any) -> None:
        self._add(name, "equals", expected, actual, actual == expected, f"expected {expected!r}, got {actual!r}")

    def expect_in_range(self, name: str, actual: float, lo: float, hi: float) -> None:
        self._add(name, "range", f"[{lo}, {hi}]", actual, lo <= actual <= hi, f"{actual} not in [{lo}, {hi}]")

    def expect_greater_than(self, name: str, actual: float, threshold: float) -> None:
        self._add(name, "greater_than", f">{threshold}", actual, actual > threshold, f"{actual} not > {threshold}")

    def expect_finite(self, name: str, actual: float) -> None:
        ok = isinstance(actual, (int, float)) and math.isfinite(actual)
        self._add(name, "finite", "finite", actual, ok, f"{actual!r} is not finite")

    def validate(self) -> Tuple[bool, List[ValidationCheck]]:
        return all(c.passed for c in self.checks), self.checks

    def to_dicts(self) -> List[dict]:
        return [asdict(c) for c in self.checks]
