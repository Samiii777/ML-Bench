"""Result validation framework for benchmark correctness checks."""

from dataclasses import dataclass, asdict
from typing import Any, List, Set, Tuple
import math


@dataclass
class ValidationCheck:
    name: str
    check_type: str  # "equals", "in_set", "range", "not_nan", "greater_than"
    expected: Any
    actual: Any = None
    passed: bool = False
    message: str = ""


class ResultValidator:
    """Accumulates validation checks and reports pass/fail."""

    def __init__(self):
        self.checks: List[ValidationCheck] = []

    def expect_equals(self, name: str, actual: Any, expected: Any) -> None:
        passed = actual == expected
        self.checks.append(ValidationCheck(
            name=name, check_type="equals", expected=expected,
            actual=actual, passed=passed,
            message="" if passed else f"expected {expected!r}, got {actual!r}",
        ))

    def expect_in_set(self, name: str, actual: Any, valid_set: Set) -> None:
        passed = actual in valid_set
        self.checks.append(ValidationCheck(
            name=name, check_type="in_set", expected=str(valid_set),
            actual=actual, passed=passed,
            message="" if passed else f"{actual!r} not in {valid_set!r}",
        ))

    def expect_in_range(self, name: str, actual: float, lo: float, hi: float) -> None:
        passed = lo <= actual <= hi
        self.checks.append(ValidationCheck(
            name=name, check_type="range", expected=f"[{lo}, {hi}]",
            actual=actual, passed=passed,
            message="" if passed else f"{actual} not in [{lo}, {hi}]",
        ))

    def expect_not_nan(self, name: str, actual: float) -> None:
        passed = not (isinstance(actual, float) and math.isnan(actual))
        self.checks.append(ValidationCheck(
            name=name, check_type="not_nan", expected="not NaN",
            actual=actual, passed=passed,
            message="" if passed else f"{name} is NaN",
        ))

    def expect_greater_than(self, name: str, actual: float, threshold: float) -> None:
        passed = actual > threshold
        self.checks.append(ValidationCheck(
            name=name, check_type="greater_than", expected=f">{threshold}",
            actual=actual, passed=passed,
            message="" if passed else f"{actual} not > {threshold}",
        ))

    def validate(self) -> Tuple[bool, List[ValidationCheck]]:
        all_passed = all(c.passed for c in self.checks)
        return all_passed, self.checks

    def to_dicts(self) -> List[dict]:
        return [asdict(c) for c in self.checks]
