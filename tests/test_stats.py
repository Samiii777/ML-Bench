import math

import pytest

from utils.benchmark_utils import compute_stats


def test_known_values():
    s = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == 3.0 and s["median"] == 3.0
    assert s["min"] == 1.0 and s["max"] == 5.0 and s["n"] == 5
    assert s["std"] == pytest.approx(math.sqrt(2.5))  # sample std (ddof=1)
    assert s["cv_pct"] == pytest.approx(math.sqrt(2.5) / 3 * 100)
    assert s["p99"] == pytest.approx(4.96)


def test_single_and_empty():
    assert compute_stats([7.0])["std"] == 0.0
    assert compute_stats([]) == {"n": 0}
