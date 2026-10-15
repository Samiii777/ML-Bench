import argparse
import json

import pytest

from core.schema import JSON_END, JSON_START


def make_args(**kw):
    base = dict(model="fake", precision="fp32", batch_size=1, device="cpu", num_warmup=1, num_runs=5, power=False)
    base.update(kw)
    return argparse.Namespace(**base)


def emitted(capsys):
    """The BenchmarkResult dict a harness printed."""
    out = capsys.readouterr().out
    return json.loads(out[out.rindex(JSON_START) + len(JSON_START):out.rindex(JSON_END)])


@pytest.fixture
def args():
    return make_args
