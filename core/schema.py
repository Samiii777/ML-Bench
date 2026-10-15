"""Canonical data structures for benchmark results.

Every benchmark script prints exactly one ``BenchmarkResult`` as JSON between
``JSON_START`` / ``JSON_END``. The orchestrator (benchmark.py) parses that block
and treats it as the single source of truth for status and metrics.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "2.0"

JSON_START = "===BENCHMARK_JSON_START==="
JSON_END = "===BENCHMARK_JSON_END==="


@dataclass
class MetricEntry:
    name: str
    value: float
    unit: str
    direction: str = "lower_is_better"  # or "higher_is_better"


@dataclass
class SystemInfo:
    device: str = "unknown"
    device_name: str = "unknown"
    torch_version: str = ""
    gpu_vendor: str = "unknown"
    gpu_memory_total_gb: Optional[float] = None
    gpu_driver_version: str = ""
    rocm_version: str = ""
    cuda_version: str = ""
    kernel_version: str = ""
    python_version: str = ""
    platform: str = ""
    cpu_model: str = ""
    hostname: str = ""


@dataclass
class BenchmarkResult:
    status: str  # "PASS", "FAIL", or "SKIP" (config not measurable here; error says why)
    framework: str
    model: str
    mode: str
    use_case: str
    precision: str  # effective precision, i.e. what actually ran
    batch_size: int
    system_info: SystemInfo = field(default_factory=SystemInfo)
    metrics: List[MetricEntry] = field(default_factory=list)
    # Per-iteration latency stats in ms (mean/median/p90/p95/p99/min/max/std/cv_pct/n)
    latency_stats: Dict[str, float] = field(default_factory=dict)
    # Name of the metric that best summarises this run (what reports lead with)
    primary_metric: str = "throughput"
    # Workload parameters needed to interpret the numbers (resolution, seq len, steps, ...)
    config: Dict[str, Any] = field(default_factory=dict)
    validation_checks: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        d = self.to_dict()
        d["__benchmark_result__"] = True
        return json.dumps(d, indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> "BenchmarkResult":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        d = dict(d)
        d.pop("__benchmark_result__", None)
        si = d.pop("system_info", {}) or {}
        if isinstance(si, dict):
            known_si = set(SystemInfo.__dataclass_fields__)
            si = SystemInfo(**{k: v for k, v in si.items() if k in known_si})
        d["system_info"] = si
        d["metrics"] = [MetricEntry(**m) if isinstance(m, dict) else m for m in d.get("metrics", [])]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def get_metric(self, name: str) -> Optional[float]:
        for m in self.metrics:
            if m.name == name:
                return m.value
        return None


def metric_map(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{name: {value, unit, direction}} for a result dict (as stored in JSON)."""
    return {m["name"]: m for m in result.get("metrics") or [] if isinstance(m, dict) and "name" in m}


def primary(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The headline metric entry of a result dict, or None."""
    return metric_map(result).get(result.get("primary_metric") or "throughput")
