"""Canonical data structures for benchmark results."""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"

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


@dataclass
class BenchmarkMeta:
    framework: str
    model_family: str
    supported_models: List[str] = field(default_factory=list)
    supported_precisions: List[str] = field(default_factory=lambda: ["fp32", "fp16", "mixed"])
    mode: str = "inference"
    use_case: str = "classification"


@dataclass
class BenchmarkResult:
    status: str  # "PASS" or "FAIL"
    framework: str
    model: str
    mode: str
    use_case: str
    precision: str
    batch_size: int
    system_info: SystemInfo = field(default_factory=SystemInfo)
    metrics: List[MetricEntry] = field(default_factory=list)
    latency_stats: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        d = asdict(self)
        d["__benchmark_result__"] = True
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "BenchmarkResult":
        d = json.loads(s)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        d = dict(d)
        d.pop("__benchmark_result__", None)
        d.pop("schema_version", None)

        si = d.pop("system_info", {})
        if isinstance(si, dict):
            si = SystemInfo(**si)
        d["system_info"] = si

        raw_metrics = d.pop("metrics", [])
        d["metrics"] = [
            MetricEntry(**m) if isinstance(m, dict) else m for m in raw_metrics
        ]

        return cls(**d)

    def get_metric(self, name: str) -> Optional[float]:
        for m in self.metrics:
            if m.name == name:
                return m.value
        return None

    def to_legacy_metrics(self) -> Dict[str, Any]:
        """Convert to the dict format that benchmark.py _parse_benchmark_output returns.

        The results display (utils/results.py) expects specific key names:
        throughput_fps, avg_latency_ms, inference_time_ms, tokens_per_second,
        best_gflops, best_bandwidth_gbs, seconds_per_image, etc.
        """
        NAME_MAP = {
            "throughput": "throughput_fps",
            "avg_latency_ms": "avg_latency_ms",
            "peak_memory_gb": "gpu_memory_allocated_gb",
            "tokens_per_second": "tokens_per_second",
            "best_gflops": "best_gflops",
            "best_bandwidth_gbs": "best_bandwidth_gbs",
            "seconds_per_image": "seconds_per_image",
        }

        legacy: Dict[str, Any] = {}
        legacy["device"] = self.system_info.device
        legacy["framework"] = self.framework

        for m in self.metrics:
            key = NAME_MAP.get(m.name, m.name)
            legacy[key] = m.value

        if self.latency_stats:
            if "mean" in self.latency_stats:
                legacy.setdefault("inference_time_ms", self.latency_stats["mean"])
                legacy.setdefault("avg_latency_ms", self.latency_stats["mean"])

        return legacy
