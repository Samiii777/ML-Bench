"""GPU power monitoring for benchmark energy efficiency metrics."""

import time
import threading
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PowerStats:
    avg_power_watts: float
    peak_power_watts: float
    energy_joules: float
    measurement_duration_s: float
    num_samples: int


class PowerSampler:
    """Background thread that samples GPU power draw at a configurable interval."""

    def __init__(self, interval_ms: int = 100):
        self._interval_s = interval_ms / 1000.0
        self._samples: List[float] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._start_time = 0.0
        self._vendor = self._detect_vendor()

    @staticmethod
    def _detect_vendor() -> str:
        try:
            import torch
            if getattr(torch.version, "hip", None):
                return "amd"
            if torch.cuda.is_available():
                return "nvidia"
        except ImportError:
            pass
        return "unknown"

    def _sample(self) -> Optional[float]:
        if self._vendor == "amd":
            return self._sample_amd()
        if self._vendor == "nvidia":
            return self._sample_nvidia()
        return None

    @staticmethod
    def _sample_amd() -> Optional[float]:
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
            if not handles:
                amdsmi.amdsmi_shut_down()
                return None
            info = amdsmi.amdsmi_get_power_info(handles[0])
            amdsmi.amdsmi_shut_down()
            return float(info.get("current_socket_power", info.get("average_socket_power", 0)))
        except Exception:
            pass
        try:
            import subprocess
            out = subprocess.run(["rocm-smi", "--showpower", "--csv"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                for line in out.stdout.strip().splitlines()[1:]:
                    parts = line.split(",")
                    for p in parts:
                        p = p.strip()
                        try:
                            val = float(p)
                            if 0 < val < 1000:
                                return val
                        except ValueError:
                            continue
        except Exception:
            pass
        return None

    @staticmethod
    def _sample_nvidia() -> Optional[float]:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            pynvml.nvmlShutdown()
            return power_mw / 1000.0
        except Exception:
            pass
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return float(out.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return None

    def _run(self):
        while not self._stop.is_set():
            val = self._sample()
            if val is not None and val > 0:
                self._samples.append(val)
            self._stop.wait(self._interval_s)

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[PowerStats]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        duration = time.perf_counter() - self._start_time

        if not self._samples:
            return None

        avg = sum(self._samples) / len(self._samples)
        peak = max(self._samples)
        energy = avg * duration

        return PowerStats(
            avg_power_watts=avg,
            peak_power_watts=peak,
            energy_joules=energy,
            measurement_duration_s=duration,
            num_samples=len(self._samples),
        )

    @staticmethod
    def available() -> bool:
        """Check if power monitoring is available on this system."""
        s = PowerSampler(interval_ms=1000)
        val = s._sample()
        return val is not None and val > 0
