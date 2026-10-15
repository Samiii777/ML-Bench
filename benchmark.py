#!/usr/bin/env python3
"""ML-Bench orchestrator.

Builds a test matrix (framework × model × precision × batch size), runs each
benchmark script in its own subprocess, and collects the structured JSON
result each script emits. The script's JSON is the source of truth for
PASS/FAIL; a non-zero exit, missing JSON, or timeout is a FAIL.
"""

import argparse
import functools
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.args import PRECISIONS
from core.discovery import discover_benchmarks, resolve_benchmark_path
from core.schema import JSON_START, JSON_END, primary
from utils import config

GPU_OPS = ["gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"]
ONNX_PROVIDER_PRIORITY = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "MIGraphXExecutionProvider",
                          "ROCMExecutionProvider", "CPUExecutionProvider"]


# ── environment probes ──────────────────────────────────────────────────

def has_gpu() -> bool:
    import torch
    return torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def free_vram_gb() -> Optional[float]:
    import torch
    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return free / 1024**3


@functools.lru_cache(maxsize=None)
def framework_unavailable(framework: str) -> Optional[str]:
    if framework == "onnx" and importlib.util.find_spec("onnxruntime") is None:
        return "onnxruntime not installed (pip install onnxruntime-gpu)"
    if framework == "ollama":
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        except Exception:
            return "Ollama server not reachable on localhost:11434"
    return None


def onnx_providers(requested: Optional[str]) -> List[Optional[str]]:
    if requested:
        return [requested]
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return [None]
    gpu = [p for p in ONNX_PROVIDER_PRIORITY[:-1] if p in available]
    return gpu or ["CPUExecutionProvider"]


# ── matrix ──────────────────────────────────────────────────────────────

def expand_models(models: List[str]) -> List[str]:
    out = []
    for m in models:
        out += GPU_OPS if m == "gpu_ops" else [m]
    return out


def framework_models(args) -> List[tuple]:
    """(framework, model) pairs. A --model without --framework runs on every framework
    that lists it (PyTorch if none do, e.g. an arbitrary HF id or torchvision name)."""
    if not args.model:
        return [(fw, m) for fw in args.framework or config.DEFAULT_FRAMEWORKS for m in config.get_models(fw, args.mode)]
    pairs = []
    for m in expand_models(args.model):
        fws = args.framework or [fw for fw, ms in config.MODELS_BY_FRAMEWORK.items() if m in ms] or ["pytorch"]
        pairs += [(fw, m) for fw in fws]
    return pairs


def build_matrix(args) -> List[Dict[str, Any]]:
    """Every test to run, plus SKIP entries (with reason) for ones that can't."""
    tests = []
    gpu = has_gpu() and args.device != "cpu"
    for framework, model in framework_models(args):
        unavailable = framework_unavailable(framework)
        if any(s in model for s in args.skip_model or []):
            continue
        use_case = args.usecase or config.get_use_case(model)
        if use_case in (args.skip_usecase or []):
            continue
        if framework in config.FRAMEWORK_PRECISION:  # Ollama/llama.cpp/ComfyUI choose precision, run bs=1
            precisions, batch_sizes = [config.FRAMEWORK_PRECISION[framework]], [1]
        else:
            precisions = args.precision or config.get_precisions(framework, use_case, args.mode)
            batch_sizes = args.batch_size or config.get_batch_sizes(framework, use_case, args.mode)
        providers = onnx_providers(args.execution_provider) if framework == "onnx" else [None]
        for precision in precisions:
            for bs in batch_sizes:
                for provider in providers:
                    t = dict(framework=framework, model=model, mode=args.mode, use_case=use_case,
                             precision=precision, batch_size=bs, execution_provider=provider)
                    t["skip"] = unavailable or skip_reason(t, gpu)
                    tests.append(t)
    return tests


def skip_reason(t: Dict[str, Any], gpu: bool) -> Optional[str]:
    if resolve_benchmark_path(t["framework"], t["model"], t["mode"], t["use_case"]) is None:
        return f"no {t['framework']} {t['mode']} script for {t['model']} ({t['use_case']})"
    if t["precision"] in ("fp16", "bf16", "mixed", "tf32") and not gpu:
        return f"{t['precision']} needs a GPU"
    if t["framework"] == "onnx" and t["precision"] not in ("fp32", "fp16"):
        return "ONNX benchmarks support fp32/fp16 only"
    return None


# ── running ─────────────────────────────────────────────────────────────

def parse_result(output: str) -> Optional[Dict[str, Any]]:
    """The last JSON result block in a script's output, or None."""
    end = output.rfind(JSON_END)
    start = output.rfind(JSON_START, 0, end)
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(output[start + len(JSON_START):end])
    except json.JSONDecodeError:
        return None


def build_command(t: Dict[str, Any], args) -> List[str]:
    cmd = [sys.executable, str(resolve_benchmark_path(t["framework"], t["model"], t["mode"], t["use_case"])),
           "--model", t["model"], "--precision", t["precision"], "--batch_size", str(t["batch_size"])]
    if args.device != "auto":
        cmd += ["--device", args.device]
    if args.num_runs:
        cmd += ["--num_runs", str(args.num_runs)]
    if args.num_warmup is not None:
        cmd += ["--num_warmup", str(args.num_warmup)]
    if args.power:
        cmd += ["--power"]
    if t["execution_provider"]:
        cmd += ["--execution_provider", t["execution_provider"]]
    return cmd + shlex.split(args.script_args or "")


def run_test(t: Dict[str, Any], args) -> Dict[str, Any]:
    record = {k: v for k, v in t.items() if k != "skip"}
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if t["skip"]:
        return {**record, "status": "SKIP", "error": t["skip"]}
    if not args.skip_vram_check and t["framework"] == "pytorch":
        free = free_vram_gb()
        reason = free is not None and config.vram_skip_reason(t["model"], t["precision"], t["batch_size"], free)
        if reason:
            return {**record, "status": "SKIP", "error": reason}

    cmd = build_command(t, args)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    if args.debug:
        print(f"\n  $ {shlex.join(cmd)}")
    t0 = time.time()
    lines: List[str] = []
    try:
        # Own process group, so a timeout also kills children (llama-bench, ComfyUI server) holding the GPU
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, start_new_session=hasattr(os, "killpg"))

        def kill():
            killed.set()
            try:
                os.killpg(proc.pid, signal.SIGKILL) if hasattr(os, "killpg") else proc.kill()
            except ProcessLookupError:
                pass

        killed = threading.Event()
        watchdog = threading.Timer(args.timeout * 60, kill) if args.timeout else None
        if watchdog:
            watchdog.start()
        for line in proc.stdout:
            lines.append(line)
            if args.verbose:
                sys.stdout.write("    " + line)
        rc = proc.wait()
        if watchdog:
            watchdog.cancel()
        timed_out = killed.is_set()
    except OSError as e:
        return {**record, "status": "FAIL", "error": str(e), "duration_s": 0.0}

    output = "".join(lines)
    record["duration_s"] = round(time.time() - t0, 1)
    result = parse_result(output)
    tail = "".join(lines[-25:])
    if timed_out:
        status, error = "FAIL", f"timed out after {args.timeout} min"
    elif result is None:
        status, error = "FAIL", f"no result emitted (exit code {rc})"
    else:
        status, error = result.get("status", "FAIL"), result.get("error")
        if status == "SKIP":
            result = None  # nothing was measured
        elif rc != 0 and status == "PASS":
            status, error = "FAIL", f"exit code {rc}"
    record.update(status=status, error=error, result=result)
    if status == "FAIL":
        record["log_tail"] = tail
    return record


def describe(record: Dict[str, Any]) -> str:
    if record["status"] != "PASS":
        return f"{record['status']}: {record.get('error') or ''}"
    p = primary(record["result"]) or {}
    s = record["result"].get("latency_stats") or {}
    text = f"{p.get('value', 0):,.2f} {p.get('unit', '')}"
    if s.get("median"):
        text += f"  (p50 {s['median']:.2f} ms, cv {s.get('cv_pct', 0):.1f}%)"
    return text


def run_matrix(tests: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    records = []
    for i, t in enumerate(tests, 1):
        label = f"{t['framework']}/{t['model']} [{t['use_case']}] {t['precision']} bs={t['batch_size']}"
        if t["execution_provider"]:
            label += f" ({t['execution_provider'].replace('ExecutionProvider', '')})"
        print(f"[{i:3d}/{len(tests)}] {label:<70} ", end="", flush=True)
        record = run_test(t, args)
        records.append(record)
        mark = {"PASS": "✓", "SKIP": "-", "FAIL": "✗"}[record["status"]]
        print(f"{mark} {describe(record)}" + (f"  [{record['duration_s']:.0f}s]" if "duration_s" in record else ""))
        if record["status"] == "FAIL" and (args.verbose or args.debug) and record.get("log_tail"):
            print("    " + record["log_tail"].replace("\n", "\n    "))
    return records


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="ML-Bench: ML workload benchmarks across frameworks and GPUs")
    p.add_argument("--framework", nargs="*", choices=["pytorch", "onnx", "ollama", "llamacpp", "comfyui"],
                   help=f"default: {' '.join(config.DEFAULT_FRAMEWORKS)}")
    p.add_argument("--model", nargs="*", help="model names, HF ids, GGUF paths, or 'gpu_ops' (default: framework list)")
    p.add_argument("--mode", default="inference", choices=["inference", "training"])
    p.add_argument("--usecase", choices=sorted(set(config.USE_CASE_BY_FAMILY.values())),
                   help="override the model's use case")
    p.add_argument("--precision", nargs="*", choices=PRECISIONS, help="default: per use case")
    p.add_argument("--batch_size", "--batch-size", type=int, nargs="*", help="default: per use case")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num_runs", "--num-runs", type=int, help="timed iterations per test (script default if unset)")
    p.add_argument("--num_warmup", "--num-warmup", type=int, help="warmup iterations per test")
    p.add_argument("--power", action="store_true", help="sample GPU power during timed loops")
    p.add_argument("--execution_provider", help="ONNX Runtime provider (default: best available)")
    p.add_argument("--script-args", help="extra args passed verbatim to every script, e.g. '--max_tokens 256'")
    p.add_argument("--skip-model", nargs="*", help="skip models whose name contains any of these")
    p.add_argument("--skip-usecase", nargs="*")
    p.add_argument("--skip-vram-check", action="store_true", help="run even if the model looks too big for free VRAM")
    p.add_argument("--timeout", type=int, default=15, help="minutes per test, 0 = none (default: 15)")
    p.add_argument("--output_dir", "--output-dir", default="benchmark_results")
    p.add_argument("--list-benchmarks", action="store_true", help="list benchmark scripts and default matrix, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the test matrix without running it")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"), help="compare two result JSON files")
    p.add_argument("--dashboard", action="store_true", help="launch the Streamlit dashboard")
    p.add_argument("-v", "--verbose", action="store_true", help="stream each script's output")
    p.add_argument("--debug", action="store_true", help="print commands and failure logs")
    args = p.parse_args()

    if args.dashboard:
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard.py")]))
    if args.compare:
        from compare import main as compare_main
        sys.exit(compare_main(args.compare))
    if args.list_benchmarks:
        for key, path in discover_benchmarks().items():
            print(f"{'/'.join(key):<55} {path.relative_to(ROOT)}")
        print("\nDefault models:")
        for fw, models in config.MODELS_BY_FRAMEWORK.items():
            print(f"  {fw:<9} {', '.join(models) or '(pass --model)'}")
        return

    tests = build_matrix(args)
    if args.dry_run:
        for t in tests:
            print(f"{t['framework']}/{t['model']} {t['use_case']} {t['precision']} bs={t['batch_size']}"
                  + (f" {t['execution_provider']}" if t["execution_provider"] else "")
                  + (f"  SKIP: {t['skip']}" if t["skip"] else ""))
        return

    print(f"Running {len(tests)} tests ({sum(1 for t in tests if not t['skip'])} runnable)\n")
    records = run_matrix(tests, args)

    from utils.results import save_results, print_summary
    print_summary(records)
    save_results(records, args.output_dir, vars(args))
    sys.exit(1 if any(r["status"] == "FAIL" for r in records) else 0)


if __name__ == "__main__":
    main()
