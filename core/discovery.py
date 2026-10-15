"""Map (framework, model, mode, use_case) to a benchmark script.

Layout convention:
  benchmarks/<framework>/<family>/<mode>/<use_case>/main.py   (pytorch, onnx)
  benchmarks/ollama/<use_case>/main.py
  benchmarks/llamacpp/main.py, benchmarks/ComfyUI/main.py
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.config import get_model_family

BENCHMARKS_DIR = _project_root / "benchmarks"


def discover_benchmarks(root: Path = _project_root) -> Dict[Tuple[str, ...], Path]:
    """{path components between benchmarks/ and main.py: script path}"""
    bdir = root / "benchmarks"
    return {p.relative_to(bdir).parent.parts: p for p in sorted(bdir.rglob("main.py"))
            if p.parent != bdir and "ComfyUI/ComfyUI" not in p.as_posix()}


def resolve_benchmark_path(framework: str, model: str, mode: str, use_case: str,
                           root: Path = _project_root) -> Optional[Path]:
    bdir = root / "benchmarks"
    candidates = {
        "ollama": bdir / "ollama" / use_case / "main.py",
        "llamacpp": bdir / "llamacpp" / "main.py",
        "comfyui": bdir / "ComfyUI" / "main.py",
    }
    if framework in candidates and mode != "inference":
        return None
    path = candidates.get(framework, bdir / framework / get_model_family(model) / mode / use_case / "main.py")
    return path if path.exists() else None
