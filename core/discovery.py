"""Convention-based benchmark discovery.

Scans ``benchmarks/`` for ``main.py`` files and builds a registry
mapping (framework, model_family, mode, use_case) → script path.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.config import get_model_family

RegistryKey = Tuple[str, ...]  # (framework, model_family, mode, use_case[, subdir])


def discover_benchmarks(root: Optional[Path] = None) -> Dict[RegistryKey, Path]:
    """Walk ``benchmarks/`` and return a registry of discovered scripts.

    Each key is a tuple of path components between ``benchmarks/`` and
    ``main.py``.  For standard scripts this is
    ``(framework, model_family, mode, use_case)``; for gpu_ops it adds a
    5th component for the operation subdirectory.
    """
    if root is None:
        root = _project_root
    benchmarks_dir = root / "benchmarks"
    registry: Dict[RegistryKey, Path] = {}

    for main_py in sorted(benchmarks_dir.rglob("main.py")):
        try:
            rel = main_py.relative_to(benchmarks_dir)
        except ValueError:
            continue
        parts = rel.parent.parts  # e.g. ("pytorch", "resnet", "inference", "classification")
        if not parts:
            continue
        registry[parts] = main_py

    return registry


def resolve_benchmark_path(
    registry: Dict[RegistryKey, Path],
    framework: str,
    model: str,
    mode: str,
    use_case: str,
) -> Optional[Path]:
    """Find the script path for a given benchmark configuration.

    Handles special directory layouts:
    - Ollama: ``benchmarks/ollama/{use_case}/main.py`` (no *mode* level)
    - ComfyUI: ``benchmarks/ComfyUI/main.py`` (flat)
    - gpu_ops: 5-level nesting with operation subdirectories
    """
    # Direct match (standard 4-level)
    key = (framework, get_model_family(model), mode, use_case)
    if key in registry:
        return registry[key]

    # Ollama: (ollama, use_case)
    if framework == "ollama":
        key2 = ("ollama", use_case)
        if key2 in registry:
            return registry[key2]

    # ComfyUI: (ComfyUI,)
    if framework == "comfyui":
        for k, v in registry.items():
            if k[0].lower() == "comfyui":
                return v

    # gpu_ops: (framework, gpu_ops, mode, use_case, subdir)
    family = get_model_family(model)
    if family == "gpu_ops":
        ops_map = {
            "gemm_ops": "gemm",
            "conv_ops": "conv",
            "memory_ops": "memory",
            "elementwise_ops": "elementwise",
            "reduction_ops": "reduction",
        }
        subdir = ops_map.get(model, model.replace("_ops", ""))
        key5 = (framework, "gpu_ops", mode, use_case, subdir)
        if key5 in registry:
            return registry[key5]

    # Fuzzy: try matching just (framework, family, ...)
    for k, v in registry.items():
        if len(k) >= 2 and k[0] == framework and k[1] == family:
            return v

    return None


def load_benchmark_meta(script_path: Path):
    """Try to import BENCHMARK_META from a script. Returns dict or None."""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("_bm_probe", str(script_path))
        if spec is None or spec.loader is None:
            return None
        # Only read the source, don't execute — look for BENCHMARK_META assignment
        source = script_path.read_text()
        if "BENCHMARK_META" not in source:
            return None
        module = importlib.util.module_from_spec(spec)
        # Prevent side effects by not fully loading heavy imports
        # Instead, parse the BENCHMARK_META from source directly
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "BENCHMARK_META":
                        return {"has_meta": True, "path": str(script_path)}
        return None
    except Exception:
        return None


# Module-level cache so the scan only runs once per process.
_registry_cache: Optional[Dict[RegistryKey, Path]] = None


def get_registry(root: Optional[Path] = None) -> Dict[RegistryKey, Path]:
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = discover_benchmarks(root)
    return _registry_cache
