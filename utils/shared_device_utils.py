"""Best-effort system fingerprint recorded with every result for reproducibility."""


def collect_system_fingerprint() -> dict:
    """Collect comprehensive system info for reproducibility.

    Returns a dict whose keys match ``core.schema.SystemInfo`` fields.
    All probes are best-effort — failures return empty strings.
    """
    import platform as _platform
    import subprocess as _sp
    import sys as _sys

    info: dict = {
        "python_version": _sys.version.split()[0],
        "platform": _platform.platform(),
        "kernel_version": _platform.release(),
        "hostname": _platform.node(),
    }

    # CPU model
    try:
        if _platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        elif _platform.system() == "Darwin":
            out = _sp.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True, timeout=5)
            info["cpu_model"] = out.stdout.strip()
    except Exception:
        pass

    # PyTorch / GPU stack versions
    try:
        import torch
        info["torch_version"] = torch.__version__
        cuda_ver = getattr(torch.version, "cuda", None)
        hip_ver = getattr(torch.version, "hip", None)
        if hip_ver:
            info["rocm_version"] = hip_ver
        if cuda_ver:
            info["cuda_version"] = cuda_ver
    except ImportError:
        pass

    # GPU driver version (each probe independent: a missing rocm-smi must not skip nvidia-smi)
    probes = [(["rocm-smi", "--showdriverversion"], lambda out: next(
                  (l.split(":")[-1].strip() for l in out.splitlines() if "driver version" in l.lower()), "")),
              (["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
               lambda out: out.strip().splitlines()[0] if out.strip() else "")]
    for cmd, parse in probes:
        try:
            out = _sp.run(cmd, capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and parse(out.stdout):
                info["gpu_driver_version"] = parse(out.stdout)
                break
        except Exception:
            continue

    return info
