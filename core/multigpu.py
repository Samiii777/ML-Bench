"""Multi-GPU support for benchmark harnesses."""

import os
from typing import List, Optional

import torch
import torch.nn as nn


def get_available_gpus() -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def wrap_data_parallel(model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
    if device_ids is None:
        device_ids = get_available_gpus()
    if len(device_ids) <= 1:
        return model
    print(f"Wrapping model in DataParallel across GPUs: {device_ids}")
    return nn.DataParallel(model, device_ids=device_ids)


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def calculate_scaling_efficiency(
    single_gpu_throughput: float,
    multi_gpu_throughput: float,
    num_gpus: int,
) -> float:
    ideal = single_gpu_throughput * num_gpus
    if ideal <= 0:
        return 0.0
    return (multi_gpu_throughput / ideal) * 100.0
