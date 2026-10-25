"""Choosing the PyTorch device once, at the composition root.

``cuda`` covers both NVIDIA (CUDA) and AMD (ROCm) builds of PyTorch. ROCm also lists
integrated AMD GPUs it cannot run on, and they report shared system memory, which can be
more than a discrete GPU has. The default therefore skips integrated GPUs, then picks the
one with the most memory.
"""

from collections.abc import Sequence
from dataclasses import dataclass

CPU = "cpu"


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """One visible GPU."""

    index: int
    name: str
    architecture: str
    memory_bytes: int
    integrated: bool


def choose_device(requested: str | None, gpus: Sequence[GpuInfo]) -> str:
    """``requested`` if given, else the discrete GPU with the most memory, else the CPU."""
    if requested is not None:
        return requested
    discrete = [gpu for gpu in gpus if not gpu.integrated]
    if not discrete:
        return CPU
    best = max(discrete, key=lambda gpu: gpu.memory_bytes)
    return f"cuda:{best.index}"


def visible_gpus() -> list[GpuInfo]:
    """GPUs PyTorch can see; empty on CPU builds."""
    import torch  # Imported here so the CLI starts without torch for data commands.

    if not torch.cuda.is_available():
        return []
    gpus: list[GpuInfo] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        architecture = str(
            getattr(properties, "gcnArchName", f"sm_{properties.major}{properties.minor}")
        )
        gpus.append(
            GpuInfo(
                index,
                properties.name,
                architecture,
                properties.total_memory,
                integrated=bool(properties.is_integrated),
            )
        )
    return gpus


def describe(device: str, gpus: Sequence[GpuInfo]) -> str:
    """A log line naming the chosen device and, for a GPU, its architecture."""
    for gpu in gpus:
        if device == f"cuda:{gpu.index}" or (device == "cuda" and gpu.index == 0):
            memory = gpu.memory_bytes / 2**30
            return f"device {device}: {gpu.name} ({gpu.architecture}, {memory:.0f} GiB)"
    return f"device {device}"
