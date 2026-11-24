from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gmdhtorch import (
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_MAX_OBSERVE_BYTES,
)
from scipy.ndimage import shift as scipy_shift
from torch.utils.data import DataLoader, TensorDataset


def split_indexes(count: int, proportions: tuple[int, int, int]) -> tuple[int, int]:
    whole = sum(proportions)
    t_prop = proportions[0]
    s_prop = proportions[1]
    sel_split = count * t_prop // whole
    val_split = count * s_prop // whole + sel_split
    return sel_split, val_split


def configure_matplotlib(backend: str | None = None) -> None:
    """Configure Matplotlib to use ``backend`` (defaults to None).

    Example:
    ```python
    # Use ``WebAgg`` for Remote-SSH without display forwarding
    configure_matplotlib("WebAgg")
    ```
    """
    if backend is not None:
        import matplotlib
        matplotlib.use(backend)


def shuffle_array(array: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    gen = np.random.default_rng() if rng is None else rng
    gen.shuffle(array)
    return array


def add_noise(signal: np.ndarray, noise: float, rng: np.random.Generator | None = None) -> np.ndarray:
    gen = np.random.default_rng() if rng is None else rng
    return signal + gen.normal(0.0, noise, size=signal.shape)


def delay(signal: np.ndarray, samples: int) -> np.ndarray:
    """Shift vector or 2D time-series ``signal`` by ``samples`` steps along axis 0 (zero-padded).

    Positive ``samples`` delays the signal ($z^{-d}$); negative ``samples`` advances it.
    """
    sig = np.asarray(signal, dtype=np.float64)
    s_int = int(samples)
    n_rows = sig.size if (sig.ndim == 1 or min(sig.shape) == 1) else int(sig.shape[0])
    if n_rows < abs(s_int):
        raise ValueError("Delay greater than the samples in the signal.")
    if s_int == 0:
        return sig.copy()
    if sig.ndim == 1 or min(sig.shape) == 1:
        orig_shape = sig.shape
        flat = sig.ravel()
        shifted = scipy_shift(flat, s_int, cval=0.0, order=0, prefilter=False)
        return shifted.reshape(orig_shape)
    if sig.ndim == 2:
        return scipy_shift(sig, (s_int, 0), cval=0.0, order=0, prefilter=False)
    raise ValueError("Signal must be a 1D vector or 2D matrix.")


def create_samples(
    rows: int,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 5-input polynomial dataset with train, selection, and validation splits."""
    gen = np.random.default_rng(rng)
    inputs = gen.random((rows, 5))
    coef = gen.random((5, 1))
    inputs = 2.0 * inputs - 1.0
    coef = 2.0 * coef - 1.0
    tmp = np.column_stack(
        [
            inputs[:, 0] ** 4,
            inputs[:, 1] ** 3,
            inputs[:, 2] ** 2,
            inputs[:, 3],
            inputs[:, 3],
        ]
    )
    outputs = (tmp @ coef).ravel()
    indexes = np.arange(1, rows + 1)
    val_filter = (indexes % 4) == 0
    select_filter = (indexes % 4) == 3
    train_filter = ~val_filter & ~select_filter
    train_samples = inputs[train_filter, :]
    train_targets = outputs[train_filter]
    select_samples = inputs[select_filter, :]
    select_targets = outputs[select_filter]
    validate_samples = inputs[val_filter, :]
    validate_targets = outputs[val_filter]
    return (
        train_samples,
        train_targets,
        select_samples,
        select_targets,
        validate_samples,
        validate_targets,
    )


def configure_torch_resources(
    *,
    device: torch.device | str | None = None,
    memory_fraction: float | None = None,
    matmul_precision: str | None = None,
) -> None:
    """Apply standard PyTorch backend resource controls for both GMDH and dense models."""
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))
    if memory_fraction is not None:
        frac = float(memory_fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError("`memory_fraction` must be in (0.0, 1.0].")
        dev = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        if dev.type == "cuda" and torch.cuda.is_available():
            dev_idx = 0 if dev.index is None else int(dev.index)
            torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
        elif dev.type == "mps" and torch.backends.mps.is_available():
            if hasattr(torch.mps, "set_per_process_memory_fraction"):
                torch.mps.set_per_process_memory_fraction(frac)


def suggest_memory_budgets(
    device: torch.device | str,
    *,
    fraction: float = 0.25,
    max_chunk_cap: int = 2 * 1024 * 1024 * 1024,
    max_observe_cap: int = 1 * 1024 * 1024 * 1024,
) -> dict[str, int]:
    """Recommend explicit ``max_chunk_bytes`` and ``max_observe_bytes`` from available device memory.

    Returns a kwargs dictionary suitable for ``ModelIncrementer(..., **suggest_memory_budgets(dev))``
    (accepted by both GMDH and no-op dense model wrappers).
    """
    dev = torch.device(device)
    frac = float(fraction)
    free_bytes: int | None = None

    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            dev_idx = 0 if dev.index is None else int(dev.index)
            free_bytes, _total = torch.cuda.mem_get_info(dev_idx)
        except RuntimeError:
            free_bytes = None
    elif dev.type == "mps" and torch.backends.mps.is_available():
        try:
            rec = int(torch.mps.recommended_max_memory())
            cur = int(torch.mps.current_allocated_memory())
            free_bytes = max(0, rec - cur)
        except RuntimeError:
            free_bytes = None

    if free_bytes is None or free_bytes <= 0:
        return {
            "max_chunk_bytes": int(DEFAULT_MAX_CHUNK_BYTES),
            "max_observe_bytes": int(DEFAULT_MAX_OBSERVE_BYTES),
        }

    budget = int(free_bytes * frac)
    chunk_bytes = min(int(max_chunk_cap), max(int(DEFAULT_MAX_CHUNK_BYTES), budget))
    observe_bytes = min(int(max_observe_cap), max(int(DEFAULT_MAX_OBSERVE_BYTES), budget))
    return {
        "max_chunk_bytes": chunk_bytes,
        "max_observe_bytes": observe_bytes,
    }


def make_tensor_dataloader(
    samples: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    batch_size: int | None = None,
    shuffle: bool = False,
    stage_on_device: bool = False,
    pin_memory: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader[Any]:
    """Build a standard PyTorch ``DataLoader`` usable identically with GMDH and dense models.

    When ``stage_on_device=True``, tensors are placed directly on ``device`` once up front so
    epoch loops incur zero Host-to-Device copies. When ``stage_on_device=False``, CPU tensors
    enable ``pin_memory`` automatically on CUDA for non-blocking transfers.
    """
    dev = torch.device(device)
    eff_dt = dtype if dtype is not None else torch.get_default_dtype()
    target_dev = dev if stage_on_device else torch.device("cpu")

    x_t = torch.as_tensor(samples, dtype=eff_dt).to(device=target_dev)
    y_t = torch.as_tensor(targets, dtype=eff_dt).to(device=target_dev)
    eff_bs = int(x_t.shape[0]) if (batch_size is None or int(batch_size) <= 0) else int(batch_size)
    use_pin = (not stage_on_device) and bool(pin_memory) and (dev.type == "cuda")
    eff_workers = 0 if stage_on_device else max(0, int(num_workers))

    return DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=eff_bs,
        shuffle=bool(shuffle),
        pin_memory=use_pin,
        num_workers=eff_workers,
        persistent_workers=(eff_workers > 0),
        drop_last=bool(drop_last),
    )
