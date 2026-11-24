"""Fixed-point quantization, bit-exact integer simulation, and closed-loop CartPole-v1 evaluation.

Supports both pruned affine GMDH checkpoints (``GMDHModel`` with ``inputs=2, degree=1, interaction=0``)
and matched dense MLP baselines (``nn.Sequential`` of ``nn.Linear`` + ``nn.ReLU`` / ``nn.Hardtanh``).
Outputs structured JSON metrics and ``vectors.hex`` golden test vectors for Verilog-2001 verification.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from torch import nn

_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from gmdhtorch import GMDHActivation, GMDHLayer, GMDHModel, GMDHNorm, GMDHOutput
from reinforcement import Reinforcement

RoundMode = Literal["nearest", "floor"]


def _signed_bounds(bits: int) -> tuple[int, int]:
    """Return inclusive two's-complement bounds ``(q_min, q_max)`` for ``bits``-bit signed integers."""
    q_min = -(1 << (bits - 1))
    q_max = (1 << (bits - 1)) - 1
    return q_min, q_max


def _quantize_float(value: float, *, frac_bits: int, q_min: int, q_max: int) -> int:
    """Quantize a floating-point scalar to a clamped signed fixed-point integer."""
    scaled = round(float(value) * float(1 << frac_bits))
    if scaled < q_min:
        return q_min
    if scaled > q_max:
        return q_max
    return scaled


def _shift_right(acc: int, *, frac_bits: int, round_mode: RoundMode) -> int:
    """Shift a wide signed accumulator right by ``frac_bits`` matching Verilog-2001 arithmetic shift."""
    if frac_bits <= 0:
        return acc
    if round_mode == "nearest":
        return (acc + (1 << (frac_bits - 1))) >> frac_bits
    if round_mode == "floor":
        return acc >> frac_bits
    raise ValueError(f"Unsupported round_mode {round_mode!r}; expected 'nearest' or 'floor'.")


def _min_signed_bits(max_abs_val: int) -> int:
    """Minimum signed two's-complement bit-width required to represent ``[-max_abs_val, +max_abs_val]``."""
    if max_abs_val <= 0:
        return 2
    return int(max_abs_val).bit_length() + 1


def _to_twos_complement(value: int, width: int) -> int:
    """Mask signed integer ``value`` into an unsigned ``width``-bit two's-complement bit pattern."""
    q_min, q_max = _signed_bounds(width)
    if value < q_min or value > q_max:
        raise OverflowError(f"Value {value} does not fit in signed {width}-bit range [{q_min}, {q_max}].")
    return value & ((1 << width) - 1)


@dataclass(frozen=True)
class ActivationSpec:
    """Quantized activation descriptor shared by GMDH and Dense layers."""

    kind: Literal["hardtanh", "relu"]
    min_val: float | None = None
    max_val: float | None = None
    min_q: int | None = None
    max_q: int | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "hardtanh":
            return {
                "type": "hardtanh",
                "min_val": self.min_val,
                "max_val": self.max_val,
                "min_q": self.min_q,
                "max_q": self.max_q,
            }
        return {"type": "relu"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActivationSpec:
        kind = str(data["type"])
        if kind == "hardtanh":
            return cls(
                kind="hardtanh",
                min_val=float(data["min_val"]),
                max_val=float(data["max_val"]),
                min_q=int(data["min_q"]),
                max_q=int(data["max_q"]),
            )
        if kind == "relu":
            return cls(kind="relu")
        raise ValueError(f"Unsupported serialized activation type {kind!r}.")


@dataclass(frozen=True)
class QuantizedGMDHLayer:
    """Quantized 2-input affine GMDH layer (``degree=1, interaction=0, forward=False``)."""

    layer_number: int
    idx0: list[list[int]]
    bias: list[int]
    weights: list[list[int]]
    activation: ActivationSpec | None

    @property
    def neuron_count(self) -> int:
        return len(self.bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "gmdh",
            "layer_number": self.layer_number,
            "neuron_count": self.neuron_count,
            "idx0": self.idx0,
            "bias": self.bias,
            "weights": self.weights,
            "activation": None if self.activation is None else self.activation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizedGMDHLayer:
        act = data.get("activation")
        return cls(
            layer_number=int(data["layer_number"]),
            idx0=[[int(i) for i in pair] for pair in data["idx0"]],
            bias=[int(b) for b in data["bias"]],
            weights=[[int(w) for w in row] for row in data["weights"]],
            activation=None if act is None else ActivationSpec.from_dict(act),
        )


@dataclass(frozen=True)
class QuantizedDenseLayer:
    """Quantized fully connected hidden layer."""

    layer_number: int
    in_features: int
    out_features: int
    bias: list[int]
    weights: list[list[int]]
    activation: ActivationSpec | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "dense",
            "layer_number": self.layer_number,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "weights": self.weights,
            "activation": None if self.activation is None else self.activation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizedDenseLayer:
        act = data.get("activation")
        return cls(
            layer_number=int(data["layer_number"]),
            in_features=int(data["in_features"]),
            out_features=int(data["out_features"]),
            bias=[int(b) for b in data["bias"]],
            weights=[[int(w) for w in row] for row in data["weights"]],
            activation=None if act is None else ActivationSpec.from_dict(act),
        )


@dataclass(frozen=True)
class QuantizedFoldedOutput:
    """Folded single-output linear head (``w_diff = w[1] - w[0]``, ``b_diff = b[1] - b[0]``)."""

    in_features: int
    bias: int
    weights: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "bias": self.bias,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizedFoldedOutput:
        return cls(
            in_features=int(data["in_features"]),
            bias=int(data["bias"]),
            weights=[int(w) for w in data["weights"]],
        )


class FixedPointModel:
    """Bit-exact integer fixed-point simulator matching Verilog-2001 signed Q-format arithmetic."""

    def __init__(
        self,
        *,
        model_type: Literal["gmdh", "dense"],
        data_bits: int,
        int_bits: int,
        round_mode: RoundMode,
        input_count: int,
        layers: list[QuantizedGMDHLayer | QuantizedDenseLayer],
        output: QuantizedFoldedOutput,
        raw_params: int,
        raw_mults: int,
        folded_params: int,
        folded_mults: int,
        source_checkpoint: str | None = None,
    ) -> None:
        if data_bits <= int_bits:
            raise ValueError(f"data_bits ({data_bits}) must be > int_bits ({int_bits}).")
        if int_bits < 2:
            raise ValueError(f"int_bits ({int_bits}) must be >= 2 for signed representation.")
        self.model_type: Literal["gmdh", "dense"] = model_type
        self.data_bits = int(data_bits)
        self.int_bits = int(int_bits)
        self.frac_bits = int(data_bits - int_bits)
        self.round_mode: RoundMode = round_mode
        self.input_count = int(input_count)
        self.layers = layers
        self.output = output
        self.raw_params = int(raw_params)
        self.raw_mults = int(raw_mults)
        self.folded_params = int(folded_params)
        self.folded_mults = int(folded_mults)
        self.source_checkpoint = source_checkpoint
        self.q_min, self.q_max = _signed_bounds(self.data_bits)

        max_fan_in = max(
            [self.output.in_features]
            + [2 if isinstance(layer, QuantizedGMDHLayer) else layer.in_features for layer in self.layers]
        )
        # Guaranteed overflow-free signed accumulator bit-width for every MAC across all layers
        self.acc_bits = int(2 * self.data_bits + max(3, (max_fan_in + 2).bit_length()))

    def quantize_inputs(self, state: Any) -> list[int]:
        """Quantize a 1-D observation vector into signed ``Q(int_bits).(frac_bits)`` integers."""
        return [
            _quantize_float(float(v), frac_bits=self.frac_bits, q_min=self.q_min, q_max=self.q_max)
            for v in state
        ]

    def _apply_activation_and_saturate(
        self,
        val: int,
        activation: ActivationSpec | None,
    ) -> tuple[int, int]:
        """Apply optional integer activation and signed saturation, returning ``(out_q, sat_count)``."""
        sat_count = 0
        if activation is not None:
            if activation.kind == "hardtanh":
                assert activation.min_q is not None and activation.max_q is not None
                if val < activation.min_q:
                    val = activation.min_q
                elif val > activation.max_q:
                    val = activation.max_q
            elif activation.kind == "relu":
                val = max(val, 0)
        if val < self.q_min:
            val = self.q_min
            sat_count = 1
        elif val > self.q_max:
            val = self.q_max
            sat_count = 1
        return val, sat_count

    def forward_int(self, x_q: list[int]) -> tuple[int, int, int, int]:
        """Execute bit-exact integer forward pass.

        Returns ``(delta_q, action, max_abs_acc, saturation_events)``.
        """
        h = list(x_q)
        max_abs_acc = 0
        sat_events = 0
        f_bits = self.frac_bits

        for layer in self.layers:
            next_h: list[int] = []
            if isinstance(layer, QuantizedGMDHLayer):
                for idx_pair, b_q, w_pair in zip(layer.idx0, layer.bias, layer.weights, strict=True):
                    acc = (b_q << f_bits) + w_pair[0] * h[idx_pair[0]] + w_pair[1] * h[idx_pair[1]]
                    max_abs_acc = max(max_abs_acc, abs(acc))
                    shifted = _shift_right(acc, frac_bits=f_bits, round_mode=self.round_mode)
                    out_q, sat = self._apply_activation_and_saturate(shifted, layer.activation)
                    sat_events += sat
                    next_h.append(out_q)
            else:
                for b_q, w_row in zip(layer.bias, layer.weights, strict=True):
                    acc = b_q << f_bits
                    for w_k, h_k in zip(w_row, h, strict=True):
                        acc += w_k * h_k
                    max_abs_acc = max(max_abs_acc, abs(acc))
                    shifted = _shift_right(acc, frac_bits=f_bits, round_mode=self.round_mode)
                    out_q, sat = self._apply_activation_and_saturate(shifted, layer.activation)
                    sat_events += sat
                    next_h.append(out_q)
            h = next_h

        delta_q = self.output.bias << f_bits
        for w_k, h_k in zip(self.output.weights, h, strict=True):
            delta_q += w_k * h_k
        max_abs_acc = max(max_abs_acc, abs(delta_q))

        action = 1 if delta_q > 0 else 0
        return delta_q, action, max_abs_acc, sat_events

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model_type": self.model_type,
            "source_checkpoint": self.source_checkpoint,
            "format": {
                "data_bits": self.data_bits,
                "int_bits": self.int_bits,
                "frac_bits": self.frac_bits,
                "acc_bits": self.acc_bits,
                "q_min": self.q_min,
                "q_max": self.q_max,
                "round_mode": self.round_mode,
            },
            "footprint": {
                "raw_params": self.raw_params,
                "raw_mults": self.raw_mults,
                "folded_params": self.folded_params,
                "folded_mults": self.folded_mults,
            },
            "input_count": self.input_count,
            "layers": [layer.to_dict() for layer in self.layers],
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixedPointModel:
        fmt = data["format"]
        fp = data["footprint"]
        layers: list[QuantizedGMDHLayer | QuantizedDenseLayer] = []
        for layer_cfg in data["layers"]:
            if layer_cfg["kind"] == "gmdh":
                layers.append(QuantizedGMDHLayer.from_dict(layer_cfg))
            elif layer_cfg["kind"] == "dense":
                layers.append(QuantizedDenseLayer.from_dict(layer_cfg))
            else:
                raise ValueError(f"Unsupported layer kind {layer_cfg['kind']!r}.")
        return cls(
            model_type=data["model_type"],
            data_bits=int(fmt["data_bits"]),
            int_bits=int(fmt["int_bits"]),
            round_mode=fmt["round_mode"],
            input_count=int(data["input_count"]),
            layers=layers,
            output=QuantizedFoldedOutput.from_dict(data["output"]),
            raw_params=int(fp["raw_params"]),
            raw_mults=int(fp["raw_mults"]),
            folded_params=int(fp["folded_params"]),
            folded_mults=int(fp["folded_mults"]),
            source_checkpoint=data.get("source_checkpoint"),
        )


def _extract_activation_spec(
    mod: nn.Module,
    *,
    frac_bits: int,
    q_min: int,
    q_max: int,
) -> ActivationSpec:
    """Validate and quantize a supported activation module (``Hardtanh`` or ``ReLU``)."""
    inner = mod.activation if isinstance(mod, GMDHActivation) else mod
    if isinstance(mod, GMDHActivation) and bool(mod.passthrough_mask.any().item()):
        raise ValueError("GMDHActivation has passthrough neurons enabled; HDL export requires forward=False.")
    if isinstance(inner, nn.Hardtanh):
        min_val = float(inner.min_val)
        max_val = float(inner.max_val)
        return ActivationSpec(
            kind="hardtanh",
            min_val=min_val,
            max_val=max_val,
            min_q=_quantize_float(min_val, frac_bits=frac_bits, q_min=q_min, q_max=q_max),
            max_q=_quantize_float(max_val, frac_bits=frac_bits, q_min=q_min, q_max=q_max),
        )
    if isinstance(inner, nn.ReLU):
        return ActivationSpec(kind="relu")
    raise ValueError(f"Unsupported activation {type(inner).__name__!r}; expected Hardtanh or ReLU.")


def quantize_model(
    q_network: nn.Module,
    *,
    data_bits: int,
    int_bits: int = 5,
    round_mode: RoundMode = "nearest",
    source_checkpoint: str | None = None,
) -> FixedPointModel:
    """Quantize a trained ``GMDHModel`` or dense ``nn.Sequential`` Q-network into ``FixedPointModel``."""
    if data_bits <= int_bits:
        raise ValueError(f"data_bits ({data_bits}) must be > int_bits ({int_bits}).")
    frac_bits = data_bits - int_bits
    q_min, q_max = _signed_bounds(data_bits)

    if isinstance(q_network, GMDHModel):
        return _quantize_gmdh_model(
            q_network,
            data_bits=data_bits,
            int_bits=int_bits,
            frac_bits=frac_bits,
            q_min=q_min,
            q_max=q_max,
            round_mode=round_mode,
            source_checkpoint=source_checkpoint,
        )
    if isinstance(q_network, nn.Sequential):
        return _quantize_dense_model(
            q_network,
            data_bits=data_bits,
            int_bits=int_bits,
            frac_bits=frac_bits,
            q_min=q_min,
            q_max=q_max,
            round_mode=round_mode,
            source_checkpoint=source_checkpoint,
        )
    raise TypeError(f"Unsupported Q-network type {type(q_network).__name__!r}.")


def _quantize_gmdh_model(
    model: GMDHModel,
    *,
    data_bits: int,
    int_bits: int,
    frac_bits: int,
    q_min: int,
    q_max: int,
    round_mode: RoundMode,
    source_checkpoint: str | None,
) -> FixedPointModel:
    layers: list[QuantizedGMDHLayer | QuantizedDenseLayer] = []
    output_mod: GMDHOutput | None = None
    modules = list(model)
    idx = 0
    raw_params = 0
    raw_mults = 0

    while idx < len(modules):
        mod = modules[idx]
        if isinstance(mod, GMDHNorm):
            raise TypeError("GMDHNorm is not supported by this quantization/HDL pipeline (train with norm=False).")
        if isinstance(mod, GMDHLayer):
            if mod.input_count != 2 or mod.degree != 1 or mod.interaction != 0:
                raise ValueError(
                    f"Layer {mod.layer_number} has (inputs={mod.input_count}, degree={mod.degree}, "
                    f"interaction={mod.interaction}); expected (2, 1, 0)."
                )
            if mod.forward_inputs or bool(mod.lock_mask.any().item()):
                raise ValueError(f"Layer {mod.layer_number} uses forwarding/passthrough; expected forward=False.")

            act_spec: ActivationSpec | None = None
            if idx + 1 < len(modules) and isinstance(modules[idx + 1], (GMDHActivation, nn.Hardtanh, nn.ReLU)):
                act_spec = _extract_activation_spec(modules[idx + 1], frac_bits=frac_bits, q_min=q_min, q_max=q_max)
                idx += 1

            W = mod.W.detach().cpu().to(torch.float64)
            idx0_list = [[int(v) for v in row] for row in mod.idx0.detach().cpu().tolist()]
            bias_q = [_quantize_float(float(W[n, 0].item()), frac_bits=frac_bits, q_min=q_min, q_max=q_max) for n in range(mod.neuron_count)]
            weights_q = [
                [
                    _quantize_float(float(W[n, 1].item()), frac_bits=frac_bits, q_min=q_min, q_max=q_max),
                    _quantize_float(float(W[n, 2].item()), frac_bits=frac_bits, q_min=q_min, q_max=q_max),
                ]
                for n in range(mod.neuron_count)
            ]
            layers.append(
                QuantizedGMDHLayer(
                    layer_number=int(mod.layer_number),
                    idx0=idx0_list,
                    bias=bias_q,
                    weights=weights_q,
                    activation=act_spec,
                )
            )
            raw_params += mod.neuron_count * 3
            raw_mults += mod.neuron_count * 2
        elif isinstance(mod, GMDHOutput):
            if output_mod is not None:
                raise ValueError("Multiple GMDHOutput modules found.")
            output_mod = mod
        else:
            raise TypeError(f"Unsupported module {type(mod).__name__!r} inside GMDHModel.")
        idx += 1

    if output_mod is None or not output_mod.dense or output_mod.linear is None:
        raise ValueError("Expected a dense GMDHOutput(count=2, dense=True) head.")
    lin = output_mod.linear
    if lin.out_features != 2:
        raise ValueError(f"Expected 2-action output head for folding, got out_features={lin.out_features}.")

    w_diff = (lin.weight[1] - lin.weight[0]).detach().cpu().to(torch.float64)
    b_diff = float((lin.bias[1] - lin.bias[0]).detach().cpu().to(torch.float64).item()) if lin.bias is not None else 0.0
    folded_out = QuantizedFoldedOutput(
        in_features=int(lin.in_features),
        bias=_quantize_float(b_diff, frac_bits=frac_bits, q_min=q_min, q_max=q_max),
        weights=[_quantize_float(float(v.item()), frac_bits=frac_bits, q_min=q_min, q_max=q_max) for v in w_diff],
    )
    out_raw_params = lin.in_features * 2 + (2 if lin.bias is not None else 0)
    out_raw_mults = lin.in_features * 2
    folded_params = raw_params + lin.in_features + 1
    folded_mults = raw_mults + lin.in_features

    return FixedPointModel(
        model_type="gmdh",
        data_bits=data_bits,
        int_bits=int_bits,
        round_mode=round_mode,
        input_count=int(model.input_count),
        layers=layers,
        output=folded_out,
        raw_params=raw_params + out_raw_params,
        raw_mults=raw_mults + out_raw_mults,
        folded_params=folded_params,
        folded_mults=folded_mults,
        source_checkpoint=source_checkpoint,
    )


def _quantize_dense_model(
    model: nn.Sequential,
    *,
    data_bits: int,
    int_bits: int,
    frac_bits: int,
    q_min: int,
    q_max: int,
    round_mode: RoundMode,
    source_checkpoint: str | None,
) -> FixedPointModel:
    modules = list(model)
    linear_indices = [i for i, m in enumerate(modules) if isinstance(m, nn.Linear)]
    if len(linear_indices) < 2:
        raise ValueError("Expected a multi-layer dense nn.Sequential model ending in an nn.Linear output head.")

    last_linear_idx = linear_indices[-1]
    if last_linear_idx != len(modules) - 1:
        raise ValueError("Dense Q-network must end with a linear output layer (no activation after output).")

    layers: list[QuantizedGMDHLayer | QuantizedDenseLayer] = []
    raw_params = 0
    raw_mults = 0
    layer_no = 0
    idx = 0
    first_in_features = int(modules[linear_indices[0]].in_features)  # type: ignore[union-attr]

    while idx < last_linear_idx:
        mod = modules[idx]
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"Unexpected module {type(mod).__name__!r} at index {idx} in dense nn.Sequential.")
        layer_no += 1
        act_spec: ActivationSpec | None = None
        if idx + 1 < last_linear_idx and isinstance(modules[idx + 1], (nn.ReLU, nn.Hardtanh)):
            act_spec = _extract_activation_spec(modules[idx + 1], frac_bits=frac_bits, q_min=q_min, q_max=q_max)
            idx += 1

        W = mod.weight.detach().cpu().to(torch.float64)
        b = mod.bias.detach().cpu().to(torch.float64) if mod.bias is not None else torch.zeros(mod.out_features, dtype=torch.float64)
        bias_q = [_quantize_float(float(b[r].item()), frac_bits=frac_bits, q_min=q_min, q_max=q_max) for r in range(mod.out_features)]
        weights_q = [
            [_quantize_float(float(W[r, c].item()), frac_bits=frac_bits, q_min=q_min, q_max=q_max) for c in range(mod.in_features)]
            for r in range(mod.out_features)
        ]
        layers.append(
            QuantizedDenseLayer(
                layer_number=layer_no,
                in_features=int(mod.in_features),
                out_features=int(mod.out_features),
                bias=bias_q,
                weights=weights_q,
                activation=act_spec,
            )
        )
        raw_params += mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0)
        raw_mults += mod.in_features * mod.out_features
        idx += 1

    out_lin = modules[last_linear_idx]
    assert isinstance(out_lin, nn.Linear)
    if out_lin.out_features != 2:
        raise ValueError(f"Expected 2-action output head for folding, got out_features={out_lin.out_features}.")

    w_diff = (out_lin.weight[1] - out_lin.weight[0]).detach().cpu().to(torch.float64)
    b_diff = float((out_lin.bias[1] - out_lin.bias[0]).detach().cpu().to(torch.float64).item()) if out_lin.bias is not None else 0.0
    folded_out = QuantizedFoldedOutput(
        in_features=int(out_lin.in_features),
        bias=_quantize_float(b_diff, frac_bits=frac_bits, q_min=q_min, q_max=q_max),
        weights=[_quantize_float(float(v.item()), frac_bits=frac_bits, q_min=q_min, q_max=q_max) for v in w_diff],
    )
    out_raw_params = out_lin.in_features * 2 + (2 if out_lin.bias is not None else 0)
    out_raw_mults = out_lin.in_features * 2
    folded_params = raw_params + out_lin.in_features + 1
    folded_mults = raw_mults + out_lin.in_features

    return FixedPointModel(
        model_type="dense",
        data_bits=data_bits,
        int_bits=int_bits,
        round_mode=round_mode,
        input_count=first_in_features,
        layers=layers,
        output=folded_out,
        raw_params=raw_params + out_raw_params,
        raw_mults=raw_mults + out_raw_mults,
        folded_params=folded_params,
        folded_mults=folded_mults,
        source_checkpoint=source_checkpoint,
    )


def evaluate_float_policy(
    q_network: nn.Module,
    *,
    dtype: torch.dtype,
    env_id: str = "CartPole-v1",
    episodes: int = 100,
    seed: int = 0,
    render: bool = False,
) -> dict[str, Any]:
    """Evaluate the floating-point Q-network closed-loop in Gymnasium."""
    q_network.eval()
    env = gym.make(env_id, render_mode="human" if render else None)
    steps_per_ep: list[int] = []
    try:
        for ep in range(episodes):
            state, _ = env.reset(seed=seed + ep)
            done = False
            steps = 0
            while not done:
                state_t = torch.as_tensor(state, dtype=dtype).unsqueeze(0)
                with torch.no_grad():
                    action = int(q_network(state_t).argmax(dim=-1).item())
                state, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                steps += 1
            steps_per_ep.append(steps)
    finally:
        env.close()

    arr = np.asarray(steps_per_ep, dtype=np.float64)
    solved = int(np.sum(arr >= 500))
    return {
        "episodes": int(episodes),
        "seed": int(seed),
        "mean_steps": float(np.mean(arr)),
        "std_steps": float(np.std(arr)),
        "min_steps": int(np.min(arr)),
        "max_steps": int(np.max(arr)),
        "solved_500_count": solved,
        "solved_500_rate": float(solved / episodes),
        "steps_per_episode": steps_per_ep,
    }


def evaluate_fixed_point_policy(
    fp_model: FixedPointModel,
    *,
    float_ref: nn.Module,
    float_dtype: torch.dtype,
    env_id: str = "CartPole-v1",
    episodes: int = 100,
    seed: int = 0,
    render: bool = False,
    record_vectors: int = 0,
) -> tuple[dict[str, Any], list[tuple[list[int], int, int]]]:
    """Evaluate ``FixedPointModel`` closed-loop in Gymnasium and optionally record golden RTL vectors."""
    float_ref.eval()
    env = gym.make(env_id, render_mode="human" if render else None)
    steps_per_ep: list[int] = []
    total_steps = 0
    action_matches = 0
    max_abs_acc = 0
    total_sat_events = 0
    recorded: list[tuple[list[int], int, int]] = []

    try:
        for ep in range(episodes):
            state, _ = env.reset(seed=seed + ep)
            done = False
            steps = 0
            while not done:
                x_q = fp_model.quantize_inputs(state)
                delta_q, action, step_max_acc, step_sat = fp_model.forward_int(x_q)
                max_abs_acc = max(max_abs_acc, step_max_acc)
                total_sat_events += step_sat

                state_t = torch.as_tensor(state, dtype=float_dtype).unsqueeze(0)
                with torch.no_grad():
                    ref_action = int(float_ref(state_t).argmax(dim=-1).item())
                if action == ref_action:
                    action_matches += 1
                total_steps += 1

                if record_vectors > 0 and len(recorded) < record_vectors:
                    recorded.append((x_q, delta_q, action))

                state, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                steps += 1
            steps_per_ep.append(steps)
    finally:
        env.close()

    arr = np.asarray(steps_per_ep, dtype=np.float64)
    solved = int(np.sum(arr >= 500))
    metrics = {
        "data_bits": fp_model.data_bits,
        "int_bits": fp_model.int_bits,
        "frac_bits": fp_model.frac_bits,
        "acc_bits_allocated": fp_model.acc_bits,
        "round_mode": fp_model.round_mode,
        "episodes": int(episodes),
        "seed": int(seed),
        "mean_steps": float(np.mean(arr)),
        "std_steps": float(np.std(arr)),
        "min_steps": int(np.min(arr)),
        "max_steps": int(np.max(arr)),
        "solved_500_count": solved,
        "solved_500_rate": float(solved / episodes),
        "total_steps": int(total_steps),
        "action_agreement_rate": float(action_matches / total_steps) if total_steps > 0 else 0.0,
        "max_abs_acc": int(max_abs_acc),
        "min_acc_bits_observed": _min_signed_bits(max_abs_acc),
        "saturation_events": int(total_sat_events),
        "steps_per_episode": steps_per_ep,
    }
    return metrics, recorded


def _build_corner_vectors(fp_model: FixedPointModel) -> list[tuple[list[int], int, int]]:
    """Generate deterministic edge-case vectors (zeros, extremes, saturation boundaries)."""
    q_min, q_max = fp_model.q_min, fp_model.q_max
    one = 1 << fp_model.frac_bits
    half = max(1, one >> 1)
    n = fp_model.input_count
    corner_inputs: list[list[int]] = [
        [0] * n,
        [q_max] * n,
        [q_min] * n,
        [q_max if i % 2 == 0 else q_min for i in range(n)],
        [q_min if i % 2 == 0 else q_max for i in range(n)],
        [one] * n,
        [-one] * n,
        [half, -half, one, -one][:n],
        [-half, half, -one, one][:n],
    ]
    out: list[tuple[list[int], int, int]] = []
    for x_q in corner_inputs:
        delta_q, action, _, _ = fp_model.forward_int(x_q)
        out.append((x_q, delta_q, action))
    return out


def export_vectors_hex(
    fp_model: FixedPointModel,
    vectors: list[tuple[list[int], int, int]],
    *,
    hex_path: Path,
    meta_path: Path,
) -> None:
    """Write golden RTL verification vectors as fixed-width two's-complement hex lines + layout JSON."""
    data_bits = fp_model.data_bits
    acc_bits = fp_model.acc_bits
    n_in = fp_model.input_count
    total_bits = n_in * data_bits + acc_bits + 1
    hex_chars = math.ceil(total_bits / 4)

    # Layout MSB -> LSB: [x_0 | x_1 | ... | x_{n-1} | delta_q | action]
    fields: list[dict[str, Any]] = []
    cursor = total_bits
    for i in range(n_in):
        msb = cursor - 1
        lsb = cursor - data_bits
        fields.append({"name": f"in_{i}", "msb": msb, "lsb": lsb, "width": data_bits, "signed": True})
        cursor = lsb
    fields.append({"name": "delta_q", "msb": cursor - 1, "lsb": cursor - acc_bits, "width": acc_bits, "signed": True})
    cursor -= acc_bits
    fields.append({"name": "action", "msb": 0, "lsb": 0, "width": 1, "signed": False})

    lines: list[str] = []
    for x_q, delta_q, action in vectors:
        packed = 0
        for v in x_q:
            packed = (packed << data_bits) | _to_twos_complement(v, data_bits)
        packed = (packed << acc_bits) | _to_twos_complement(delta_q, acc_bits)
        packed = (packed << 1) | (int(action) & 1)
        lines.append(f"{packed:0{hex_chars}x}")

    hex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "version": 1,
        "num_vectors": len(vectors),
        "total_bits": total_bits,
        "hex_chars": hex_chars,
        "data_bits": data_bits,
        "acc_bits": acc_bits,
        "fields": fields,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _add_common_quant_args(subparser: argparse.ArgumentParser, *, default_episodes: int) -> None:
    subparser.add_argument("--int-bits", type=int, default=5, help="Signed integer bits including sign bit (default=5 for [-16, 16))")
    subparser.add_argument("--round-mode", type=str, choices=["nearest", "floor"], default="nearest", help="Right-shift rounding mode (default=nearest)")
    subparser.add_argument("--episodes", type=int, default=default_episodes, help=f"Number of deterministic CartPole-v1 episodes (default={default_episodes})")
    subparser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic episode resets (default=0)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-point quantization, closed-loop CartPole-v1 evaluation, and RTL vector export.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sweep_p = sub.add_parser("sweep", help="Benchmark a list of bit-widths and save sweep_results.json into output_dir")
    sweep_p.add_argument("checkpoint", type=str, help="Path to a saved Reinforcement checkpoint (.pth)")
    sweep_p.add_argument("output_dir", type=str, help="Output directory where sweep_results.json will be saved")
    sweep_p.add_argument("--bits", type=int, nargs="+", default=[8, 9, 10, 11, 12, 14, 16], help="Bit-widths to evaluate (default=8 9 10 11 12 14 16)")
    _add_common_quant_args(sweep_p, default_episodes=100)

    export_p = sub.add_parser("export", help="Quantize a single bit-width and export quantized_model.json + vectors.hex into output_dir")
    export_p.add_argument("checkpoint", type=str, help="Path to a saved Reinforcement checkpoint (.pth)")
    export_p.add_argument("output_dir", type=str, help="Output directory for quantized_model.json, vectors.hex, vectors_meta.json, and eval_result.json")
    export_p.add_argument("--bits", type=int, default=12, help="Fixed-point total bit-width (default=12)")
    export_p.add_argument("--num-vectors", type=int, default=500, help="Number of rollout vectors to include in vectors.hex (default=500)")
    _add_common_quant_args(export_p, default_episodes=10)

    render_p = sub.add_parser("render", help="Visually evaluate a quantized controller in CartPole-v1 (writes no files)")
    render_p.add_argument("checkpoint", type=str, help="Path to a saved Reinforcement checkpoint (.pth)")
    render_p.add_argument("--bits", type=int, default=12, help="Fixed-point total bit-width (default=12)")
    _add_common_quant_args(render_p, default_episodes=3)

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be > 0.")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        parser.error(f"Checkpoint not found: {ckpt_path}")

    bit_list = list(args.bits) if args.command == "sweep" else [int(args.bits)]
    is_render = args.command == "render"

    agent = Reinforcement.from_checkpoint(str(ckpt_path), device="cpu")
    try:
        q_network = agent._q_network
        float_dtype = agent.dtype
        env_id = agent._env_id

        first_fp = quantize_model(
            q_network,
            data_bits=bit_list[0],
            int_bits=args.int_bits,
            round_mode=args.round_mode,
            source_checkpoint=str(ckpt_path),
        )
        print(
            f"Loaded {first_fp.model_type} model from {ckpt_path} "
            f"(folded: {first_fp.folded_params} params, {first_fp.folded_mults} mults)"
        )

        float_metrics: dict[str, Any] | None = None
        if not is_render:
            float_metrics = evaluate_float_policy(
                q_network,
                dtype=float_dtype,
                env_id=env_id,
                episodes=args.episodes,
                seed=args.seed,
                render=False,
            )
            print(
                f"Float32 baseline ({args.episodes} eps): "
                f"mean={float_metrics['mean_steps']:.2f}, min={float_metrics['min_steps']}, "
                f"solved={float_metrics['solved_500_count']}/{args.episodes}"
            )

        quant_results: list[dict[str, Any]] = []
        exported_fp: FixedPointModel | None = None
        recorded_vectors: list[tuple[list[int], int, int]] = []

        for bits in bit_list:
            fp_model = quantize_model(
                q_network,
                data_bits=bits,
                int_bits=args.int_bits,
                round_mode=args.round_mode,
                source_checkpoint=str(ckpt_path),
            )
            record_count = int(args.num_vectors) if args.command == "export" else 0
            q_metrics, rollout_vecs = evaluate_fixed_point_policy(
                fp_model,
                float_ref=q_network,
                float_dtype=float_dtype,
                env_id=env_id,
                episodes=args.episodes,
                seed=args.seed,
                render=is_render,
                record_vectors=record_count,
            )
            quant_results.append(q_metrics)
            if args.command == "export":
                exported_fp = fp_model
                recorded_vectors = rollout_vecs
            print(
                f"Q{fp_model.int_bits}.{fp_model.frac_bits} ({bits:2d}b, {args.round_mode}): "
                f"mean={q_metrics['mean_steps']:6.2f}, min={q_metrics['min_steps']:3d}, "
                f"solved={q_metrics['solved_500_count']:3d}/{args.episodes}, "
                f"agree={q_metrics['action_agreement_rate'] * 100:6.2f}%, "
                f"acc_bits_obs={q_metrics['min_acc_bits_observed']:2d}, "
                f"sat={q_metrics['saturation_events']}"
            )

        if is_render:
            return

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_payload: dict[str, Any] = {
            "version": 1,
            "checkpoint": str(ckpt_path),
            "env_id": env_id,
            "model_type": first_fp.model_type,
            "footprint": {
                "raw_params": first_fp.raw_params,
                "raw_mults": first_fp.raw_mults,
                "folded_params": first_fp.folded_params,
                "folded_mults": first_fp.folded_mults,
            },
            "float32": float_metrics,
            "quantized": quant_results,
        }

        if args.command == "sweep":
            sweep_path = output_dir / "sweep_results.json"
            sweep_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
            print(f"Saved sweep results to {sweep_path}")
        elif args.command == "export" and exported_fp is not None:
            model_json_path = output_dir / "quantized_model.json"
            model_json_path.write_text(json.dumps(exported_fp.to_dict(), indent=2) + "\n", encoding="utf-8")

            all_vectors = _build_corner_vectors(exported_fp) + recorded_vectors
            export_vectors_hex(
                exported_fp,
                all_vectors,
                hex_path=output_dir / "vectors.hex",
                meta_path=output_dir / "vectors_meta.json",
            )
            eval_json_path = output_dir / "eval_result.json"
            eval_json_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
            print(f"Exported quantized model and {len(all_vectors)} vectors to {output_dir}")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
