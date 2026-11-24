# Cross-Platform Fixed-Point Quantization & Verilog-2001 HDL Workflow

Reproducible Python → Verilog-2001 pipeline for comparing pruned GMDH (`../gmdh.pth`) and matched compact dense (`../dense.pth`) CartPole-v1 controllers on **macOS, Linux, and Windows**. All commands below assume they are executed from this directory.

For a beginner-friendly guide to each open-source EDA tool (`iverilog`, `verilator`, `yosys`, `nextpnr`, `cocotb`) and how to run projects manually, see **[TOOLS_OVERVIEW.md](TOOLS_OVERVIEW.md)**.

---

## File Overview

- `quantize.py`: Bit-exact fixed-point (`Q(I).(F)`) integer simulator, closed-loop `CartPole-v1` sweep (with optional `--render`), structured JSON metrics, and `$readmemh` golden vector (`vectors.hex`) export.
- `rtl/gmdh_cell.v`: Reusable 2-input affine GMDH Processing Element (Verilog-2001).
- `export_verilog.py`: Generates pipelined Verilog-2001 top-level RTL (`<model>_cartpole.v`) and a self-checking testbench (`tb_<model>_cartpole.v`).
- `run_flow.py`: Cross-platform driver for standalone RTL simulation (`iverilog` / `verilator`) and Yosys FPGA synthesis (`synth_results.json`).
- `test_cartpole_cocotb.py`: Closed-loop `CartPole-v1` RTL-in-the-loop testbench via `cocotb` (supports `--render`).
- `TOOLS_OVERVIEW.md`: Introduction to the open-source EDA tools, manual workflows, and `cocotb` architecture.

---

## 1. Fixed-Point Quantization & Export (Python only)

```bash
# Sweep bit widths over 100 deterministic CartPole-v1 episodes (saves sweep_results.json in model dir)
python quantize.py sweep ../gmdh.pth build/gmdh --bits 8 9 10 11 12 14 16 --episodes 100
python quantize.py sweep ../dense.pth build/dense --bits 8 9 10 11 12 14 16 --episodes 100

# Export a chosen bit width into a bit-width subfolder (quantized_model.json + vectors.hex + vectors_meta.json + eval_result.json)
python quantize.py export ../gmdh.pth build/gmdh/12b --bits 12 --episodes 10
python quantize.py export ../dense.pth build/dense/12b --bits 12 --episodes 10

# Optional: visually inspect the quantized controller balancing the pole (writes no files)
python quantize.py render ../gmdh.pth --bits 10 --episodes 3
```

*(See [Section 4](#4-quantization-concepts--generated-files) below for details on the fixed-point math, `sweep`, `export`, and the meaning of each generated file.)*

---

## 2. Generate Verilog-2001 RTL & Self-Checking Testbench (Python only)

```bash
python export_verilog.py build/gmdh/12b
python export_verilog.py build/dense/12b
```

This writes `<model>_cartpole.v` (pipelined Verilog-2001 RTL), `tb_<model>_cartpole.v` (standalone `$readmemh` testbench), and `hdl_manifest.json` (used by `run_flow.py` and `test_cartpole_cocotb.py`) into each output directory. For GMDH models, `export_verilog.py` also copies `rtl/gmdh_cell.v` into the output directory so the entire folder is self-contained and directly exportable (e.g., to Vivado, Quartus, or another machine).

---

## 3. Simulation & FPGA Synthesis (macOS / Linux / Windows)

### Installing the EDA Tools: OSS CAD Suite vs Manual Install

The easiest cross-platform way to get all open-source EDA tools is the **YosysHQ OSS CAD Suite**:
- **Download**: <https://github.com/YosysHQ/oss-cad-suite-build/releases> (prebuilt portable archives for **macOS arm64/x64**, **Linux x64/arm64**, and **Windows x64**; unpack and add its `bin/` directory to `PATH`).

Alternatively, on macOS/Linux you can install individual tools manually via Homebrew (`brew install icarus-verilog verilator yosys`) or `apt`.

**Do you lose anything by installing manually instead of using OSS CAD Suite?**
- **For everything currently run in this repository (`iverilog`, `verilator`, `yosys`)**: **No.** Manual installs give you the exact same Verilog simulation and Yosys FPGA synthesis (`synth_xilinx`, `synth_ecp5`, `synth_ice40`, `synth_gowin`) results.
- **What OSS CAD Suite provides that manual installs often lack**:
  1. **Place-and-route & bitstream tools bundled out of the box**: `nextpnr-ecp5`, `nextpnr-ice40`, `nextpnr-himbaechel` (Gowin), `icestorm`, `prjtrellis`, `openFPGALoader`, `ghdl`, `sby` (formal verification), and `gtkwave`—several of which are not packaged in standard Homebrew/apt repositories or require lengthy source builds.
  2. **Pinned, mutually tested versions**: Every tool in a dated OSS CAD Suite build is compiled and tested together, avoiding version-skew issues between Yosys JSON netlists and `nextpnr` chip databases (helpful for exact paper reproducibility).
  3. **Zero-friction Windows support**: Native Windows lacks `brew`/`apt`, making the OSS CAD Suite archive by far the simplest way for Windows users to reproduce simulations and synthesis.

### Running Simulation & Synthesis

```bash
# Standalone Verilog-2001 testbench (bit-exact check against vectors.hex)
python run_flow.py simulate build/gmdh/12b --sim iverilog
python run_flow.py simulate build/dense/12b --sim iverilog

# Or using Verilator (compiles Verilog to C++ — very fast)
python run_flow.py simulate build/gmdh/12b --sim verilator
python run_flow.py simulate build/dense/12b --sim verilator

# Yosys FPGA synthesis resource statistics (saves synth_results.json)
python run_flow.py synth build/gmdh/12b --targets xilinx ecp5 ice40 gowin
python run_flow.py synth build/dense/12b --targets xilinx ecp5 ice40 gowin

# Closed-loop CartPole-v1 RTL verification (requires `pip install cocotb`)
python test_cartpole_cocotb.py build/gmdh/12b --episodes 10 --sim icarus
python test_cartpole_cocotb.py build/gmdh/12b --episodes 3 --sim icarus --render
```

---

## 4. Quantization Concepts & Generated Files

### Fixed-Point Representation (`Q(I).(F)`) & Output Folding

1. **Signed Fixed-Point Format (`Q(I).(F)`)**:
   - Every floating-point input, weight, bias, and hidden activation is quantized to a `W`-bit signed two's-complement integer with `I` integer bits (including the sign bit, default `I = 5` covering `[-16, +16)`) and `F = W - I` fractional bits:

```math
q = \text{clamp}\!\left(\text{round}(v \cdot 2^F),\; -2^{W-1},\; 2^{W-1} - 1\right)
```

   - Inside each neuron, multiplications and additions run in a wide, overflow-free signed accumulator (`ACC_W` bits):

```math
\text{acc} = (b_q \ll F) + \sum_k w_{q,k} \cdot x_{q,k}
```

   - The accumulator is shifted right by `F` (`nearest` adds $2^{F-1}$ before arithmetic right-shift `>> F`; `floor` applies plain `>> F`), then clamped by the integer activation (`Hardtanh` to `[min_q, max_q]` or `ReLU` to `[0, q_max]`) and saturated to `W` bits.
   - Python's `FixedPointModel` executes these **exact integer operations**, guaranteeing 100% bit-for-bit agreement with the generated Verilog-2001 hardware.

2. **Output Head Folding (2 Q-Values → 1 Signed Score)**:
   - Because CartPole-v1 has two discrete actions (`{0, 1}`), the policy only depends on the sign of $Q(s, 1) - Q(s, 0)$.
   - `quantize.py` folds the 2-output linear head into a single vector `w_diff = w[1] - w[0]` and bias `b_diff = b[1] - b[0]`, cutting output-layer multipliers in half.
   - The final stage outputs the wide integer score `delta_q` and `action = 1 if delta_q > 0 else 0`.

### The `sweep` Operation (Finding the Minimum Lossless Bit-Width)

`quantize.py sweep <checkpoint> <output_dir> [--bits W1 W2 ...]` benchmarks quantization robustness across a range of total bit-widths on identical deterministic `CartPole-v1` episodes (`seed, seed+1, ..., seed+episodes-1`):

1. Runs the original **float32** checkpoint closed-loop across all `episodes` to establish the reference baseline.
2. For each bit-width `W` in `--bits`, quantizes the model to `Q(I).(W-I)` and runs the same `episodes` closed-loop using `FixedPointModel`:
   - Records episode lengths (`mean_steps`, `std_steps`, `min_steps`, `max_steps`, `solved_500_count`, `solved_500_rate`).
   - Queries the float32 reference model on every visited state to measure `action_agreement_rate`.
   - Tracks `max_abs_acc`, `min_acc_bits_observed` (minimum signed accumulator width actually needed), and `saturation_events` (non-activation overflows).
3. Saves all baseline and swept metrics into `<output_dir>/sweep_results.json` for downstream analysis or LaTeX/poster table scripts.

### The `export` Operation (Exporting for HDL)

Once a target bit-width is selected, `quantize.py export <checkpoint> <output_dir> --bits <W>` evaluates that single fixed-point model closed-loop, records real visited rollout states, and writes **four files** into `<output_dir>`:

| File | Purpose & Contents |
| --- | --- |
| `quantized_model.json` | Complete fixed-point intermediate representation (IR): Q-format (`data_bits`, `int_bits`, `frac_bits`, `acc_bits`, `round_mode`), parameter/multiplier footprint, layer-by-layer integer weights/biases, GMDH `idx0` input indices (or Dense shapes), quantized activation clamp bounds, and folded output weights. Read by `export_verilog.py`, `run_flow.py`, and `test_cartpole_cocotb.py`. |
| `vectors.hex` | Golden verification vectors formatted as one fixed-width two's-complement hexadecimal word per line (MSB→LSB: `in_0 \| in_1 \| in_2 \| in_3 \| delta_q \| action`), ready for Verilog's `$readmemh`. Combines **9 synthetic corner cases** (zeros, min/max signed bounds, alternating extremes, unit values) with `--num-vectors` (default 500) real closed-loop CartPole rollout states. |
| `vectors_meta.json` | Bit-layout manifest for `vectors.hex` (`num_vectors`, `total_bits`, `hex_chars`, and exact `msb`/`lsb`/`width`/`signed` for each field) so `export_verilog.py` generates exact bit-slices in the Verilog testbench. |
| `eval_result.json` | Structured JSON closed-loop evaluation report for the exported bit-width alongside the float32 baseline. |
| `hdl_manifest.json` | *(Written by `export_verilog.py`)* Records `module_name`, `model_type`, `rtl_files`, and `tb_file` inside `<output_dir>` so `run_flow.py` and `test_cartpole_cocotb.py` validate that HDL generation has been executed and locate all sources automatically. |

---

## 5. Known Limitations

While **the number of hidden layers (`L >= 1`)** and **the number of network inputs (`input_count`, i.e., observation dimension)** are handled dynamically by `quantize.py` and `export_verilog.py`, the current HDL pipeline enforces the following architectural constraints:

- **GMDH neuron basis (`inputs=2, degree=1, interaction=0`)**: Each `GMDHLayer` must use 2-input affine cells without higher powers (`degree >= 2`) or cross-product terms (`interaction >= 2`).
- **No forwarding or normalization**: Passthrough forwarders (`forward=True` / `lock_mask`) and `GMDHNorm` layers are not supported.
- **Supported activations**: Only `nn.Hardtanh`, `nn.ReLU`, and linear (`None`) hidden activations are supported. Other activations (`SoftClip`, `Tanh`, `Sigmoid`, `LeakyReLU`, `GELU`, `SiLU`, `ELU`) are rejected.
- **2-action dense output head**: Only 2-output dense heads (`GMDHOutput(count=2, dense=True)` or final `nn.Linear(..., 2)`) folded into a 1-bit `action` are supported; non-dense selector heads (`dense=False`), multi-action (`> 2` actions), and continuous action spaces are not yet supported.
- **Uniform fixed-point format**: All layers, weights, biases, and activations share a single uniform `Q(I).(F)` format (no per-layer mixed precision).
- **RTL architecture**: Only fully unrolled, 1-stage-per-layer pipelined RTL is generated (no combinational mode or time-multiplexed PE sharing for larger models).
- **Environment assumption in `cocotb`**: `quantize.py` reads `env_id` from the checkpoint, whereas `test_cartpole_cocotb.py` currently hardcodes `CartPole-v1` and 1-bit `action` outputs.

---

## 6. Future Work

Brief roadmap of changes needed to remove the limitations above and generalize beyond `CartPole-v1`:

- **Richer GMDH basis (`inputs > 2`, `degree >= 2`, `interaction >= 2`)**: Extend `FixedPointModel` and parameterize `gmdh_cell.v` (or add dedicated polynomial/bilinear PE variants) to compute optional $x_i^2$ and $x_i x_j$ products with proper fractional rescaling.
- **Forwarding & `GMDHNorm` folding**: Map locked forwarders to direct bypass wires in `export_verilog.py`, and fold frozen `GMDHNorm` mean/std/affine statistics into the preceding or following layer weights during quantization (zero extra hardware cost).
- **Additional activations**: Add shift-based `nn.LeakyReLU` (power-of-2 negative slopes via arithmetic right-shift) directly in `gmdh_cell.v`, and optional piecewise-linear LUT blocks for smooth activations (`Tanh`/`Sigmoid`/`SoftClip`).
- **Multi-action & continuous outputs**: Emit $K$ output accumulators plus a pipelined `argmax` comparator tree for discrete spaces with $K > 2$ actions, or quantized scalar/vector output ports for continuous control and regression.
- **Flexible RTL & quantization modes**: Add per-layer `Q(I).(F)` formats, a `--combinational` generation flag, and a time-multiplexed (folded-PE) architecture for large networks on small FPGAs.
- **Additional Gymnasium environments**: Store `env_id` and action-space metadata (`discrete`/`continuous`, action count/bounds) in `quantized_model.json` so `test_cartpole_cocotb.py` (renamed to a generic closed-loop RTL runner) can drive arbitrary Gymnasium environments without hardcoding `CartPole-v1`.

