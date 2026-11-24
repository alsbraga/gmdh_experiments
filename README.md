# `gmdh_experiments` — Benchmarks, Reinforcement Learning & Verilog HDL Workflows

Standalone research experiments and hardware synthesis workflows built on top of [**`gmdhtorch`**](https://github.com/alsbraga/gmdhtorch) (Group Method of Data Handling in PyTorch).

---

## Installation

Running `pip install -r requirements.txt` installs the experiment dependencies and automatically pulls [**`gmdhtorch`**](https://github.com/alsbraga/gmdhtorch) directly from GitHub. Note that **`gmdhtorch` itself requires PyTorch (`torch>=2.0`)** (also listed in `requirements.txt`).

`gmdhtorch` is pure Python, performs no unusual install-time steps, and—as of now—its module name does not conflict with any other known Python package. Still, if you want complete isolation from your global Python installation, creating a local virtual environment is recommended.

### Using Python's built-in `venv` (recommended)

```bash
# 1. Create a local virtual environment
python3 -m venv .venv

# 2. Activate it
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
# Windows (Command Prompt):
# .venv\Scripts\activate.bat

# 3. Install dependencies (pulls gmdhtorch from GitHub)
pip install -r requirements.txt
```

> **PyTorch / GPU note:** If you need a specific CUDA build of PyTorch, install PyTorch first inside the activated environment using the command from [pytorch.org](https://pytorch.org/get-started/locally/), and then run `pip install -r requirements.txt`.
>
> **Git LFS note:** Pre-trained `.pth` checkpoints are stored with [Git LFS](https://git-lfs.com/). Make sure `git lfs install` is enabled (or run `git lfs pull`) so `gmdh.pth` and `dense.pth` download as binary checkpoints rather than pointer files.

### Other environment managers

If you prefer managing Python versions or environments with another tool:

- **[pyenv](https://github.com/pyenv/pyenv)** (and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)) — great for installing and switching between multiple CPython versions compiled locally.
- **[uv](https://docs.astral.sh/uv/)** — a very fast drop-in replacement for `pip`/`venv`. It works smoothly with prebuilt wheels and pure-Python packages like `gmdhtorch`, though note that when a dependency must be **compiled from source** (C/C++/CUDA extensions or legacy `setup.py` packages), `uv`'s strict build isolation and standalone Python builds can occasionally run into compilation/header quirks that require extra flags (such as `--no-build-isolation`).
- **[Miniconda / Conda](https://docs.conda.io/)** — helpful if you also manage non-Python scientific or CUDA toolchains.

---

## Repository Contents

| Path | Description |
| :--- | :--- |
| `iris.ipynb`, `iris.csv`, `iris.names` | Iris pattern-recognition notebook comparing GMDH and dense neural networks |
| `reinforcement.py`, [`REINFORCEMENT.md`](REINFORCEMENT.md) | DQN reinforcement learning on `CartPole-v1` using constructively grown GMDH and compact dense Q-networks |
| `gmdh.pth`, `dense.pth` | Pre-trained CartPole-v1 checkpoints (pruned GMDH vs. compact dense MLP) |
| `hdl/` | Cross-platform fixed-point quantization (`Q(I).(F)`) and Verilog-2001 RTL generation, simulation (`iverilog`, `verilator`, `cocotb`), and FPGA synthesis (`yosys`) |
| `utils.py` | Shared PyTorch/NumPy data, device-budget, and plotting helpers |

---

## Quickstart

### 1. CartPole-v1 Reinforcement Learning (`reinforcement.py`)

Replay a pre-trained checkpoint and inspect its architecture:

```bash
python reinforcement.py replay gmdh.pth -i -r 3
python reinforcement.py replay dense.pth -i -r 3
```

Train a new GMDH or dense DQN agent (runs on CPU by default; pass `--gpu` to use CUDA/MPS):

```bash
python reinforcement.py train my_gmdh.pth --gmdh -t 1000
python reinforcement.py train my_dense.pth --dense -t 1000
```

See [REINFORCEMENT.md](REINFORCEMENT.md) for the full checkpoint, growth/pruning (`--max-layers`, `--select`), and parameter guide.

### 2. Fixed-Point Quantization & Verilog-2001 HDL (`hdl/`)

Quantize `gmdh.pth` / `dense.pth`, export Verilog-2001 RTL and `$readmemh` testbenches, simulate with Icarus Verilog or Verilator, and synthesize with Yosys:

```bash
cd hdl
python quantize.py export ../gmdh.pth build/gmdh/12b --bits 12 --episodes 10
python export_verilog.py build/gmdh/12b
python run_flow.py simulate build/gmdh/12b --sim iverilog
```

See [hdl/README.md](hdl/README.md) for the full quantization/synthesis pipeline and [hdl/TOOLS_OVERVIEW.md](hdl/TOOLS_OVERVIEW.md) for an introduction to the open-source EDA tools.

### 3. Iris Classification (`iris.ipynb`)

Open `iris.ipynb` in Jupyter / VS Code and run all cells to compare GMDH and dense models on the UCI Iris dataset (`iris.csv`).

---

## License

BSD 3-Clause License — see [LICENSE](LICENSE) for details.
