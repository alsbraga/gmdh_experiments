"""Cross-platform Python driver for RTL simulation (Icarus/Verilator) and Yosys FPGA synthesis.

Reads ``hdl_manifest.json`` written by ``export_verilog.py``, runs identically on macOS, Linux,
and Windows (e.g. with YosysHQ OSS CAD Suite on PATH), and saves synthesis metrics as JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_YOSYS_TARGETS: dict[str, str] = {
    "xilinx": "synth_xilinx -top {top}",
    "ecp5": "synth_ecp5 -top {top}",
    "ice40": "synth_ice40 -top {top}",
    "gowin": "synth_gowin -top {top}",
}


def _require_tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise FileNotFoundError(
            f"Required tool {name!r} not found on PATH. Install YosysHQ OSS CAD Suite "
            "(available for macOS, Linux, and Windows) and add its bin/ directory to PATH."
        )
    return exe


def cmd_simulate(output_dir: Path, *, mod_name: str, rtl_files: list[Path], tb_file: Path, sim: str) -> None:
    """Compile and run the standalone Verilog-2001 testbench against ``vectors.hex``."""
    if sim == "iverilog":
        iverilog = _require_tool("iverilog")
        vvp = _require_tool("vvp")
        sim_out = output_dir / f"sim_{mod_name}.vvp"
        compile_cmd = [iverilog, "-g2005", "-Wall", "-o", str(sim_out), *(str(p) for p in rtl_files), str(tb_file)]
        subprocess.run(compile_cmd, cwd=str(output_dir), check=True)
        run_res = subprocess.run([vvp, str(sim_out)], cwd=str(output_dir), capture_output=True, text=True, check=True)
    elif sim == "verilator":
        verilator = _require_tool("verilator")
        obj_dir = output_dir / f"obj_{mod_name}"
        compile_cmd = [
            verilator,
            "--binary",
            "-j",
            "0",
            "--timing",
            "-Wno-WIDTHTRUNC",
            "-Wno-WIDTHEXPAND",
            "--Mdir",
            str(obj_dir),
            "--top-module",
            f"tb_{mod_name}",
            *(str(p) for p in rtl_files),
            str(tb_file),
        ]
        subprocess.run(compile_cmd, cwd=str(output_dir), check=True)
        sim_bin = obj_dir / f"Vtb_{mod_name}"
        if sys.platform == "win32" and not sim_bin.exists():
            sim_bin = sim_bin.with_suffix(".exe")
        run_res = subprocess.run([str(sim_bin)], cwd=str(output_dir), capture_output=True, text=True, check=True)
    else:
        raise ValueError(f"Unsupported simulator {sim!r}; expected 'iverilog' or 'verilator'.")

    print(run_res.stdout.strip())
    if "PASS:" not in run_res.stdout or "FAIL:" in run_res.stdout:
        raise RuntimeError(f"Simulation failed for {mod_name}.")


def cmd_synth(output_dir: Path, *, mod_name: str, rtl_files: list[Path], targets: list[str]) -> None:
    """Synthesize RTL via Yosys for each target family and save structured JSON resource statistics."""
    yosys = _require_tool("yosys")
    model_cfg = json.loads((output_dir / "quantized_model.json").read_text(encoding="utf-8"))

    results: dict[str, Any] = {
        "version": 1,
        "module_name": mod_name,
        "model_type": model_cfg["model_type"],
        "format": model_cfg["format"],
        "footprint": model_cfg["footprint"],
        "targets": {},
    }

    for target in targets:
        if target not in _YOSYS_TARGETS:
            raise ValueError(f"Unsupported Yosys target {target!r}; expected one of {list(_YOSYS_TARGETS)}.")
        stat_json_path = output_dir / f"yosys_stat_{target}.json"
        log_path = output_dir / f"yosys_{target}.log"
        read_cmds = "; ".join(f"read_verilog {p.as_posix()}" for p in rtl_files)
        synth_cmd = _YOSYS_TARGETS[target].format(top=mod_name)
        yosys_script = f"{read_cmds}; {synth_cmd}; tee -q -o {stat_json_path.as_posix()} stat -json"

        res = subprocess.run([yosys, "-p", yosys_script], cwd=str(output_dir), capture_output=True, text=True, check=True)
        log_path.write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")

        raw_stat = json.loads(stat_json_path.read_text(encoding="utf-8"))
        design_stat = raw_stat.get("design", {})
        cells = design_stat.get("num_cells_by_type", {})
        results["targets"][target] = {
            "num_wires": design_stat.get("num_wires", 0),
            "num_wire_bits": design_stat.get("num_wire_bits", 0),
            "num_cells": design_stat.get("num_cells", 0),
            "num_cells_by_type": cells,
        }
        print(f"[{target}] total cells={design_stat.get('num_cells', 0)}, breakdown={cells}")

    out_path = output_dir / "synth_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved synthesis results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform Verilog simulation and Yosys synthesis runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    sim_p = sub.add_parser("simulate", help="Run standalone Verilog-2001 testbench against vectors.hex")
    sim_p.add_argument("output_dir", type=str, help="Exported model directory containing hdl_manifest.json")
    sim_p.add_argument("--sim", type=str, choices=["iverilog", "verilator"], default="iverilog", help="Simulator (default=iverilog)")

    syn_p = sub.add_parser("synth", help="Run Yosys FPGA synthesis and save structured JSON resource stats")
    syn_p.add_argument("output_dir", type=str, help="Exported model directory containing hdl_manifest.json")
    syn_p.add_argument("--targets", type=str, nargs="+", default=["xilinx", "ecp5", "ice40"], choices=list(_YOSYS_TARGETS))

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "hdl_manifest.json"
    if not manifest_path.exists():
        parser.error(f"Missing {manifest_path}; run 'export_verilog.py {args.output_dir}' first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mod_name = str(manifest["module_name"])
    rtl_files = [output_dir / name for name in manifest["rtl_files"]]

    if args.command == "simulate":
        tb_name = manifest.get("tb_file")
        if not tb_name:
            parser.error(f"No testbench recorded in {manifest_path}; export vectors via quantize.py export first.")
        cmd_simulate(output_dir, mod_name=mod_name, rtl_files=rtl_files, tb_file=output_dir / tb_name, sim=args.sim)
    elif args.command == "synth":
        cmd_synth(output_dir, mod_name=mod_name, rtl_files=rtl_files, targets=args.targets)


if __name__ == "__main__":
    main()
