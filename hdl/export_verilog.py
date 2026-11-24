"""Generate readable, pipelined Verilog-2001 RTL and a self-checking testbench from ``quantized_model.json``.

Supports both pruned GMDH models (instantiating ``rtl/gmdh_cell.v``) and matched dense baselines
under the exact same top-level port interface and pipeline handshake.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

_RTL_DIR = Path(__file__).resolve().parent / "rtl"


def _s_const(val: int, width: int) -> str:
    """Format signed integer ``val`` as an explicit Verilog-2001 signed literal."""
    if val < 0:
        return f"-$signed({width}'d{abs(val)})"
    return f"$signed({width}'d{val})"


def _act_params(act: dict[str, Any] | None) -> tuple[int, int, int]:
    """Return ``(act_type, act_min, act_max)`` for ``gmdh_cell.v``."""
    if act is None:
        return 0, 0, 0
    if act["type"] == "hardtanh":
        return 1, int(act["min_q"]), int(act["max_q"])
    if act["type"] == "relu":
        return 2, 0, 0
    raise ValueError(f"Unsupported activation type {act['type']!r}.")


def _emit_folded_output(
    lines: list[str],
    *,
    prev_signals: list[str],
    output_cfg: dict[str, Any],
    acc_w: int,
    frac_w: int,
) -> None:
    """Emit the final pipeline stage for the folded single-output linear head."""
    lines.append("    // --- Output Stage: Folded linear head (w1 - w0) + sign comparator ---")
    terms: list[str] = [f"({_s_const(int(output_cfg['bias']), acc_w)} <<< FRAC_W)"]
    for idx, (sig, w_val) in enumerate(zip(prev_signals, output_cfg["weights"], strict=True)):
        lines.append(f"    wire signed [ACC_W-1:0] out_x_{idx} = {sig};")
        terms.append(f"({_s_const(int(w_val), acc_w)} * out_x_{idx})")

    lines.append("    wire signed [ACC_W-1:0] out_acc =")
    for i, term in enumerate(terms):
        sep = " +" if i < len(terms) - 1 else ";"
        lines.append(f"        {term}{sep}")
    lines.append("")
    lines.append("    always @(posedge clk) begin")
    lines.append("        if (!rst_n) begin")
    lines.append("            delta_q <= {ACC_W{1'b0}};")
    lines.append("            action  <= 1'b0;")
    lines.append("        end else begin")
    lines.append("            delta_q <= out_acc;")
    lines.append("            action  <= (out_acc > 0) ? 1'b1 : 1'b0;")
    lines.append("        end")
    lines.append("    end")


def generate_rtl(model_cfg: dict[str, Any], *, module_name: str) -> str:
    """Return Verilog-2001 source code for a pipelined GMDH or Dense fixed-point controller."""
    fmt = model_cfg["format"]
    data_w = int(fmt["data_bits"])
    frac_w = int(fmt["frac_bits"])
    acc_w = int(fmt["acc_bits"])
    round_nearest = 1 if fmt["round_mode"] == "nearest" else 0
    n_in = int(model_cfg["input_count"])
    layers = model_cfg["layers"]
    latency = len(layers) + 1

    lines: list[str] = [
        f"// Auto-generated pipelined Verilog-2001 RTL ({model_cfg['model_type'].upper()} model)",
        f"// Source: {model_cfg.get('source_checkpoint', 'unknown')}",
        (
            f"// Format: Q{fmt['int_bits']}.{frac_w} ({data_w}-bit signed), ACC_W={acc_w}, "
            f"round={fmt['round_mode']}, latency={latency} cycles"
        ),
        "",
        "`timescale 1ns/1ps",
        "",
        f"module {module_name} #(",
        f"    parameter integer DATA_W = {data_w},",
        f"    parameter integer FRAC_W = {frac_w},",
        f"    parameter integer ACC_W  = {acc_w}",
        ") (",
        "    input  wire                     clk,",
        "    input  wire                     rst_n,",
        "    input  wire                     valid_in,",
    ]
    for i in range(n_in):
        lines.append(f"    input  wire signed [DATA_W-1:0] in_{i},")
    lines.extend([
        "    output wire                     valid_out,",
        "    output reg                      action,",
        "    output reg  signed [ACC_W-1:0]  delta_q",
        ");",
        f"    localparam integer LATENCY = {latency};",
        f"    localparam integer ROUND_NEAREST = {round_nearest};",
    ])
    if model_cfg["model_type"] == "dense":
        lines.extend([
            "    localparam signed [ACC_W-1:0] Q_MIN = -(1 <<< (DATA_W - 1));",
            "    localparam signed [ACC_W-1:0] Q_MAX =  (1 <<< (DATA_W - 1)) - 1;",
            "    localparam signed [ACC_W-1:0] ROUND_BIAS = (ROUND_NEAREST != 0 && FRAC_W > 0) ? (1 <<< (FRAC_W - 1)) : 0;",
        ])
    lines.extend([
        "",
        "    // Pipeline valid shift register",
        "    reg [LATENCY-1:0] valid_pipe;",
        "    always @(posedge clk) begin",
        "        if (!rst_n)",
        "            valid_pipe <= {LATENCY{1'b0}};",
        "        else",
        "            valid_pipe <= {valid_pipe[LATENCY-2:0], valid_in};",
        "    end",
        "    assign valid_out = valid_pipe[LATENCY-1];",
        "",
    ])

    prev_signals = [f"in_{i}" for i in range(n_in)]

    for layer in layers:
        lno = int(layer["layer_number"])
        kind = layer["kind"]
        act = layer.get("activation")
        act_type, act_min, act_max = _act_params(act)
        act_label = "none" if act is None else act["type"]
        curr_signals: list[str] = []

        if kind == "gmdh":
            n_neurons = int(layer["neuron_count"])
            lines.append(f"    // --- Layer {lno}: GMDH ({n_neurons} cells, act={act_label}) ---")
            for n_idx in range(n_neurons):
                sig = f"l{lno}_n{n_idx}"
                curr_signals.append(sig)
                i1, i2 = layer["idx0"][n_idx]
                w0 = int(layer["bias"][n_idx])
                w1, w2 = int(layer["weights"][n_idx][0]), int(layer["weights"][n_idx][1])
                lines.append(f"    wire signed [DATA_W-1:0] {sig};")
                lines.append(
                    f"    gmdh_cell #("
                    f".DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W), "
                    f".W0({w0}), .W1({w1}), .W2({w2}), "
                    f".ACT_TYPE({act_type}), .ACT_MIN({act_min}), .ACT_MAX({act_max}), "
                    f".ROUND_NEAREST(ROUND_NEAREST)) u_{sig} ("
                    f".clk(clk), .rst_n(rst_n), .x1({prev_signals[i1]}), .x2({prev_signals[i2]}), .y({sig}));"
                )
            lines.append("")
        elif kind == "dense":
            n_out = int(layer["out_features"])
            lines.append(f"    // --- Layer {lno}: Dense ({layer['in_features']} -> {n_out}, act={act_label}) ---")
            for k, ps in enumerate(prev_signals):
                lines.append(f"    wire signed [ACC_W-1:0] l{lno}_in_{k} = {ps};")
            for n_idx in range(n_out):
                sig = f"l{lno}_n{n_idx}"
                curr_signals.append(sig)
                b_q = int(layer["bias"][n_idx])
                w_row = layer["weights"][n_idx]
                terms = [f"({_s_const(b_q, acc_w)} <<< FRAC_W)"] + [
                    f"({_s_const(int(w_k), acc_w)} * l{lno}_in_{k})" for k, w_k in enumerate(w_row)
                ]
                lines.append(f"    wire signed [ACC_W-1:0] {sig}_acc = {' + '.join(terms)};")
                lines.append(f"    wire signed [ACC_W-1:0] {sig}_shf = ({sig}_acc + ROUND_BIAS) >>> FRAC_W;")
                if act_type == 1:
                    lo_expr = _s_const(act_min, acc_w)
                    hi_expr = _s_const(act_max, acc_w)
                elif act_type == 2:
                    lo_expr = f"$signed({acc_w}'d0)"
                    hi_expr = "Q_MAX"
                else:
                    lo_expr = "Q_MIN"
                    hi_expr = "Q_MAX"
                lines.append(
                    f"    wire signed [ACC_W-1:0] {sig}_clp = "
                    f"({sig}_shf < {lo_expr}) ? {lo_expr} : (({sig}_shf > {hi_expr}) ? {hi_expr} : {sig}_shf);"
                )
                lines.append(f"    reg  signed [DATA_W-1:0] {sig};")
            lines.append("    always @(posedge clk) begin")
            lines.append("        if (!rst_n) begin")
            for sig in curr_signals:
                lines.append(f"            {sig} <= {{DATA_W{{1'b0}}}};")
            lines.append("        end else begin")
            for sig in curr_signals:
                lines.append(f"            {sig} <= {sig}_clp[DATA_W-1:0];")
            lines.append("        end")
            lines.append("    end")
            lines.append("")
        else:
            raise ValueError(f"Unsupported layer kind {kind!r}.")

        prev_signals = curr_signals

    _emit_folded_output(
        lines,
        prev_signals=prev_signals,
        output_cfg=model_cfg["output"],
        acc_w=acc_w,
        frac_w=frac_w,
    )
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def generate_testbench(
    model_cfg: dict[str, Any],
    vectors_meta: dict[str, Any],
    *,
    module_name: str,
    vectors_filename: str = "vectors.hex",
) -> str:
    """Generate a self-checking Verilog-2001 testbench verifying every entry in ``vectors.hex``."""
    fmt = model_cfg["format"]
    data_w = int(fmt["data_bits"])
    acc_w = int(fmt["acc_bits"])
    n_in = int(model_cfg["input_count"])
    latency = len(model_cfg["layers"]) + 1
    num_vecs = int(vectors_meta["num_vectors"])
    total_bits = int(vectors_meta["total_bits"])
    fields = {f["name"]: f for f in vectors_meta["fields"]}

    lines: list[str] = [
        f"// Self-checking Verilog-2001 testbench for {module_name}",
        "`timescale 1ns/1ps",
        "",
        f"module tb_{module_name};",
        f"    localparam integer DATA_W      = {data_w};",
        f"    localparam integer ACC_W       = {acc_w};",
        f"    localparam integer LATENCY     = {latency};",
        f"    localparam integer NUM_VECTORS = {num_vecs};",
        f"    localparam integer TOTAL_BITS  = {total_bits};",
        "",
        "    reg clk = 1'b0;",
        "    reg rst_n = 1'b0;",
        "    reg valid_in = 1'b0;",
    ]
    for i in range(n_in):
        lines.append(f"    reg signed [DATA_W-1:0] in_{i} = 0;")
    lines.extend([
        "    wire valid_out;",
        "    wire action;",
        "    wire signed [ACC_W-1:0] delta_q;",
        "",
        f"    {module_name} dut (",
        "        .clk(clk),",
        "        .rst_n(rst_n),",
        "        .valid_in(valid_in),",
    ])
    for i in range(n_in):
        lines.append(f"        .in_{i}(in_{i}),")
    lines.extend([
        "        .valid_out(valid_out),",
        "        .action(action),",
        "        .delta_q(delta_q)",
        "    );",
        "",
        "    always #5 clk = ~clk;",
        "",
        "    reg [TOTAL_BITS-1:0] vectors [0:NUM_VECTORS-1];",
        "    integer feed_idx;",
        "    integer check_idx;",
        "    integer errors;",
        "    reg signed [ACC_W-1:0] exp_delta_q;",
        "    reg exp_action;",
        "",
        "    initial begin",
        f'        $readmemh("{vectors_filename}", vectors);',
        "        errors = 0;",
        "        check_idx = 0;",
        "        rst_n = 1'b0;",
        "        valid_in = 1'b0;",
        "        repeat (3) @(posedge clk);",
        "        rst_n = 1'b1;",
        "",
        "        for (feed_idx = 0; feed_idx < NUM_VECTORS + LATENCY + 2; feed_idx = feed_idx + 1) begin",
        "            @(negedge clk);",
        "            if (feed_idx < NUM_VECTORS) begin",
        "                valid_in = 1'b1;",
    ])
    for i in range(n_in):
        f = fields[f"in_{i}"]
        lines.append(f"                in_{i} = $signed(vectors[feed_idx][{f['msb']}:{f['lsb']}]);")
    lines.extend([
        "            end else begin",
        "                valid_in = 1'b0;",
    ])
    for i in range(n_in):
        lines.append(f"                in_{i} = 0;")
    lines.extend([
        "            end",
        "",
        "            @(posedge clk);",
        "            #1;",
        "            if (valid_out) begin",
        "                if (check_idx < NUM_VECTORS) begin",
        f"                    exp_delta_q = $signed(vectors[check_idx][{fields['delta_q']['msb']}:{fields['delta_q']['lsb']}]);",
        f"                    exp_action  = vectors[check_idx][{fields['action']['msb']}];",
        "                    if (delta_q !== exp_delta_q || action !== exp_action) begin",
        '                        $display("MISMATCH vec %0d: got delta_q=%0d action=%0b, exp delta_q=%0d action=%0b",',
        "                                 check_idx, delta_q, action, exp_delta_q, exp_action);",
        "                        errors = errors + 1;",
        "                    end",
        "                    check_idx = check_idx + 1;",
        "                end",
        "            end",
        "        end",
        "",
        "        if (check_idx !== NUM_VECTORS) begin",
        '            $display("FAIL: checked %0d of %0d vectors.", check_idx, NUM_VECTORS);',
        "            errors = errors + 1;",
        "        end",
        "        if (errors == 0)",
        f'            $display("PASS: {module_name} matched all %0d vectors.", NUM_VECTORS);',
        "        else",
        f'            $display("FAIL: {module_name} had %0d mismatches.", errors);',
        "        $finish;",
        "    end",
        "endmodule",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Verilog-2001 RTL, testbench, and hdl_manifest.json from an exported quantized_model.json directory.",
    )
    parser.add_argument("output_dir", type=str, help="Directory containing quantized_model.json (and vectors_meta.json)")
    parser.add_argument("--module-name", type=str, default=None, help="Override top-level Verilog module name")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_json = output_dir / "quantized_model.json"
    if not model_json.exists():
        parser.error(f"Missing {model_json}; run 'quantize.py export <checkpoint> {output_dir} --bits <W>' first.")

    model_cfg = json.loads(model_json.read_text(encoding="utf-8"))
    module_name = args.module_name or f"{model_cfg['model_type']}_cartpole"
    rtl_files: list[str] = []

    if model_cfg["model_type"] == "gmdh":
        cell_src = _RTL_DIR / "gmdh_cell.v"
        cell_dst = output_dir / "gmdh_cell.v"
        shutil.copy2(cell_src, cell_dst)
        rtl_files.append("gmdh_cell.v")
        print(f"Copied RTL:    {cell_dst}")

    rtl_filename = f"{module_name}.v"
    rtl_path = output_dir / rtl_filename
    rtl_path.write_text(generate_rtl(model_cfg, module_name=module_name), encoding="utf-8")
    rtl_files.append(rtl_filename)
    print(f"Generated RTL: {rtl_path}")

    tb_filename: str | None = None
    meta_json = output_dir / "vectors_meta.json"
    if meta_json.exists():
        vectors_meta = json.loads(meta_json.read_text(encoding="utf-8"))
        tb_filename = f"tb_{module_name}.v"
        tb_path = output_dir / tb_filename
        tb_path.write_text(generate_testbench(model_cfg, vectors_meta, module_name=module_name), encoding="utf-8")
        print(f"Generated TB:  {tb_path}")

    manifest = {
        "version": 1,
        "module_name": module_name,
        "model_type": model_cfg["model_type"],
        "rtl_files": rtl_files,
        "tb_file": tb_filename,
    }
    manifest_path = output_dir / "hdl_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
