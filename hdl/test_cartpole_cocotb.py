"""Closed-loop CartPole-v1 RTL-in-the-loop verification using cocotb (Icarus Verilog or Verilator).

Usage (with cocotb and iverilog/verilator installed):
    python test_cartpole_cocotb.py <output_dir> --episodes 10 [--render] [--sim icarus]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

_HDL_DIR = Path(__file__).resolve().parent
if str(_HDL_DIR) not in sys.path:
    sys.path.insert(0, str(_HDL_DIR))

from quantize import FixedPointModel

try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import RisingEdge, Timer
except ImportError:  # Allow importing module outside cocotb runner
    cocotb = None


if cocotb is not None:

    @cocotb.test()
    async def test_closed_loop_cartpole(dut: Any) -> None:
        """Drive the compiled Verilog controller closed-loop in CartPole-v1 and assert bit-exact parity."""
        build_dir = Path(os.environ["GMDH_BUILD_DIR"])
        episodes = int(os.environ.get("GMDH_EPISODES", "10"))
        seed = int(os.environ.get("GMDH_SEED", "0"))
        render = os.environ.get("GMDH_RENDER", "0") == "1"

        model_cfg = json.loads((build_dir / "quantized_model.json").read_text(encoding="utf-8"))
        fp_model = FixedPointModel.from_dict(model_cfg)
        latency = len(fp_model.layers) + 1

        cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
        dut.rst_n.value = 0
        dut.valid_in.value = 0
        for i in range(fp_model.input_count):
            getattr(dut, f"in_{i}").value = 0
        for _ in range(3):
            await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        await RisingEdge(dut.clk)

        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        steps_per_ep: list[int] = []
        total_steps = 0

        try:
            for ep in range(episodes):
                state, _ = env.reset(seed=seed + ep)
                done = False
                steps = 0
                while not done:
                    x_q = fp_model.quantize_inputs(state)
                    exp_delta_q, exp_action, _, _ = fp_model.forward_int(x_q)

                    dut.valid_in.value = 1
                    for i, v in enumerate(x_q):
                        getattr(dut, f"in_{i}").value = v
                    await RisingEdge(dut.clk)
                    dut.valid_in.value = 0

                    for _ in range(latency):
                        await RisingEdge(dut.clk)
                        await Timer(1, "ns")
                        if int(dut.valid_out.value) == 1:
                            break

                    assert int(dut.valid_out.value) == 1, "Expected valid_out == 1 after pipeline latency."
                    rtl_delta_q = int(dut.delta_q.value.to_signed())
                    rtl_action = int(dut.action.value)
                    assert rtl_delta_q == exp_delta_q, f"ep={ep} step={steps}: delta_q RTL={rtl_delta_q} != PY={exp_delta_q}"
                    assert rtl_action == exp_action, f"ep={ep} step={steps}: action RTL={rtl_action} != PY={exp_action}"

                    state, _, terminated, truncated, _ = env.step(rtl_action)
                    done = bool(terminated or truncated)
                    steps += 1
                    total_steps += 1
                steps_per_ep.append(steps)
        finally:
            env.close()

        arr = np.asarray(steps_per_ep, dtype=np.float64)
        solved = int(np.sum(arr >= 500))
        rtl_eval = {
            "episodes": episodes,
            "seed": seed,
            "total_steps_verified": total_steps,
            "mean_steps": float(np.mean(arr)),
            "min_steps": int(np.min(arr)),
            "max_steps": int(np.max(arr)),
            "solved_500_count": solved,
            "solved_500_rate": float(solved / episodes),
            "steps_per_episode": steps_per_ep,
        }
        (build_dir / "rtl_eval.json").write_text(json.dumps(rtl_eval, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from cocotb_tools.runner import get_runner

    parser = argparse.ArgumentParser(description="Run closed-loop CartPole-v1 RTL verification via cocotb.")
    parser.add_argument("output_dir", type=str, help="Exported model directory containing hdl_manifest.json")
    parser.add_argument("--sim", type=str, choices=["icarus", "iverilog", "verilator"], default="icarus", help="Simulator (default=icarus)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of CartPole-v1 episodes (default=10)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (default=0)")
    parser.add_argument("--render", action="store_true", help="Render CartPole-v1 live during RTL simulation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "hdl_manifest.json"
    if not manifest_path.exists():
        parser.error(f"Missing {manifest_path}; run 'export_verilog.py {args.output_dir}' first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mod_name = str(manifest["module_name"])
    sources = [output_dir / name for name in manifest["rtl_files"]]
    sim_name = "icarus" if args.sim == "iverilog" else args.sim

    runner = get_runner(sim_name)
    runner.build(
        sources=[str(s) for s in sources],
        hdl_toplevel=mod_name,
        build_dir=str(output_dir / f"cocotb_{sim_name}"),
        always=True,
    )
    runner.test(
        hdl_toplevel=mod_name,
        test_module="test_cartpole_cocotb",
        build_dir=str(output_dir / f"cocotb_{sim_name}"),
        extra_env={
            "PYTHONPATH": str(_HDL_DIR),
            "GMDH_BUILD_DIR": str(output_dir),
            "GMDH_EPISODES": str(args.episodes),
            "GMDH_SEED": str(args.seed),
            "GMDH_RENDER": "1" if args.render else "0",
        },
    )


if __name__ == "__main__":
    main()
