"""Deep Q-Learning (DQN) reinforcement learning example encapsulated in ``Reinforcement``."""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import time
import warnings
from glob import iglob
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gmdhtorch import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_MAX_OBSERVE_BYTES,
    GMDHLayer,
    GMDHModel,
    GMDHOutput,
    ModelIncrementer,
    SelectRelativeThreshold,
)
from torch import nn
from torch.optim import AdamW

try:
    from .utils import configure_torch_resources
except ImportError:
    from utils import configure_torch_resources  # type: ignore[no-redef]


_DEFAULT_CHUNK_SIZE = DEFAULT_CHUNK_SIZE * 2
_DEFAULT_MAX_CHUNK_BYTES = DEFAULT_MAX_CHUNK_BYTES * 2
_DEFAULT_MAX_OBSERVE_BYTES = DEFAULT_MAX_OBSERVE_BYTES * 2


def _gpu_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError(f"Unsupported dtype {dtype!r}; expected torch.float32 or torch.float64.")


def _name_to_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {name!r}; expected 'float32' or 'float64'.")


class ReplayMemory:
    """Pre-allocated contiguous tensor ring-buffer replay memory."""

    def __init__(
        self,
        *,
        capacity: int = 100000,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.dtype = dtype

        self._states: torch.Tensor = None  # type: ignore[assignment]
        self._next_states: torch.Tensor = None  # type: ignore[assignment]
        self._actions: torch.Tensor = None  # type: ignore[assignment]
        self._rewards: torch.Tensor = None  # type: ignore[assignment]
        self._dones: torch.Tensor = None  # type: ignore[assignment]
        self._size = 0
        self.position = 0

    def _ensure_buffers(self, state: Any) -> None:
        obs_shape = np.asarray(state).shape
        self._states = torch.empty((self.capacity, *obs_shape), dtype=self.dtype, device=self.device)
        self._next_states = torch.empty((self.capacity, *obs_shape), dtype=self.dtype, device=self.device)
        self._actions = torch.empty((self.capacity, 1), dtype=torch.int64, device=self.device)
        self._rewards = torch.empty((self.capacity, 1), dtype=self.dtype, device=self.device)
        self._dones = torch.empty((self.capacity, 1), dtype=torch.bool, device=self.device)

    def insert(self, transition: tuple[Any, int, float, bool, Any] | list[Any]) -> None:
        state, action, reward, done, next_state = transition
        if self._states is None:
            self._ensure_buffers(state)
        pos = self.position
        self._states[pos].copy_(torch.as_tensor(state, dtype=self.dtype, device=self.device))
        self._next_states[pos].copy_(torch.as_tensor(next_state, dtype=self.dtype, device=self.device))
        self._actions[pos, 0] = int(action)
        self._rewards[pos, 0] = float(reward)
        self._dones[pos, 0] = bool(done)

        self.position = (pos + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def _sample_indices(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self._size, (batch_size,), device=self.device)

    def sample_states(self, batch_size: int) -> torch.Tensor:
        idx = self._sample_indices(batch_size)
        return self._states[idx]

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self._sample_indices(batch_size)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._dones[idx],
            self._next_states[idx],
        )

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size * 10

    def clear(self) -> None:
        self._size = 0
        self.position = 0

    def __len__(self) -> int:
        return self._size


class Reinforcement:
    """Encapsulates DQN environment interaction, Q-network, replay memory, training, evaluation, and checkpoints."""

    RAD2DEG: float = 180.0 / float(np.pi)

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int | None = 42,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        learning_rate: float = 1e-4,
        replay_capacity: int = 100000,
        target_update_episodes: int = 10,
        model: Literal["gmdh", "dense"] = "gmdh",
        max_layers: int = 4,
        max_neurons: int = 150,
        select_threshold: float = 0.01,
        final_select_threshold: float = 0.1,
        chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
        max_chunk_bytes: int | None = _DEFAULT_MAX_CHUNK_BYTES,
        max_observe_bytes: int | None = _DEFAULT_MAX_OBSERVE_BYTES,
        memory_fraction: float | None = None,
        matmul_precision: str | None = None,
        show_info: bool = False,
    ) -> None:
        state_dims, num_actions = self._setup_config(
            env_id="CartPole-v1",
            device=device,
            dtype=dtype,
            seed=seed,
            batch_size=batch_size,
            gamma=gamma,
            epsilon=epsilon,
            learning_rate=learning_rate,
            replay_capacity=replay_capacity,
            target_update_episodes=target_update_episodes,
            model=model,
            max_layers=max_layers,
            max_neurons=max_neurons,
            select_threshold=select_threshold,
            final_select_threshold=final_select_threshold,
            chunk_size=chunk_size,
            max_chunk_bytes=max_chunk_bytes,
            max_observe_bytes=max_observe_bytes,
            memory_fraction=memory_fraction,
            matmul_precision=matmul_precision,
            show_info=show_info,
        )
        self._q_network: nn.Module = self._build_q_network(state_dims, num_actions)
        self._target_q_network: nn.Module = copy.deepcopy(self._q_network).eval().to(self._device)
        self._optimizer: torch.optim.Optimizer = AdamW(self._q_network.parameters(), lr=self._learning_rate)
        self._incrementer: ModelIncrementer = self._build_incrementer(self._q_network, self._optimizer)

    @classmethod
    def from_checkpoint(
        cls,
        filename: str,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_layers: int | None = None,
        max_neurons: int | None = None,
        select_threshold: float | None = None,
        final_select_threshold: float | None = None,
        show_info: bool | None = None,
    ) -> Reinforcement:
        """Restore a Reinforcement agent with its saved configuration and states from a checkpoint file."""
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        cfg = checkpoint["config"]

        agent = cls.__new__(cls)
        state_dims, num_actions = agent._setup_config(
            env_id=cfg["env_id"],
            device=device,
            dtype=_name_to_dtype(cfg["dtype"]) if dtype is None else dtype,
            seed=cfg["seed"],
            batch_size=cfg["batch_size"],
            gamma=cfg["gamma"],
            epsilon=cfg["epsilon"],
            learning_rate=cfg["learning_rate"],
            replay_capacity=cfg["replay_capacity"],
            target_update_episodes=cfg["target_update_episodes"],
            model=cfg["model"],
            max_layers=cfg["max_layers"] if max_layers is None else max_layers,
            max_neurons=cfg["max_neurons"] if max_neurons is None else max_neurons,
            select_threshold=cfg.get("select_threshold", 0.01) if select_threshold is None else select_threshold,
            final_select_threshold=cfg.get("final_select_threshold", 0.05) if final_select_threshold is None else final_select_threshold,
            chunk_size=cfg["chunk_size"],
            max_chunk_bytes=cfg["max_chunk_bytes"],
            max_observe_bytes=cfg["max_observe_bytes"],
            memory_fraction=cfg["memory_fraction"],
            matmul_precision=cfg["matmul_precision"],
            show_info=cfg["show_info"] if show_info is None else show_info,
        )

        if agent._model == "gmdh":
            agent._q_network = GMDHModel.from_config(
                checkpoint["model_config"],
                dtype=agent._dtype,
                device=agent._device,
            )
        else:
            agent._q_network = agent._build_q_network(state_dims, num_actions)

        agent._q_network.load_state_dict(checkpoint["q_network_state"])
        agent._target_q_network = copy.deepcopy(agent._q_network).eval().to(agent._device)
        agent._optimizer = AdamW(agent._q_network.parameters(), lr=agent._learning_rate)
        agent._optimizer.load_state_dict(checkpoint["optim_state"])
        agent._incrementer = agent._build_incrementer(agent._q_network, agent._optimizer)
        return agent

    def _setup_config(
        self,
        *,
        env_id: str,
        device: str | torch.device | None,
        dtype: torch.dtype,
        seed: int | None,
        batch_size: int,
        gamma: float,
        epsilon: float,
        learning_rate: float,
        replay_capacity: int,
        target_update_episodes: int,
        model: Literal["gmdh", "dense"],
        max_layers: int,
        max_neurons: int,
        select_threshold: float,
        final_select_threshold: float,
        chunk_size: int | None,
        max_chunk_bytes: int | None,
        max_observe_bytes: int | None,
        memory_fraction: float | None,
        matmul_precision: str | None,
        show_info: bool,
    ) -> tuple[int, int]:
        if model not in ("gmdh", "dense"):
            raise ValueError(f"Unsupported model {model!r}; expected 'gmdh' or 'dense'.")
        self._env_id = str(env_id)
        self._device = torch.device("cpu") if device is None else torch.device(device)
        self._dtype = dtype
        if self._device.type == "mps" and self._dtype == torch.float64:
            warnings.warn("MPS does not support torch.float64; falling back to torch.float32.", UserWarning, stacklevel=3)
            self._dtype = torch.float32
        self._seed = seed
        self._batch_size = int(batch_size)
        self._gamma = float(gamma)
        self._epsilon = float(epsilon)
        self._learning_rate = float(learning_rate)
        self._replay_capacity = int(replay_capacity)
        self._target_update_episodes = int(target_update_episodes)
        self._model: Literal["gmdh", "dense"] = model
        self._max_layers = int(max_layers)
        self._max_neurons = int(max_neurons)
        self._select_threshold = float(select_threshold)
        self._final_select_threshold = float(final_select_threshold)
        if self._select_threshold < 0.0 or self._final_select_threshold < 0.0:
            raise ValueError("select_threshold and final_select_threshold must be >= 0.0.")
        self._chunk_size = int(chunk_size) if chunk_size is not None else None
        self._max_chunk_bytes = int(max_chunk_bytes) if max_chunk_bytes is not None else None
        self._max_observe_bytes = int(max_observe_bytes) if max_observe_bytes is not None else None
        self._memory_fraction = float(memory_fraction) if memory_fraction is not None else None
        self._matmul_precision = matmul_precision
        self._show_info = bool(show_info)

        configure_torch_resources(
            device=self._device,
            memory_fraction=self._memory_fraction,
            matmul_precision=self._matmul_precision,
        )

        self.reset_rng(self._seed)
        self._env: gym.Env = gym.make(self._env_id)
        self._env.reset(seed=self._seed)

        obs_shape = self._env.observation_space.shape
        state_dims = int(obs_shape[0])  # type: ignore[index]
        num_actions = int(self._env.action_space.n)  # type: ignore[attr-defined]
        self._memory = ReplayMemory(
            capacity=self._replay_capacity,
            device=self._device,
            dtype=self._dtype,
        )
        return state_dims, num_actions

    @property
    def model(self) -> Literal["gmdh", "dense"]:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset random number generators from ``seed``."""
        self._seed = seed
        self._rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(int(seed))

    def _build_q_network(self, state_dims: int, num_actions: int) -> nn.Sequential:
        """Construct the dense MLP or eager GMDH Q-network."""
        if self._model == "gmdh":
            layer1 = GMDHLayer(
                in_features=int(state_dims),
                inputs=2,
                degree=1,
                interaction=0,
                max_neurons=self._max_neurons,
                init="xavier",
                rng=self._rng,
                dtype=self._dtype,
                device=self._device,
                chunk_size=self._chunk_size,
                max_chunk_bytes=self._max_chunk_bytes,
            )
            return GMDHModel(
                layer1,
                GMDHOutput(
                    count=int(num_actions),
                    dense=True,
                    in_features=layer1.out_features,
                    dtype=self._dtype,
                    device=self._device,
                ),
                dtype=self._dtype,
                device=self._device,
            )

        return nn.Sequential(
            nn.Linear(int(state_dims), 6, dtype=self._dtype, device=self._device),
            nn.ReLU(),
            nn.Linear(6, 7, dtype=self._dtype, device=self._device),
            nn.ReLU(),
            nn.Linear(7, 10, dtype=self._dtype, device=self._device),
            nn.ReLU(),
            nn.Linear(10, int(num_actions), dtype=self._dtype, device=self._device),
        )

    def _build_incrementer(
        self,
        q_network: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> ModelIncrementer:
        if self._model == "gmdh":
            return ModelIncrementer(
                q_network,
                optimizer=optimizer,
                layer=GMDHLayer(
                    inputs=2,
                    degree=1,
                    interaction=0,
                    forward=False,
                    max_neurons=self._max_neurons,
                    init="xavier",
                    dtype=self._dtype,
                    device=self._device,
                    chunk_size=self._chunk_size,
                    max_chunk_bytes=self._max_chunk_bytes,
                ),
                activation=nn.Hardtanh(min_val=-2.0, max_val=2.0),
                norm=False,
                select=SelectRelativeThreshold(self._select_threshold),
                final_select=SelectRelativeThreshold(self._final_select_threshold),
                max_layers=self._max_layers,
                gradient_during_growth=True,
                solver=None,
                chunk_size=self._chunk_size,
                max_chunk_bytes=self._max_chunk_bytes,
                max_observe_bytes=self._max_observe_bytes,
                logs=self._show_info,
                rng=self._rng,
            )
        return ModelIncrementer(
            q_network,
            chunk_size=self._chunk_size,
            max_chunk_bytes=self._max_chunk_bytes,
            max_observe_bytes=self._max_observe_bytes,
            logs=False,
        )

    def save_checkpoint(self, filename: str, *, overwrite: bool = False) -> None:
        """Save configuration, Q-network, and optimizer states with automatic numbered backups."""
        out_path = Path(filename)
        if not overwrite and out_path.exists():
            basename = out_path.stem
            pattern = re.compile(r".+_(\d+)$")
            backup_no = 0
            for backup_filename in iglob(str(out_path.parent / f"{basename}_*{out_path.suffix}")):
                backup_basename = Path(backup_filename).stem
                match = pattern.match(backup_basename)
                no = int(match.group(1)) if match is not None else 0
                backup_no = max(no + 1, backup_no)
            shutil.copy2(out_path, out_path.with_name(f"{basename}_{backup_no:03}{out_path.suffix}"))

        checkpoint: dict[str, Any] = {
            "config": {
                "env_id": self._env_id,
                "dtype": _dtype_to_name(self._dtype),
                "seed": self._seed,
                "batch_size": self._batch_size,
                "gamma": self._gamma,
                "epsilon": self._epsilon,
                "learning_rate": self._learning_rate,
                "replay_capacity": self._replay_capacity,
                "target_update_episodes": self._target_update_episodes,
                "model": self._model,
                "max_layers": self._max_layers,
                "max_neurons": self._max_neurons,
                "select_threshold": self._select_threshold,
                "final_select_threshold": self._final_select_threshold,
                "chunk_size": self._chunk_size,
                "max_chunk_bytes": self._max_chunk_bytes,
                "max_observe_bytes": self._max_observe_bytes,
                "memory_fraction": self._memory_fraction,
                "matmul_precision": self._matmul_precision,
                "show_info": self._show_info,
            },
            "q_network_state": self._q_network.state_dict(),
            "optim_state": self._optimizer.state_dict(),
        }
        if isinstance(self._q_network, GMDHModel):
            checkpoint["model_config"] = self._q_network.architecture_config()
        print(f'Saving "{out_path}"')
        torch.save(checkpoint, out_path)

    def _policy(self, state: Any, epsilon: float) -> int:
        """Select an epsilon-greedy action using the Q-network."""
        num_actions = int(self._env.action_space.n)  # type: ignore[attr-defined]
        if float(self._rng.random()) < epsilon:
            return int(self._rng.integers(0, num_actions))

        state_t = torch.as_tensor(state, dtype=self._dtype, device=self._device)
        with torch.no_grad():
            action_values = self._q_network(state_t)
        return int(action_values.argmax(dim=-1).item())

    def deep_q_learning(
        self,
        *,
        episodes: int,
        grow_episodes: int | None = None,
    ) -> None:
        """Train the Q-network with Deep Q-Learning and experience replay."""
        eff_grow_episodes = max(1, episodes // 10) if grow_episodes is None else max(1, int(grow_episodes))

        self._target_q_network = copy.deepcopy(self._q_network).eval().to(self._device)
        self._memory.clear()

        total_episodes = episodes + self._batch_size
        log_episodes = max(1, episodes // 10)

        ini_time = time.perf_counter()
        prv_time = ini_time
        prv_episode = 0
        for episode in range(1, total_episodes + 1):
            if (episode % log_episodes == 0) or (episode == total_episodes):
                cur_time = time.perf_counter()
                tot_time = cur_time - ini_time
                eps = (episode - prv_episode) / (cur_time - prv_time)
                est = (total_episodes - episode) / eps
                prv_time = cur_time
                prv_episode = episode
                print(f"Episode: {episode:5d} / {total_episodes:5d} [{tot_time:8.2f} s | {eps:6.2f} e/s | {est:6.2f} s]")
            state, _ = self._env.reset()
            done = False

            while not done:
                action = self._policy(state, self._epsilon)
                next_state, reward, terminated, truncated, _ = self._env.step(action)
                done = bool(terminated or truncated)

                self._memory.insert((state, action, float(reward), bool(terminated), next_state))

                if self._memory.can_sample(self._batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self._memory.sample(self._batch_size)
                    self._q_network.train()
                    qsa_b = self._q_network(state_b).gather(1, action_b)

                    with torch.no_grad():
                        next_qsa_b = self._target_q_network(next_state_b)
                        next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                        target_b = reward_b + (~done_b).to(dtype=self._dtype) * self._gamma * next_qsa_b

                    loss = F.mse_loss(qsa_b, target_b)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                state = next_state

            if self._incrementer.growing and (episode % eff_grow_episodes == 0) and self._memory.can_sample(self._batch_size):
                cal_count = min(len(self._memory), 256)
                cal_states = self._memory.sample_states(cal_count)
                self._incrementer.grow([(cal_states, None)], train_loader=[(cal_states, None)])
                self._target_q_network = copy.deepcopy(self._q_network).eval().to(self._device)
            elif episode % self._target_update_episodes == 0:
                self._target_q_network.load_state_dict(self._q_network.state_dict())

    def replay(self, *, episodes: int = 10, wait_for_enter: bool = True) -> None:
        """Render and evaluate the learned policy on a human-rendered environment."""
        if episodes < 1:
            return

        print("Testing Q-Network Policy")
        stable_debounce = 20.0
        stable_angle = 0.01
        env_t = gym.make(self._env_id, render_mode="human")
        self._q_network.eval()
        try:
            for episode in range(1, episodes + 1):
                state, _ = env_t.reset()
                done = False
                steps = 0
                start_time = time.time()
                previous_time = start_time
                progress_time = start_time
                duration = 0.0
                stable_duration = 0.0
                print(
                    f"Episode {episode:02}. Initial Angle: {state[2] * self.RAD2DEG:6.2f}° ",
                    end="",
                    flush=True,
                )
                while not done and stable_duration < stable_debounce:
                    action = self._policy(state, 0.0)
                    next_state, _reward, terminated, truncated, _ = env_t.step(action)
                    done = bool(terminated or truncated)
                    steps += 1

                    current_time = time.time()
                    if abs(next_state[2]) > stable_angle:
                        previous_time = current_time
                        stable_duration = 0.0
                        progress_mark = "."
                    else:
                        stable_duration = current_time - previous_time
                        progress_mark = f" {int(stable_duration)}"

                    duration = current_time - start_time
                    state = next_state

                    if current_time - progress_time >= 1.0:
                        print(progress_mark, end="", flush=True)
                        progress_time = current_time

                print(f" done. Steps: {steps}. Duration: {duration:6.2f}s.")

            if wait_for_enter:
                input("Press ENTER to finish...")
        finally:
            env_t.close()

    def print_objects(self) -> None:
        """Display environment, Q-network, and optimizer details."""
        print("Environment:")
        print(self._env)
        print()
        print("Network:")
        print(self._q_network)
        print()
        print("Optimizer:")
        print(self._optimizer)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser()
    opr_args = parser.add_argument_group('Operation')
    opr_args.add_argument("operation", type=str, choices=["train", "replay"], default="train", help="Define if training (and replaying) or only replaying")
    opr_args.add_argument("checkpoint", type=str, nargs="?", help="Checkpoint file to load (if existing) and save after training")
    opr_args.add_argument(
        "-s",
        "--save",
        dest="save_file",
        const=True,
        nargs="?",
        help="File name to save the checkpoint (defaults to checkpoint argument, or 'checkpoint.pth' if both are omitted)",
    )
    opr_args.add_argument("-o", "--overwrite", action="store_true", help="Does not create a backup of existing checkpoint when saving")
    opr_args.add_argument("-t", "--train-episodes", type=int, default=1000, help="Number of training episodes (after first batch) (default=1000)")
    opr_args.add_argument("-r", "--replay-episodes", type=int, default=3, help="Number of replay episodes (default=3)")

    mdl_args = parser.add_argument_group('Model')
    mdo_args = mdl_args.add_mutually_exclusive_group()
    mdo_args.add_argument("-g", "--gmdh", dest="model", action="store_const", const="gmdh", help="Uses a GMDH model for q learning. (default)")
    mdo_args.add_argument("-d", "--dense", dest="model", action="store_const", const="dense", help="Uses a dense model for q learning")

    trp_args = parser.add_argument_group('Extra training parameters')
    trp_args.add_argument("-b", "--batch_size", type=int, default=32, help="Number of samples in the batch size")
    trp_args.add_argument("-L", "--learning-rate", type=float, default=1e-3, help="Learning rate")
    trp_args.add_argument("--grow-episodes", type=int, default=None, help="Episodes between GMDH growth steps.")
    trp_args.add_argument("--max-layers", type=int, default=None, help="Maximum GMDH layers (default=4)")
    trp_args.add_argument("--max-neurons", type=int, default=None, help="Maximum neurons per GMDH layer (default=150)")
    trp_args.add_argument(
        "--select",
        type=float,
        nargs=2,
        metavar=("SELECT", "FINAL_SELECT"),
        default=None,
        help="Relative selection thresholds for intermediate and final GMDH layers (default=0.01 0.1)",
    )

    inf_args = parser.add_argument_group('Information')
    inf_args.add_argument("-i", "--info", action="store_true", help="Print information about the environment and the network")

    prf_args = parser.add_argument_group('Performance parameters')
    s_def_chunk_s = f"(default={_DEFAULT_CHUNK_SIZE})"
    s_max_chunk_b = f"(default={_DEFAULT_MAX_CHUNK_BYTES // 1024 // 1024:d} MiB)"
    s_max_obsrv_b = f"(default={_DEFAULT_MAX_OBSERVE_BYTES // 1024 // 1024:d} MiB)"
    prf_args.add_argument("--gpu", action="store_true", help="Uses CUDA or MPS GPU if available (default: CPU)")
    prf_args.add_argument("--dtype", type=str, choices=["float32", "float64"], default=None, help="Model & replay float dtype (default=float32)")
    prf_args.add_argument("--memory-fraction", type=float, default=0.8, help="Optional per-process GPU/MPS memory fraction (0.0-1.0)")
    prf_args.add_argument("--matmul-precision", type=str, choices=["highest", "high", "medium"], default=None, help="PyTorch float32 matmul precision")
    prf_args.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE, help=f"GMDH neuron chunk limit {s_def_chunk_s}")
    prf_args.add_argument("--max-chunk-bytes", type=int, default=_DEFAULT_MAX_CHUNK_BYTES, help=f"GMDH chunk byte ceiling {s_max_chunk_b}")
    prf_args.add_argument("--max-observe-bytes", type=int, default=_DEFAULT_MAX_OBSERVE_BYTES*2, help=f"GMDH observe spill byte threshold {s_max_obsrv_b}")

    parser.set_defaults(model="gmdh")
    args = parser.parse_args()

    train_model = False
    save_model = False
    load_model = False
    show_info = args.info

    if args.operation == "train":
        if args.train_episodes <= 0:
            parser.error("--train-episodes must be > 0 when operation is 'train'.")
        if args.batch_size <= 0:
            parser.error("--batch_size must be > 0 when operation is 'train'.")
        train_model = True
        save_model = (args.save_file is not None) or (args.checkpoint is not None)
        load_model = args.checkpoint is not None and Path(args.checkpoint).exists()
        replay_envr = args.replay_episodes > 0
    else:
        if args.replay_episodes <= 0:
            parser.error("--replay-episodes must be > 0 when operation is 'replay'.")
        if args.checkpoint is None:
            parser.error("A checkpoint file argument is required when operation is 'replay'.")
        if not Path(args.checkpoint).exists():
            parser.error(f"Cannot replay non-existing checkpoint {args.checkpoint!r}.")
        if args.save_file is not None:
            parser.error("--save cannot be used when operation is 'replay'.")
        replay_envr = True
        load_model = True

    device = _gpu_device() if args.gpu else torch.device("cpu")
    dtype = None if args.dtype is None else _name_to_dtype(args.dtype)
    select_threshold, final_select_threshold = (None, None) if args.select is None else args.select

    if load_model:
        checkpoint_status = f"Loaded checkpoint {args.checkpoint!r}"
        agent = Reinforcement.from_checkpoint(
            args.checkpoint,
            device=device,
            dtype=dtype,
            max_layers=args.max_layers,
            max_neurons=args.max_neurons,
            select_threshold=select_threshold,
            final_select_threshold=final_select_threshold,
            show_info=args.info,
        )
    else:
        checkpoint_status = "New model (no checkpoint loaded)"
        agent = Reinforcement(
            device=device,
            dtype=torch.float32 if dtype is None else dtype,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model=args.model,
            max_layers=4 if args.max_layers is None else args.max_layers,
            max_neurons=150 if args.max_neurons is None else args.max_neurons,
            select_threshold=0.01 if select_threshold is None else select_threshold,
            final_select_threshold=0.1 if final_select_threshold is None else final_select_threshold,
            chunk_size=args.chunk_size,
            max_chunk_bytes=args.max_chunk_bytes,
            max_observe_bytes=args.max_observe_bytes,
            memory_fraction=args.memory_fraction,
            matmul_precision=args.matmul_precision,
            show_info=args.info,
        )

    save_target: str | None = None
    if save_model:
        checkpoint_file = args.checkpoint if args.checkpoint is not None else "checkpoint.pth"
        save_target = args.save_file if isinstance(args.save_file, str) else checkpoint_file

    print(f"Checkpoint status.....: {checkpoint_status}")
    print(f"Save checkpoint.......: {save_target if save_target is not None else 'disabled'}")
    print(f"Model.................: {agent.model}")
    print(f"Device................: {agent.device}")
    print(f"Dtype.................: {agent.dtype}")
    print(f"Batch size............: {agent.batch_size}")
    print(f"Training episodes.....: {args.train_episodes if train_model else 'disabled'}")
    print(f"Replay episodes.......: {args.replay_episodes if replay_envr else 'disabled'}")

    try:
        if train_model:
            agent.deep_q_learning(episodes=args.train_episodes, grow_episodes=args.grow_episodes)

        if show_info:
            agent.print_objects()

        if save_target is not None:
            agent.save_checkpoint(filename=save_target, overwrite=args.overwrite)

        if replay_envr:
            agent.replay(episodes=args.replay_episodes)

    finally:
        agent.close()


if __name__ == "__main__":
    main()
