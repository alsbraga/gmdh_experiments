# CartPole-v1 Reinforcement Learning (`reinforcement.py`)

`reinforcement.py` trains and evaluates Deep Q-Learning (DQN) agents on Gymnasium's `CartPole-v1` environment, comparing constructively grown sparse affine **GMDH** Q-networks (`gmdhtorch`) against a matched compact **dense MLP** baseline.

---

## Quick Usage Overview

```bash
python reinforcement.py {train,replay} [checkpoint] [options]
```

- **`train`**: Trains a Q-network (from scratch or continuing from an existing checkpoint), optionally saves the checkpoint, and runs visual replay episodes (unless `-r 0`).
- **`replay`**: Loads an existing checkpoint and renders evaluation episodes without training.

### Checkpoint Workflows (`train`)

| Command | Behavior |
| :--- | :--- |
| `python reinforcement.py train <checkpoint-name>` | If `<checkpoint-name>` **does not exist**, trains a new model from scratch and saves it to `<checkpoint-name>`.<br>If `<checkpoint-name>` **already exists**, loads it, continues training, creates a numbered backup of the previous file (`<stem>_000.pth`, `<stem>_001.pth`, ... unless `-o` is passed), and saves the updated checkpoint back to `<checkpoint-name>`. |
| `python reinforcement.py train <checkpoint-name> -s <new-checkpoint-name>` | Loads `<checkpoint-name>` (if it exists), continues training, and saves the result under `<new-checkpoint-name>` (creating a numbered backup of `<new-checkpoint-name>` if it already exists, unless `-o` is passed).<br>If `<checkpoint-name>` **does not exist**, it is ignored and training starts from scratch, saving the result to `<new-checkpoint-name>`. |
| `python reinforcement.py train` | Trains a new model from scratch without saving a checkpoint (pass `-s` to save as `checkpoint.pth`, or `-s <name>` to choose a file name). |
| `python reinforcement.py train -s` | Trains a new model from scratch and saves it to the default `checkpoint.pth` (since neither `<checkpoint-name>` nor `<new-checkpoint-name>` was provided). |
| `python reinforcement.py replay <checkpoint-name>` | Loads `<checkpoint-name>` and runs rendered evaluation episodes (`--save` is not allowed in `replay` mode). |

**Backup naming:** Whenever saving would replace an existing file and `-o` / `--overwrite` is not set, the existing file is copied first to `<stem>_000<ext>`, `<stem>_001<ext>`, etc. (for example, `gmdh.pth` → `gmdh_000.pth`).

---

## Key Parameters

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `-t`, `--train-episodes` | `1000` | Number of DQN training episodes (executed after the initial `batch_size` warm-up episodes). |
| `-r`, `--replay-episodes` | `3` | Number of human-rendered `CartPole-v1` evaluation episodes run after training (or during `replay`). Pass `-r 0` with `train` to skip visual replay. |
| `-g`, `--gmdh` | *(default)* | Build a constructively grown sparse affine GMDH Q-network (`inputs=2`, `degree=1`, `interaction=0`, `Hardtanh(-2, 2)`). Mutually exclusive with `-d`. |
| `-d`, `--dense` | `False` | Build a compact dense MLP Q-network (`4 → 6 → 7 → 10 → 2`, `ReLU`) for baseline comparison. Mutually exclusive with `-g`. |
| `-o`, `--overwrite` | `False` | Overwrite an existing checkpoint file directly without creating a numbered backup (`_000`, `_001`, ...). |
| `-i`, `--info` | `False` | Print environment details, Q-network architecture, optimizer configuration, and GMDH layer-growth logs. |

Other useful training parameters:
- `-b`, `--batch_size` (default `32`): Experience-replay mini-batch size.
- `-L`, `--learning-rate` (default `1e-3`): `AdamW` learning rate.
- `--grow-episodes` (default `train_episodes // 10`): Episodes between GMDH constructive growth steps.
- `--max-neurons` (default `150`): Maximum candidate neurons per GMDH layer.
- `--dtype` (`float32` \| `float64`, default `float32`): Floating-point dtype for the Q-network and replay buffer.

---

## Controlling GMDH Growth & Pruning (`--max-layers` and `--select`)

GMDH models in `reinforcement.py` use solver-free constructive growth (`ModelIncrementer` with `solver=None` and `gradient_during_growth=True`):

- **`--max-layers N`** (default `4`):
  - **Training from scratch:** Starts with 1 eager GMDH layer and adds a new layer every `--grow-episodes` until `N` layers are reached. At the final growth step (`N`), the model applies `FINAL_SELECT` pruning and backward cascade trimming (`model.trim()`), then continues fine-tuning the pruned network for the remaining episodes.
  - **Retraining an existing checkpoint:** Restores the saved `max_layers` by default.
    - Passing a **larger** `--max-layers` than the checkpoint's current layer count allows `ModelIncrementer` to grow additional layers on top of the loaded network during continued training.
    - Leaving `--max-layers` unset (or equal to / smaller than the checkpoint's current layer count) keeps the loaded topology fixed and fine-tunes only its weights.

- **`--select SELECT FINAL_SELECT`** (default `0.01 0.1`):
  - Accepts **two relative importance thresholds** (`SelectRelativeThreshold`, in `[0, 1]`):
    1. **`SELECT`** (default `0.01`): Light pruning applied at **intermediate** growth steps—neurons whose importance score is below `SELECT * max_importance` are pruned before expanding to the next layer.
    2. **`FINAL_SELECT`** (default `0.1`): Stricter pruning applied at the **final** growth step (`max_layers`), immediately followed by backward cascade pruning (`model.trim()`) to remove unreferenced upstream neurons.
  - **Training from scratch:** Lower values keep wider layers; higher values prune more aggressively for a smaller hardware-friendly network.
  - **Retraining a checkpoint:** Overrides the checkpoint's saved selection thresholds—useful when expanding a checkpoint with a higher `--max-layers` and wanting tighter or looser pruning on the newly grown layers.

### Examples

```bash
# Train a new 4-layer GMDH agent from scratch and save to gmdh_run.pth (skipping replay window)
python reinforcement.py train gmdh_run.pth -g -t 1000 -r 0 -i

# Train a more aggressively pruned 5-layer GMDH model
python reinforcement.py train gmdh_5L.pth -g -t 1500 --max-layers 5 --select 0.02 0.15 -r 0 -i

# Continue training gmdh_run.pth in place (creates gmdh_run_000.pth backup automatically)
python reinforcement.py train gmdh_run.pth -t 500 -r 3

# Load gmdh_run.pth, expand from 4 to 5 layers with custom selection thresholds, and save as a new file
python reinforcement.py train gmdh_run.pth -s gmdh_run_5L.pth -t 800 --max-layers 5 --select 0.01 0.12 -i

# Train a dense MLP baseline and overwrite dense_run.pth without creating a backup
python reinforcement.py train dense_run.pth -d -t 1000 -o -r 0

# Replay an existing checkpoint and print its architecture
python reinforcement.py replay gmdh.pth -i -r 5
```

---

## CPU vs. GPU (`--gpu`) Note

> **Performance note:** Passing `--gpu` works whenever a CUDA GPU or Apple Silicon MPS device is available (otherwise `reinforcement.py` defaults to **CPU**). However, **at the moment our reinforcement learning training runs faster on CPU** than on GPU/MPS because the CartPole Q-networks are very small and single-environment rollout steps incur frequent CPU↔GPU transfer and kernel-launch overhead. We plan to investigate how to improve GPU utilization in future updates—until then, leaving `--gpu` off (CPU default) is recommended.
