import sys
import json
import functools
import numpy as np

print("=== LOAD_PRETRAINED STEP-BY-STEP PROBE ===")

import octo_patch
from octo.model.octo_model import OctoModel, OctoModule
import orbax.checkpoint
import jax
import jax.numpy as jnp
import flax
import tensorflow as tf

checkpoint_path = "octo_small_checkpoint"

# 1. Config
print("[1] Reading config.json...")
with tf.io.gfile.GFile(f"{checkpoint_path}/config.json", "r") as f:
    config = json.load(f)
if "pred_horizon" in config["model"]["heads"]["action"]["kwargs"]:
    config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = config["model"]["heads"]["action"]["kwargs"].pop("pred_horizon")
print("  [OK] Config loaded.")

# 2. Example batch
print("[2] Reading example_batch.msgpack...")
with tf.io.gfile.GFile(f"{checkpoint_path}/example_batch.msgpack", "rb") as f:
    example_batch = flax.serialization.msgpack_restore(f.read())
if "tasks" in example_batch:
    example_batch["task"] = example_batch.pop("tasks")
if "timestep_pad_mask" not in example_batch["observation"]:
    example_batch["observation"]["timestep_pad_mask"] = example_batch["observation"]["pad_mask"]
print("  [OK] Example batch loaded.")

# 3. Dataset statistics
print("[3] Reading dataset_statistics.json...")
with tf.io.gfile.GFile(f"{checkpoint_path}/dataset_statistics.json", "r") as f:
    dataset_statistics = json.load(f)
    dataset_statistics = jax.tree_map(np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict))
print("  [OK] Dataset statistics loaded.")

# 4. OctoModule creation
print("[4] Creating OctoModule definition...")
module = OctoModule.create(**config["model"])
print("  [OK] Module created.")

# 5. jax.eval_shape params_shape
print("[5] Inferring parameter shapes via jax.eval_shape...")
init_args = (
    example_batch["observation"],
    example_batch["task"],
    example_batch["observation"]["timestep_pad_mask"],
)
params_shape = jax.eval_shape(
    functools.partial(module.init, train=False), jax.random.PRNGKey(0), *init_args
)["params"]
print("  [OK] Parameter shapes inferred successfully.")

# 6. Orbax checkpointer initialization
print("[6] Initializing Orbax CheckpointManager...")
checkpointer = orbax.checkpoint.CheckpointManager(
    checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
)
step = checkpointer.latest_step()
print(f"  [OK] Orbax step detected: {step}")

# 7. Checkpoint restore via TensorStore
print(f"[7] Restoring parameters from disk (Step {step}) via TensorStore...")
params = checkpointer.restore(step, params_shape)
print("  [OK] Parameters restored from checkpoint successfully!")

print("\n=== ALL SUB-STEPS OF LOAD_PRETRAINED PASSED ===")
