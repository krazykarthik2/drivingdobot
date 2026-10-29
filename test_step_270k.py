import os
import sys
import json
import functools
import numpy as np

print("=== TESTING LOAD_PRETRAINED WITH EXACT STEP 270000 ===")

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

# 3. OctoModule creation
print("[3] Creating OctoModule definition...")
module = OctoModule.create(**config["model"])

# 4. jax.eval_shape params_shape
print("[4] Inferring parameter shapes via jax.eval_shape...")
init_args = (
    example_batch["observation"],
    example_batch["task"],
    example_batch["observation"]["timestep_pad_mask"],
)
params_shape = jax.eval_shape(
    functools.partial(module.init, train=False), jax.random.PRNGKey(0), *init_args
)["params"]
print("  [OK] Parameter shapes inferred successfully.")

# 5. Direct Restore using PyTreeCheckpointer on step 270000 directly
step_270k_path = os.path.join(checkpoint_path, "270000", "default")
print(f"[5] Restoring parameters directly from: {step_270k_path}...")
ckpter = orbax.checkpoint.PyTreeCheckpointer()
params = ckpter.restore(step_270k_path, params_shape)
print("  [SUCCESS!] Parameters restored directly from step 270000 without Orbax manager error!")

# 6. Full OctoModel object assembly
print("[6] Assembling full OctoModel instance...")
with tf.io.gfile.GFile(f"{checkpoint_path}/dataset_statistics.json", "r") as f:
    dataset_statistics = json.load(f)
    dataset_statistics = jax.tree_map(np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict))

model = OctoModel(
    module=module,
    params=params,
    text_processor=None,
    example_batch=example_batch,
    config=config,
    dataset_statistics=dataset_statistics
)
print("  [COMPLETE SUCCESS!] OctoModel is fully loaded and ready for training!")
