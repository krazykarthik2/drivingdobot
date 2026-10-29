import os
import sys

print("=== PROBING [4] MODULE INIT COMPONENTS ===")

import octo_patch
from octo.model.octo_model import OctoModel, OctoModule
import jax
import jax.numpy as jnp
import flax
import json

with open("octo_small_checkpoint/config.json") as f:
    config = json.load(f)
if "pred_horizon" in config["model"]["heads"]["action"]["kwargs"]:
    config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = config["model"]["heads"]["action"]["kwargs"].pop("pred_horizon")

with open("octo_small_checkpoint/example_batch.msgpack", "rb") as f:
    example_batch = flax.serialization.msgpack_restore(f.read())
if "tasks" in example_batch:
    example_batch["task"] = example_batch.pop("tasks")
if "timestep_pad_mask" not in example_batch["observation"]:
    example_batch["observation"]["timestep_pad_mask"] = example_batch["observation"]["pad_mask"]

# Sub-test 1: Check Tokenizers keys
print("\n[A] Octo Tokenizers in config:")
print("  Observation tokenizers:", list(config["model"]["observation_tokenizers"].keys()))
print("  Task tokenizers:", list(config["model"]["task_tokenizers"].keys()))

# Sub-test 2: Test Language Tokenizer (T5 Encoder) directly
print("\n[B] Testing FlaxT5EncoderModel setup...")
from transformers import AutoConfig, FlaxT5EncoderModel
cfg = AutoConfig.from_pretrained("t5-base")
print("  T5 AutoConfig loaded successfully.")
print("  Initializing FlaxT5EncoderModel(cfg)...")
t5_model = FlaxT5EncoderModel(cfg)
print("  [OK] FlaxT5EncoderModel loaded without segfault!")

# Sub-test 3: Run jax.eval_shape with CPU device context
print("\n[C] Running jax.eval_shape under CPU device...")
module = OctoModule.create(**config["model"])
init_args = (
    example_batch["observation"],
    example_batch["task"],
    example_batch["observation"]["timestep_pad_mask"],
)

with jax.default_device(jax.devices("cpu")[0]):
    shapes = jax.eval_shape(
        lambda *args: module.init(jax.random.PRNGKey(0), *args, train=False),
        *init_args
    )
print("  [SUCCESS] Parameter shapes inferred on CPU without segfault!")
