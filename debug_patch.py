import sys
print("Debugging octo_patch line by line...")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_FLAX"] = "1"
print("[1] env set")

import tensorflow as tf
print("[2] tensorflow imported")

import jax
import jax.numpy as jnp
import jax.tree_util
import jax.interpreters.xla
print("[3] jax imported")

import google.protobuf
print("[4] protobuf imported")

print("[5] importing transformers...")
import transformers
print("[5.1] transformers imported")

print("[6] importing flax auto...")
import transformers.models.auto.modeling_flax_auto as _flax_auto
print("[6.1] flax auto imported")

print("[7] importing flax t5...")
import transformers.models.t5.modeling_flax_t5 as _flax_t5
print("[7.1] flax t5 imported")

print("[8] importing octo tokenizers...")
import octo.model.components.tokenizers as _tok
print("[8.1] octo tokenizers imported")

print("SUCCESS: No crash inside patch components!")
