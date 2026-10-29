import sys

print("=== STEP BY STEP OCTO_PATCH IMPORT PROBE ===")

print("[1] Setting env...")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("  env set.")

print("[2] Importing JAX...")
import jax
print(f"  JAX backend: {jax.default_backend()}")

print("[3] Importing JAX submodules...")
import jax.numpy as jnp
import jax.tree_util
import jax.interpreters.xla
print("  JAX submodules imported.")

print("[4] Importing protobuf...")
import google.protobuf
print("  Protobuf imported.")

print("[5] Importing tensorflow...")
import tensorflow as tf
print(f"  TensorFlow version: {tf.__version__}")

print("[6] Importing transformers...")
from transformers import AutoConfig, FlaxT5EncoderModel
print("  Transformers T5 imported.")

print("[7] Importing octo...")
import octo
print("  Octo imported.")

print("=== ALL PASSED ===")
