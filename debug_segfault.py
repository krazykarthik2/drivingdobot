import os
import sys

print("=== SEGFAULT PINPOINT DIAGNOSTIC ===")
print(f"PID: {os.getpid()}")

# Test 1: JAX CUDA Allocation
print("\n[Step 1] Initializing JAX CUDA...")
import jax
import jax.numpy as jnp
x = jnp.ones((100, 100))
print("  JAX CUDA Success! Devices:", jax.devices())

# Test 2: TensorFlow CUDA Conflict
print("\n[Step 2] Importing TensorFlow (Checking GPU conflict with JAX)...")
import tensorflow as tf
print("  TensorFlow version:", tf.__version__)
print("  Visible physical GPUs to TF:", tf.config.list_physical_devices('GPU'))

# Test 3: Octo patch and imports
print("\n[Step 3] Loading octo_patch...")
import octo_patch
print("  octo_patch loaded.")

# Test 4: Tensorstore & Orbax Checkpointing
print("\n[Step 4] Importing TensorStore & Orbax...")
import tensorstore as ts
print("  Tensorstore version:", getattr(ts, '__version__', 'ok'))
import orbax.checkpoint as ocp
print("  Orbax loaded.")

# Test 5: Octo Model definition
print("\n[Step 5] Importing OctoModel...")
from octo.model.octo_model import OctoModel
print("  OctoModel class imported.")

# Test 6: Checkpoint Load
print("\n[Step 6] Running OctoModel.load_pretrained('octo_small_checkpoint')...")
model = OctoModel.load_pretrained("octo_small_checkpoint")
print("  OctoModel LOADED SUCCESSFULLY!")

print("\n=== ALL STEPS PASSED WITHOUT SEGFAULT ===")
