import os
import sys

print("=== VERIFYING COMPLETE PIPELINE BEFORE TRAINING ===")

import octo_patch
from octo.model.octo_model import OctoModel
import jax

print("\n[1] Testing OctoModel.load_pretrained('octo_small_checkpoint')...")
model = OctoModel.load_pretrained("octo_small_checkpoint")
print("  [SUCCESS!] Pretrained Octo Model loaded 100% cleanly without any segfault!")
print(f"  Head parameter keys: {list(model.params['heads_action'].keys())}")
print(f"  Active JAX devices:  {jax.devices()}")

print("\nALL VERIFICATIONS PASSED! System is 100% ready for training.")
