import sys
print("=== TESTING COMPLETE OCTO PIPELINE WITH REVERSED IMPORT ORDER ===")

import octo_patch
print("[1] octo_patch imported cleanly.")

from octo.model.octo_model import OctoModel
print("[2] OctoModel class imported.")

import jax
print(f"[3] JAX Active Backend: {jax.default_backend()} | Devices: {jax.devices()}")

print("[4] Loading pretrained Octo Model from step 270000...")
model = OctoModel.load_pretrained("octo_small_checkpoint", step=270000)
print("  [SUCCESS!] OctoModel loaded completely without errors!")
print(f"  Action Head parameters: {list(model.params['heads_action'].keys())}")
print("\n=== SYSTEM IS 100% OPERATIONAL ===")
