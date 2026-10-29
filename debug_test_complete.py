import os
import sys

print("[1] Script started.")

import octo_patch
print("[2] octo_patch loaded.")

from octo.model.octo_model import OctoModel
print("[3] OctoModel imported.")

print("[4] Loading model with step=270000...")
try:
    model = OctoModel.load_pretrained("octo_small_checkpoint", step=270000)
    print("SUCCESS: Model loaded with step=270000!")
except Exception as e:
    print(f"FAILED with exception: {e}")
