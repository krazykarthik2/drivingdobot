"""
test_smolvla_forward.py

Runs a single forward pass test on SmolVLAPolicy with the exact input features
discovered from the config:
- observation.state: (1, 6)
- observation.images.camera1: (1, 3, 256, 256)
- observation.images.camera2: (1, 3, 256, 256)
- observation.images.camera3: (1, 3, 256, 256)
- observation.language.tokens / tokenized prompt
"""

import sys
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

import huggingface_hub
if not hasattr(huggingface_hub, 'sync_bucket'):
    huggingface_hub.sync_bucket = None

import transformers.utils
if not hasattr(transformers.utils, 'torch_compilable_check'):
    transformers.utils.torch_compilable_check = lambda: False

import torch
import numpy as np
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("[Test] Loading tokenizer & policy...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.eval()

# Construct dummy batch
instruction = "pick up the red block and place it on the red plate"
tokens = tokenizer(instruction, return_tensors="pt", padding="max_length", max_length=64, truncation=True)

img = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
state = torch.zeros((1, 6), dtype=torch.float32)

batch = {
    "observation.state": state,
    "observation.images.camera1": img,
    "observation.images.camera2": img,
    "observation.images.camera3": img,
    "observation.language.tokens": tokens["input_ids"],
    "observation.language.attention_mask": tokens["attention_mask"],
}

print("[Test] Testing select_action...")
with torch.no_grad():
    action = policy.select_action(batch)
print("[Test] Action output:", type(action), action.shape if hasattr(action, 'shape') else action)
