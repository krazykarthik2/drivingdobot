import os
import inspect
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

with open(inspect.getfile(BDDLBaseDomain), encoding="utf-8") as f:
    text = f.read()
    idx = text.find("def _setup_arena")
    if idx == -1:
        idx = text.find("def _load_model")
    print(text[idx:idx+1200])
