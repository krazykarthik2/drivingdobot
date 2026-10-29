import os
import inspect
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

with open(inspect.getfile(BDDLBaseDomain), encoding="utf-8") as f:
    text = f.read()
    idx = text.find("def step")
    print(text[idx:idx+800])
