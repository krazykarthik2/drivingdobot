import inspect
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

print("BDDLBaseDomain file:", inspect.getfile(BDDLBaseDomain))
with open(inspect.getfile(BDDLBaseDomain), encoding="utf-8") as f:
    text = f.read()
    idx = text.find("class BDDLBaseDomain")
    print(text[idx:idx+800])
