import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_spatial")()
task = b.get_task(0)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)
obs = env.reset()

print("Available geom names in sim:")
for i in range(min(30, env.env.sim.model.ngeom)):
    print(f"Geom {i}: {env.env.sim.model.geom_id2name(i)}")

# Look for plate and bowl geoms
for i in range(env.env.sim.model.ngeom):
    name = env.env.sim.model.geom_id2name(i)
    if name and ('bowl' in name or 'plate' in name):
        pos = env.env.sim.data.geom_xpos[i]
        print(f"Geom: {name:<25} | Pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

env.close()
