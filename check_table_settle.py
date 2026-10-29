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

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
print("Initial bowl Z:", env.env.sim.data.body_xpos[bowl_id][2])

# Step 0: pure zero action
obs, _, _, _ = env.step(np.zeros(7))
print("Bowl Z after zero action step 0:", env.env.sim.data.body_xpos[bowl_id][2])

for s in range(10):
    obs, _, _, _ = env.step(np.zeros(7))
print("Bowl Z after 10 zero steps:", env.env.sim.data.body_xpos[bowl_id][2])

env.close()
