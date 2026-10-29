import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_spatial")()
task = b.get_task(8)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)

for offset in [0.035, 0.040, 0.045, 0.050, 0.055]:
    obs = env.reset()
    for _ in range(15):
        obs, _, _, _ = env.step(np.zeros(7))
    
    f1 = env.env.sim.model.geom_name2id("gripper0_finger1_pad_collision")
    f2 = env.env.sim.model.geom_name2id("gripper0_finger2_pad_collision")
    env.env.sim.model.geom_friction[f1][0] = 50.0
    env.env.sim.model.geom_friction[f2][0] = 50.0

    b_id = env.env.obj_body_id["akita_black_bowl_1"]
    b_pos = env.env.sim.data.body_xpos[b_id].copy()

    # Pre-approach
    pre_goal = np.array([b_pos[0] + offset, b_pos[1], 1.15])
    for _ in range(35):
        curr = env.env._get_observations()["robot0_eef_pos"]
        diff = pre_goal - curr
        env.step(np.clip(np.concatenate([diff * 8.0, [0, 0, 0, -1]]), -0.6, 0.6))

    # Descend
    grasp_goal = np.array([b_pos[0] + offset, b_pos[1], 0.920])
    for _ in range(40):
        curr = env.env._get_observations()["robot0_eef_pos"]
        diff = grasp_goal - curr
        env.step(np.clip(np.concatenate([diff * 8.0, [0, 0, 0, -1]]), -0.5, 0.5))

    # Clamp
    for _ in range(30):
        env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

    # Lift
    for _ in range(35):
        env.step(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))

    curr_b = env.env.sim.data.body_xpos[b_id]
    dz = curr_b[2] - b_pos[2]
    print(f"Offset {offset:+.3f} | Final Bowl Pos: {curr_b} | Delta Z: {dz:+.4f}m")

env.close()
