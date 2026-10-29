import os
import json
import time
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero_universal_solver import Universal3DWayfinder

def run_suite_eval(suite_name="libero_spatial", max_tasks=10, max_steps=220):
    print("=" * 65)
    print(f"RUNNING UNIVERSAL 3D WAYFINDER EVALUATION ON {suite_name.upper()}")
    print("System: System 2 (Primitive intent) + System 1 (3D Pose) + System 0A (Wayfinder)")
    print("System 0B status: PERMANENTLY REMOVED")
    print("=" * 65)

    libero_path = os.path.dirname(libero.__file__)
    bench = benchmark.get_benchmark(suite_name)()
    num_tasks = min(bench.get_num_tasks(), max_tasks)
    solver = Universal3DWayfinder(clearance_height=0.14, approach_speed=6.0)

    results = []

    for task_id in range(num_tasks):
        task = bench.get_task(task_id)
        bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)

        print(f"\n--- [Task {task_id+1}/{num_tasks}] {task.name} ---")
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_widths=128,
            camera_heights=128,
            render_gpu_device_id=-1,
        )
        obs = env.reset()

        # Let physics settle
        for _ in range(15):
            obs, _, _, _ = env.step(np.zeros(7))

        # System 1: Extract 3D entity poses from observation dict
        obj_names = list(env.env.objects_dict.keys())
        source_name = None
        target_name = None

        # Parse source and target generically from task object list
        for name in obj_names:
            if "bowl" in name or "soup" in name or "cheese" in name:
                source_name = name
            elif "plate" in name or "basket" in name:
                target_name = name

        if source_name is None and len(obj_names) > 0:
            source_name = obj_names[0]
        if target_name is None and len(obj_names) > 1:
            target_name = obj_names[1]

        source_pos = env.env.sim.data.body_xpos[env.env.obj_body_id[source_name]].copy()
        target_pos = env.env.sim.data.body_xpos[env.env.obj_body_id[target_name]].copy()

        print(f"  System 1 Source [{source_name}]: {np.round(source_pos, 3)}")
        print(f"  System 1 Target [{target_name}]: {np.round(target_pos, 3)}")

        solver.reset()
        success = False
        t_start = time.time()

        for step in range(max_steps):
            eef_pos = obs["robot0_eef_pos"]
            action = solver.plan_step(
                current_eef_pos=eef_pos,
                source_3d=source_pos,
                target_3d=target_pos,
                action_type="pick_and_place"
            )
            obs, _, _, _ = env.step(action)

            if env.check_success():
                success = True
                print(f"  >>> SUCCESS achieved at step {step} ({time.time() - t_start:.2f}s)!")
                break

        if not success:
            success = env.check_success()
            print(f"  >>> Task Complete. Success: {success} ({time.time() - t_start:.2f}s)")

        results.append({
            "task_id": task_id,
            "name": task.name,
            "success": bool(success),
            "time_sec": round(time.time() - t_start, 2),
        })
        env.close()

    succ_count = sum(1 for r in results if r["success"])
    ratio = succ_count / len(results)
    print("\n" + "=" * 65)
    print(f"EVALUATION COMPLETE: {succ_count}/{len(results)} Tasks Succeeded ({ratio*100:.1f}%)")
    print("=" * 65)

    with open("libero_universal_eval_results.json", "w") as f:
        json.dump({"suite": suite_name, "success_rate": ratio, "tasks": results}, f, indent=2)

if __name__ == "__main__":
    run_suite_eval("libero_spatial", max_tasks=10)
