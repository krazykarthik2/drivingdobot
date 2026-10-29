import os
import sys
import json
import libero
from libero.libero import benchmark
from system2_reasoning import System2ReasoningEngine

suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
engine = System2ReasoningEngine()

print("=" * 80)
print("TESTING SYSTEM 2 (VLM REASONING) ON LIBERO BENCHMARK TASKS")
print("=" * 80)

sample_tasks = [
    ('libero_spatial', 0),
    ('libero_spatial', 2),
    ('libero_object', 0),
    ('libero_goal', 0),
    ('libero_10', 0)
]

for suite_name, task_idx in sample_tasks:
    b = benchmark.get_benchmark(suite_name)()
    task = b.get_task(task_idx)
    print(f"\n[{suite_name.upper()} | Task {task_idx+1}]")
    print(f"Instruction: '{task.language}'")
    res = engine.parse_instruction(task.language)
    print("VLM Reasoning Target Spec:")
    print(json.dumps(res, indent=2))
