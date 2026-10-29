from robosuite.environments.manipulation import pick_place

with open(pick_place.__file__, encoding="utf-8") as f:
    lines = f.readlines()
    for j in range(640, 660):
        print(f"Line {j+1}: {lines[j].rstrip()}")
