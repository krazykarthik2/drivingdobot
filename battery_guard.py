import time
import os
import psutil
import subprocess

def check_battery_and_hibernate():
    b = psutil.sensors_battery()
    if not b:
        return
    percent = b.percent
    plugged = b.power_plugged
    print(f"[Battery Guard] Battery: {percent}%, Plugged in: {plugged}")
    if percent <= 5 and not plugged:
        print("[Battery Guard] CRITICAL BATTERY (<= 5%). Committing changes and hibernating!")
        subprocess.run(["git", "add", "."], check=False)
        subprocess.run(["git", "commit", "-m", "Auto-checkpoint: battery critical <= 5%"], check=False)
        subprocess.run(["git", "push"], check=False)
        # Hibernate windows
        subprocess.run(["shutdown", "/h"], check=False)
        sys.exit(0)

if __name__ == "__main__":
    check_battery_and_hibernate()
