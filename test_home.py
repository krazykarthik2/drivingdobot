import pydobot
from serial.tools import list_ports
import time
from pydobot.message import Message

def force_home(device):
    """Sends the raw hardware HOME calibration command with 0-byte payload."""
    msg = Message()
    msg.id = 31 # SetHOMECmd
    msg.ctrl = 0x03 # Write + Queued
    msg.params = bytearray() # ZERO BYTES (was incorrectly 4 bytes before)
    device._send_command(msg)

def main():
    ports = list_ports.comports()
    if not ports:
        print("No ports found")
        return
        
    p = ports[0].device
    print(f"Connecting to {p}...")
    device = pydobot.Dobot(port=p, verbose=True)
    
    print("\nFetching current pose...")
    x, y, z, r, j1, j2, j3, j4 = device.pose()
    print(f"Current Position: X={x}, Y={y}, Z={z}")
    
    if z < 0:
        print("\n[WARNING] Your robot's Z-axis is negative (below the desk)! ")
        print("The firmware is silently rejecting all movement commands to prevent crashing.")
        print("We MUST run the hardware HOME calibration to unlock it.")
        
    print("\n--- ATTEMPTING HARDWARE HOME CALIBRATION ---")
    force_home(device)
    
    print("Home command sent. Please wait for the robot to move...")
    time.sleep(15) # Wait for homing to complete
    
    print("Done!")
    device.close()

if __name__ == "__main__":
    main()
