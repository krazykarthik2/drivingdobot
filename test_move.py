import pydobot
from serial.tools import list_ports
import time
from pydobot.message import Message

def clear_alarms(device):
    msg = Message()
    msg.id = 20 # ClearAllAlarmsState
    msg.ctrl = 0x01
    msg.params = bytearray()
    device._send_command(msg)

def main():
    ports = list_ports.comports()
    if not ports:
        print("No ports found")
        return
        
    p = ports[0].device
    print(f"Connecting to {p}...")
    device = pydobot.Dobot(port=p, verbose=True)
    
    print("Clearing alarms...")
    clear_alarms(device)
    time.sleep(0.5)
    
    print("Fetching current pose...")
    x, y, z, r, j1, j2, j3, j4 = device.pose()
    print(f"Current Position: X={x}, Y={y}, Z={z}")
    
    print("\n--- ATTEMPTING MOVEMENT ---")
    print("Moving Z up by 20mm...")
    
    # Send native PTP command
    device.move_to(x, y, z + 20, r, wait=True)
    
    print("Movement command finished.")
    device.close()

if __name__ == "__main__":
    main()
