import pygame
import time
import struct
import sys
from serial.tools import list_ports
import pydobot
from pydobot.message import Message

def clear_alarms_and_init_queue(device):
    """Fixes pydobot's queue bug and clears hardware alarms."""
    # 1. Clear Queue (245)
    msg_clear = Message()
    msg_clear.id = 245
    msg_clear.ctrl = 0x01
    device._send_command(msg_clear)
    time.sleep(0.1)
    
    # 2. Start Execution Queue (240)
    msg_start = Message()
    msg_start.id = 240
    msg_start.ctrl = 0x01
    device._send_command(msg_start)
    time.sleep(0.1)

    # 3. Clear Alarms (20)
    msg_alarm = Message()
    msg_alarm.id = 20
    msg_alarm.ctrl = 0x01
    msg_alarm.params = bytearray()
    device._send_command(msg_alarm)
    time.sleep(0.5)

def set_jog_params(device):
    """Sets up the velocities and accelerations for JOG movements."""
    msg1 = Message()
    msg1.id = 71
    msg1.ctrl = 0x03 # MUST be Queued (0x03) or the robot rejects it!
    msg1.params = bytearray(struct.pack('ff', 100.0, 100.0))
    device._send_command(msg1)

    msg2 = Message()
    msg2.id = 70
    msg2.ctrl = 0x03 # MUST be Queued (0x03)
    msg2.params = bytearray(struct.pack('8f', 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0))
    device._send_command(msg2)

def set_jog_cmd(device, isJoint, cmd):
    """Sends a JOG movement command."""
    msg = Message()
    msg.id = 73
    msg.ctrl = 0x01 # Write + Immediate
    msg.params = bytearray(struct.pack('BB', isJoint, cmd))
    device._send_command(msg)

def set_home_cmd(device):
    """Sends the Dobot back to its home calibration point directly."""
    msg = Message()
    msg.id = 31
    msg.ctrl = 0x03 # Home MUST be queued (0x03)
    msg.params = bytearray() # ZERO BYTES
    device._send_command(msg)

def set_suction(device, enable):
    """Overrides pydobot's queued suction with an immediate one."""
    msg = Message()
    msg.id = 62 # SET_END_EFFECTOR_SUCTION_CUP
    msg.ctrl = 0x01 # Write + Immediate
    msg.params = bytearray([0x01, 0x01 if enable else 0x00])
    device._send_command(msg)

def main():
    print("Initializing Controller...")
    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("Dobot Joystick (Pure Python)")
    
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick found! Exiting.")
        sys.exit(1)

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Controller Connected: {joystick.get_name()}")
    
    print("Searching for DoBot...")
    ports = list_ports.comports()
    if not ports:
        print("No serial ports found! Plug in the DoBot.")
        sys.exit(1)
        
    target_port = ports[0].device
    print(f"Connecting to DoBot on {target_port}...")
    
    try:
        device = pydobot.Dobot(port=target_port, verbose=False)
        clear_alarms_and_init_queue(device)
        print("Successfully connected via pure serial protocol!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    set_jog_params(device)
    print("JOG limits set. Ready to move!")
    
    last_cmd = 0 # 0 = Idle
    suction_state = False
    
    # JOG Commands
    JOG_IDLE = 0
    JOG_X_PLUS = 1
    JOG_X_MINUS = 2
    JOG_Y_PLUS = 3
    JOG_Y_MINUS = 4
    JOG_Z_PLUS = 5
    JOG_Z_MINUS = 6

    try:
        while True:
            # 1. Process discrete events (like pushing a button once)
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button {event.button} Pressed!")
                    if event.button == 1: # B button
                        suction_state = not suction_state
                        print(f"Toggling Suction -> {suction_state}")
                        set_suction(device, suction_state)
                    elif event.button == 11: # Right stick click
                        print("Going HOME! (Immediate Mode)")
                        set_home_cmd(device)

            # 2. Continuous State Polling for JOG movements
            cmd = JOG_IDLE
            
            if joystick.get_button(0): # A button (Z-)
                cmd = JOG_Z_MINUS
            elif joystick.get_button(3): # Y button (Z+)
                cmd = JOG_Z_PLUS
            else:
                # Check Axes
                y_axis = joystick.get_axis(1)
                x_axis = joystick.get_axis(0)
                
                if y_axis < -0.2:
                    cmd = JOG_X_PLUS # Forward
                elif y_axis > 0.2:
                    cmd = JOG_X_MINUS # Backward
                elif x_axis < -0.2:
                    cmd = JOG_Y_PLUS # Left
                elif x_axis > 0.2:
                    cmd = JOG_Y_MINUS # Right

            # Send JOG Command over Serial ONLY when it changes
            if cmd != last_cmd:
                print(f"Sending JOG Command: {cmd}")
                set_jog_cmd(device, isJoint=1, cmd=cmd)
                last_cmd = cmd

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        set_jog_cmd(device, 1, 0) # Force stop
        device.close()
        pygame.quit()

if __name__ == "__main__":
    main()
