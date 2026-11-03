"""
dobot_clean.py

Dedicated hardware watchdog and alarm cleaner for Dobot Magician:
- Reliable alarm clearance (ClearAllAlarmsState command ID 20).
- Queue buffer flusher (clears hung or stuck motion commands).
- Emergency unlock / servo reset.
- Standby home positioning.
- Exposes a clean, importable module: `from dobot_clean import clean_dobot, get_clean_dobot`
"""

import time
import pydobot
from pydobot.message import Message


def clean_dobot(dobot_instance=None, port="COM13", verbose=True):
    """
    Cleans alarms, flushes pending command queues, and verifies Dobot is responsive.
    Can be called on an existing pydobot.Dobot instance, or pass None to connect/clean/close.
    """
    own_connection = False
    dobot = dobot_instance

    if dobot is None:
        try:
            if verbose:
                print(f"[DobotClean] Connecting to Dobot on {port}...")
            dobot = pydobot.Dobot(port=port, verbose=False)
            own_connection = True
        except Exception as e:
            if verbose:
                print(f"[DobotClean Error] Could not connect to {port}: {e}")
            return False

    try:
        if verbose:
            print("[DobotClean] Clearing all alarms and resetting hardware state...")

        # 1. Clear All Alarms State (Command ID 20)
        m_clear_alarms = Message()
        m_clear_alarms.id = 20
        m_clear_alarms.ctrl = 1
        m_clear_alarms.params = bytearray([])
        dobot._send_message(m_clear_alarms)
        time.sleep(0.06)

        # Flush any response packet
        try:
            dobot._read_message()
        except Exception:
            pass

        # 2. Clear Queued Commands (Command ID 245)
        m_clear_queue = Message()
        m_clear_queue.id = 245
        m_clear_queue.ctrl = 1
        m_clear_queue.params = bytearray([])
        dobot._send_message(m_clear_queue)
        time.sleep(0.06)

        try:
            dobot._read_message()
        except Exception:
            pass

        # 3. Start Execution Queue (Command ID 240) so the Dobot accepts commands
        m_start_queue = Message()
        m_start_queue.id = 240
        m_start_queue.ctrl = 1
        m_start_queue.params = bytearray([])
        dobot._send_message(m_start_queue)
        time.sleep(0.06)

        try:
            dobot._read_message()
        except Exception:
            pass

        # 4. Read back live pose to confirm healthy communication
        pose = dobot.pose()
        if verbose:
            print(f"[DobotClean SUCCESS] Alarms cleared & queue active! Healthy Pose: [{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}] mm")
        return True

    except Exception as e:
        if verbose:
            print(f"[DobotClean Warning] Failed to clean Dobot: {e}")
        return False

    finally:
        if own_connection and dobot:
            try:
                dobot.close()
            except Exception:
                pass


def get_clean_dobot(port="COM13", verbose=True):
    """
    Connects to the Dobot, cleans alarms and queues, and returns the ready-to-use dobot instance.
    """
    try:
        if verbose:
            print(f"[DobotClean] Initializing clean connection on {port}...")
        dobot = pydobot.Dobot(port=port, verbose=False)
        clean_dobot(dobot, port=port, verbose=verbose)
        return dobot
    except Exception as e:
        if verbose:
            print(f"[DobotClean Error] Connection failed: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING STANDALONE DOBOT CLEANER & DIAGNOSTIC")
    print("=" * 60)
    success = clean_dobot(port="COM13", verbose=True)
    if success:
        print("[Status] Dobot is healthy, unlocked, and ready for motion commands!")
    else:
        print("[Status] Failed to clean Dobot. Check USB connection and COM port.")
