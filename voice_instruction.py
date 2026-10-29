"""
Real-time Voice Instruction Listener for Robot Control.
Supports microphone capture and text-to-speech feedback.
"""
import speech_recognition as sr
import threading

def listen_for_instruction(timeout=10, phrase_time_limit=8):
    """
    Listens to the default microphone for a spoken robot instruction.
    Returns: Recognized text string or None.
    """
    r = sr.Recognizer()
    r.energy_threshold = 400
    r.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            print("\n[MIC] Adjusting for ambient room noise (1 sec)...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            print("[MIC] Listening... SPEAK YOUR ROBOT INSTRUCTION NOW (e.g. 'pick up the red cube and place on green')")
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            print("[MIC] Processing speech audio...")
            
            # Using Google Speech Recognition (free, reliable, built into speech_recognition)
            text = r.recognize_google(audio)
            print(f"[MIC SUCCESS] You said: \"{text}\"")
            return text.strip()
    except sr.WaitTimeoutError:
        print("[MIC] Listening timed out (no speech detected).")
        return None
    except sr.UnknownValueError:
        print("[MIC] Could not understand audio.")
        return None
    except Exception as e:
        print(f"[MIC ERROR] {e}")
        return None

if __name__ == "__main__":
    res = listen_for_instruction()
    print("Result:", res)
