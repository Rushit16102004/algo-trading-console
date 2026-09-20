import os
import json

KILL_SWITCH_FILE = "backend_engine/kill_switch_state.json"

def get_kill_switch_state() -> bool:
    """Returns True if the kill switch is active (trading halted), False otherwise."""
    if os.path.exists(KILL_SWITCH_FILE):
        try:
            with open(KILL_SWITCH_FILE, "r") as f:
                data = json.load(f)
                return data.get("active", False)
        except Exception:
            pass
    return False

def set_kill_switch_state(active: bool):
    """Sets the emergency kill switch state persistently."""
    try:
        with open(KILL_SWITCH_FILE, "w") as f:
            json.dump({"active": active}, f)
    except Exception as e:
        print(f"Error saving kill switch state: {e}")
