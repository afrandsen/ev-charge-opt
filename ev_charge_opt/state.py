import json
import os


STATE_FILE = os.path.expanduser("~/repos/ev-charge-opt/tmp/ev_charging_state.json")


def load_state() -> dict:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(update: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state = load_state()
    state.update(update)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def should_notify(current_amp: int, last_amp: int):
    if last_amp == 0 and current_amp > 0:
        return True, f"🔌 Charging started at {current_amp}A"
    if last_amp > 0 and current_amp == 0:
        return True, "⏹️ Charging stopped"
    if last_amp > 0 and current_amp > 0 and current_amp != last_amp:
        return True, f"⚡ Amps changed from {last_amp}A to {current_amp}A"
    return False, "No change"
