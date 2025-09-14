import requests
from ics import Calendar
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
import json
import re
import os
import sys

ENV_FILE = ".env.local"

from dotenv import load_dotenv
load_dotenv(ENV_FILE)

LOCAL_TZ = ZoneInfo("Europe/Copenhagen")  # Your local timezone

# Read ICS_URL from environment
ICS_URL = os.getenv("ICS_URL")
if not ICS_URL:
    print("Error: ICS_URL not found in .env.local")
    sys.exit(1)

# =====================
# FETCH CALENDAR WITH ERROR HANDLING
# =====================
try:
    response = requests.get(ICS_URL, timeout=10)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Error fetching ICS feed: {e}")
    print("Keeping previous .env.local (if it exists).")
    sys.exit(0)

calendar = Calendar(response.text)

# =====================
# PARSE EVENTS
# =====================
output = []
now = datetime.now(tz=LOCAL_TZ)

for event in calendar.events:
    start = event.begin.datetime.astimezone(LOCAL_TZ)
    end = event.end.datetime.astimezone(LOCAL_TZ)

    # Skip past events
    if end < now:
        continue

    event_text = f"{event.name or ''} {event.description or ''}"

    # Extract distance in km
    match_km = re.search(r"(\d+)\s*km", event_text, re.IGNORECASE)
    distance = int(match_km.group(1)) if match_km else None

    # Extract manual kWh
    match_kwh = re.search(r"(\d+(\.\d+)?)\s*kwh", event_text, re.IGNORECASE)
    trip_kwh = float(match_kwh.group(1)) if match_kwh else None

    if distance is not None:
        output.append({
            "day": start.strftime("%A").lower(),
            "away_start": start.strftime("%H:%M"),
            "away_end": end.strftime("%H:%M"),
            "distance_km": distance,
            "trip_kwh": trip_kwh
        })

# =====================
# SORT CHRONOLOGICALLY BY WEEKDAY AND START TIME
# =====================
weekday_order = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6
}
output.sort(key=lambda x: (weekday_order[x["day"]], x["away_start"]))

# =====================
# UPDATE TRIPS IN .env.local
# =====================
# Read existing .env.local to preserve other variables
env_vars = {}
if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, val = line.split("=", 1)
                env_vars[key] = val

# Update TRIPS variable
env_vars["TRIPS"] = json.dumps(output, separators=(",", ":"))

# Write back all variables
with open(ENV_FILE, "w") as f:
    for key, val in env_vars.items():
        f.write(f"{key}={val}\n")

print(f"Updated TRIPS in {ENV_FILE} with {len(output)} events.")
