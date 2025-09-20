import requests
from ics import Calendar
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
import json
import re
import os
import sys
from dotenv import load_dotenv

# =====================
# LOAD ENV VARIABLES
# =====================
ENV_FILE = ".env.local"
load_dotenv(ENV_FILE)

LOCAL_TZ = ZoneInfo("Europe/Copenhagen")

# Read ICS URLs (comma-separated)
ICS_URLS = os.getenv("ICS_URLS")
if not ICS_URLS:
    print("Error: ICS_URLS not found in .env.local")
    sys.exit(1)

ics_urls = [url.strip() for url in ICS_URLS.split(",") if url.strip()]

# =====================
# NORMALIZE TIMES
# =====================
def normalize_time(time_str, start=True):
    h, m = map(int, time_str.split(":"))
    if start and h == 0 and 0 < m <= 10:
        return "00:00"
    if not start and h == 23 and m >= 50:
        return "23:59"
    return f"{h:02d}:{m:02d}"

# =====================
# EVENT HELPERS
# =====================
def is_excluded(event):
    """Return True if event start matches any EXDATE in raw ICS extras."""
    exdates = []
    for extra in event.extra:
        if extra.name == "EXDATE":
            values = extra.value if isinstance(extra.value, list) else [extra.value]
            for v in values:
                if isinstance(v, str):
                    try:
                        exdates.append(datetime.strptime(v, "%Y%m%dT%H%M%S"))
                    except ValueError:
                        continue
                else:
                    exdates.append(v)
    return any(
        event.begin.datetime.replace(tzinfo=None) == exdate.replace(tzinfo=None)
        for exdate in exdates if hasattr(exdate, "replace")
    )

# =====================
# FETCH AND PARSE ALL CALENDARS
# =====================
output = []
now = datetime.now(tz=LOCAL_TZ)

for ICS_URL in ics_urls:
    try:
        response = requests.get(ICS_URL, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ICS feed {ICS_URL}: {e}")
        continue

    calendar = Calendar(response.text)

    for event in calendar.events:
        start = event.begin.datetime.astimezone(LOCAL_TZ)
        end = event.end.datetime.astimezone(LOCAL_TZ)

        # Skip cancelled or excluded events
        status = getattr(event, "status", "") or ""
        if status.lower() == "cancelled":
            continue
        if is_excluded(event):
            continue

        # Skip past events
        if end < now:
            continue

        # Only include events within today and the next 6 days
        days_ahead = (start.date() - now.date()).days
        if days_ahead < 0 or days_ahead > 6:
            continue

        event_text = f"{event.name or ''} {event.description or ''}"

        # Extract distance in km
        match_km = re.search(r"(\d+)\s*km", event_text, re.IGNORECASE)
        distance = int(match_km.group(1)) if match_km else None

        # Extract manual kWh
        match_kwh = re.search(r"trip\s*:(\d+(?:\.\d+)?)\s*kwh", event_text, re.IGNORECASE)
        trip_kwh = float(match_kwh.group(1)) if match_kwh else None

        # Extract max SOC %
        match_max_soc = re.search(r"max\s*:\s*(\d+)\s*%", event_text, re.IGNORECASE)
        max_soc_pct = float(match_max_soc.group(1)) / 100 if match_max_soc else None

        # Extract supercharge kWh
        match_sc_kwh = re.search(r"sc\s*:(\d+(?:\.\d+)?)\s*kwh", event_text, re.IGNORECASE)
        sc_kwh = float(match_sc_kwh.group(1)) if match_sc_kwh else None

        if distance is not None:
            away_start = normalize_time(start.strftime("%H:%M"), start=True)
            away_end = normalize_time(end.strftime("%H:%M"), start=False)

            output.append({
                "day": start.strftime("%A").lower(),
                "away_start": away_start,
                "away_end": away_end,
                "distance_km": distance,
                "trip_kwh": trip_kwh,
                "supercharge_kwh": sc_kwh,
                "max_soc_pct": max_soc_pct
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
env_vars = {}
if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, val = line.split("=", 1)
                env_vars[key] = val

env_vars["TRIPS"] = json.dumps(output, separators=(",", ":"))

with open(ENV_FILE, "w") as f:
    for key, val in env_vars.items():
        f.write(f"{key}={val}\n")

print(f"Updated TRIPS in {ENV_FILE} with {len(output)} events from {len(ics_urls)} calendars.")
