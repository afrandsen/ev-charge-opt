import requests
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+
import json
import re
import os
import sys
from dotenv import load_dotenv

# recurrence expansion library + ical parser
import icalendar
import recurring_ical_events

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

# helper to convert ical dt to timezone-aware datetime in LOCAL_TZ
def to_local_dt(dt):
    # dt can be date or datetime
    if isinstance(dt, date) and not isinstance(dt, datetime):
        # all-day: treat as midnight
        return datetime.combine(dt, time(0, 0)).replace(tzinfo=LOCAL_TZ)
    # datetime case
    if dt.tzinfo is None:
        # floating time - assume local timezone (same behavior as your previous approach)
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)

# =====================
# FETCH AND PARSE ALL CALENDARS WITH RECURRENCE EXPANSION
# =====================
output = []
now = datetime.now(tz=LOCAL_TZ)

# Build query window: today .. next 6 days (inclusive)
start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
# include whole last day by adding 6 days + 23:59:59
query_end = start_of_today + timedelta(days=6, hours=23, minutes=59, seconds=59)

seen_occurrences = set()  # to deduplicate by (uid, occurrence-start-iso)

for ICS_URL in ics_urls:
    try:
        resp = requests.get(ICS_URL, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ICS feed {ICS_URL}: {e}")
        continue

    # parse with icalendar
    try:
        cal = icalendar.Calendar.from_ical(resp.content)
    except Exception as e:
        print(f"Error parsing ICS from {ICS_URL}: {e}")
        continue

    # expand recurrences and yield actual occurrences in the date/time window
    try:
        occurrences = recurring_ical_events.of(cal).between(start_of_today, query_end)
    except Exception as e:
        print(f"Error expanding recurrences for {ICS_URL}: {e}")
        continue

    for comp in occurrences:
        # comp is a icalendar component (VEVENT-like) for a single occurrence

        # get DTSTART
        dtstart_prop = comp.get("DTSTART")
        if not dtstart_prop:
            continue
        dtstart_val = dtstart_prop.dt
        start_dt = to_local_dt(dtstart_val)

        # get DTEND (fallback to DTSTART if missing)
        dtend_prop = comp.get("DTEND")
        if dtend_prop:
            dtend_val = dtend_prop.dt
            end_dt = to_local_dt(dtend_val)
        else:
            # try DURATION
            dur = comp.get("DURATION")
            if dur:
                try:
                    end_dt = start_dt + dur.dt
                except Exception:
                    end_dt = start_dt
            else:
                end_dt = start_dt

        # Skip past events
        if end_dt < now:
            continue

        # Only include events within today and the next 6 days (based on start date)
        days_ahead = (start_dt.date() - now.date()).days
        if days_ahead < 0 or days_ahead > 6:
            continue

        # dedupe by UID + occurrence start
        uid_prop = comp.get("UID")
        uid = str(uid_prop) if uid_prop else None
        occ_key = (uid, start_dt.isoformat()) if uid else (start_dt.isoformat(), comp.get("SUMMARY"))
        if occ_key in seen_occurrences:
            continue
        seen_occurrences.add(occ_key)

        # Build searchable text from SUMMARY and DESCRIPTION
        summary = comp.get("SUMMARY")
        description = comp.get("DESCRIPTION")
        # ical properties may be of type vText etc. convert to str safely
        event_text = f"{str(summary) if summary else ''} {str(description) if description else ''}"

        # Extract distance in km
        match_km = re.search(r"(\d+)\s*km", event_text, re.IGNORECASE)
        distance = int(match_km.group(1)) if match_km else None

        # Extract manual kWh
        match_kwh = re.search(r"trip\s*:\s*(\d+(?:\.\d+)?)\s*kwh", event_text, re.IGNORECASE)
        trip_kwh = float(match_kwh.group(1)) if match_kwh else None

        # Extract max SOC %
        match_max_soc = re.search(r"max\s*:\s*(\d+)\s*%", event_text, re.IGNORECASE)
        max_soc_pct = float(match_max_soc.group(1)) / 100 if match_max_soc else None

        # Extract supercharge kWh
        match_sc_kwh = re.search(r"sc\s*:\s*(\d+(?:\.\d+)?)\s*kwh", event_text, re.IGNORECASE)
        sc_kwh = float(match_sc_kwh.group(1)) if match_sc_kwh else None

        if distance is not None:
            away_start = normalize_time(start_dt.strftime("%H:%M"), start=True)
            away_end = normalize_time(end_dt.strftime("%H:%M"), start=False)

            output.append({
                "day": start_dt.strftime("%A").lower(),
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
output.sort(key=lambda x: (weekday_order.get(x["day"], 7), x["away_start"]))

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
