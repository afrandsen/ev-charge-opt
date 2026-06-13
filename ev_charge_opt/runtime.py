import json
import os
import sys
from dataclasses import dataclass

import pandas as pd


@dataclass
class RuntimeInputs:
    initial_soc_pct: float
    is_home: bool
    eff_kwh_per_km: float
    charge_eff: float


@dataclass
class AppConfig:
    battery_kwh: float = 75
    charger_kw: float = 11
    charger_min_a: float = 5
    charger_volt: float = 400
    phases: int = 3
    solar_eff: float = 0.97
    panel_area: float = 11.5
    panel_eff: float = 0.2046
    noct_c: float = 41.0
    panel_temp_beta: float = 0.003
    panel_temp_ref_c: float = 25.0
    solar_max_kwh: float = 2.5
    systemtarif: float = 0.09000
    nettarif_tso: float = 0.05375
    elafgift: float = 0.01000
    tillaeg: float = 0.00000
    refusion: float = 0.0
    tilt: float = 25
    azimuth: float = 0
    tz: str = "Europe/Copenhagen"


def parse_runtime_inputs(argv, log) -> RuntimeInputs:
    if len(argv) < 5:
        log("No SOC/IS_HOME/EFF/CHARGE_EFF provided!")
        sys.exit(1)

    try:
        initial_soc_pct = float(argv[1])
        if initial_soc_pct > 1:
            initial_soc_pct /= 100.0
    except ValueError:
        log("SOC is not a valid number!")
        sys.exit(1)

    is_home = argv[2] == "t"

    try:
        eff_kwh_per_km = float(argv[3]) if (0 <= float(argv[3]) <= 1) else 0.128
    except ValueError:
        log("EFF_KWH_PER_KM is not a valid number!")
        sys.exit(1)

    try:
        charge_eff = float(argv[4]) if (0.7 <= float(argv[4]) <= 1) else 0.95
    except ValueError:
        log("CHARGE_EFF is not a valid number!")
        sys.exit(1)

    return RuntimeInputs(
        initial_soc_pct=initial_soc_pct,
        is_home=is_home,
        eff_kwh_per_km=eff_kwh_per_km,
        charge_eff=charge_eff,
    )


def load_env_inputs(log):
    env = {
        "soc_min_pct": float(os.getenv("SOC_MIN_PCT")),
        "soc_max_pct": float(os.getenv("SOC_MAX_PCT")),
        "lat": float(os.getenv("LAT")),
        "lon": float(os.getenv("LON")),
        "token_id": os.getenv("SOLAX_TOKEN_ID"),
        "wifi_sn": os.getenv("SOLAX_WIFI_SN"),
        "carnot_apikey": os.getenv("CARNOT_APIKEY"),
        "carnot_username": os.getenv("CARNOT_USERNAME"),
        "soft_soc_window_hours": float(os.getenv("SOFT_SOC_WINDOW_HOURS", "3")),
        "soft_soc_abs_max_pct": float(os.getenv("SOFT_SOC_ABS_MAX_PCT", "1.0")),
        "soft_soc_min_window_hours": float(os.getenv("SOFT_SOC_MIN_WINDOW_HOURS", "0")),
        "soft_soc_abs_min_pct": float(os.getenv("SOFT_SOC_ABS_MIN_PCT", "0.0")),
        "tm_db_name": os.getenv("TM_DB_NAME", "teslamate"),
        "tm_db_user": os.getenv("TM_DB_USER", "teslamate"),
        "tm_db_schema": os.getenv("TM_DB_SCHEMA", "history"),
        "tm_db_container": os.getenv("TM_DB_CONTAINER", ""),
    }

    trips_json = os.getenv("TRIPS", "[]")
    try:
        trips = pd.DataFrame(json.loads(trips_json))
    except Exception:
        log("⚠️ Failed to parse TRIPS from .env.local, using empty trip list")
        trips = pd.DataFrame([])

    return env, trips


def shift_active_trip_windows(trips: pd.DataFrame, is_home: bool, tz: str, log) -> pd.DataFrame:
    if trips.empty or not is_home:
        return trips

    now_slot = pd.Timestamp.now(tz=tz).floor("15min")
    now_minutes = now_slot.hour * 60 + now_slot.minute
    today_wday = now_slot.day_name().lower()

    trips = trips.copy()
    log("⚡ Car is home -> shifting trip times")
    for i, t in trips.iterrows():
        if str(t.get("day", "")).lower() != today_wday:
            continue

        try:
            start_h, start_m = map(int, str(t["away_start"]).split(":"))
            end_h, end_m = map(int, str(t["away_end"]).split(":"))
        except Exception:
            continue

        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        if start_minutes <= now_minutes < end_minutes:
            dist_to_start = now_minutes - start_minutes
            dist_to_end = end_minutes - now_minutes

            if dist_to_start <= dist_to_end:
                new_start_minutes = now_minutes + 15
                new_start = f"{new_start_minutes // 60:02d}:{new_start_minutes % 60:02d}"
                log(f"Trip on {t['day']} shifted: away_start {t['away_start']} -> {new_start}")
                trips.at[i, "away_start"] = new_start
            else:
                new_end_minutes = now_minutes
                new_end = f"{new_end_minutes // 60:02d}:{new_end_minutes % 60:02d}"
                log(f"Trip on {t['day']} shifted: away_end {t['away_end']} -> {new_end} (car home early)")
                trips.at[i, "away_end"] = new_end

    return trips
