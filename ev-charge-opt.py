from dotenv import load_dotenv
load_dotenv('.env.local')

import sys
import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import pulp
from datetime import datetime, timedelta
from pytz import timezone
from pandas.api.types import is_datetime64_any_dtype

# --- Logging ---
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(str(msg))

# --- Constants & Configuration ---
BATTERY_KWH = 75
CHARGER_KW = 11
CHARGER_MIN_A = 6
CHARGER_VOLT = 400
PHASES = 3
SOLAR_EFF = 0.97
PANEL_AREA = 11.5
PANEL_EFF = 0.2046
SYSTEMTARIF = 0.09250
NETTARIF_TSO = 0.07625
ELAFGIFT = 0.40000
LOOAD_TILLAEG = 0.08000
REFUSION = 0.5
TILT = 25
AZIMUTH = 0
TZ = "Europe/Copenhagen"

# --- Environment Variables ---
if len(sys.argv) < 2:
    log("No SOC provided!")
    sys.exit(1)

try:
    INITIAL_SOC_PCT = float(sys.argv[1])
    if INITIAL_SOC_PCT > 1:
        INITIAL_SOC_PCT /= 100.0
except ValueError:
    log("SOC is not a valid number!")
    sys.exit(1)

try:
    IS_HOME = sys.argv[2] == "t"
except ValueError:
    log("IS_HOME is not true!")
    sys.exit(1)

try:
    EFF_KWH_PER_KM = float(sys.argv[3]) if (0 <= float(sys.argv[3]) <= 1) else 0.128
except ValueError:
    log("EFF_KWH_PER_KM is not a valid number!")
    sys.exit(1)

try:
    CHARGE_EFF = float(sys.argv[4]) if (0.7 <= float(sys.argv[4]) <= 1) else 0.95
except ValueError:
    log("EFF_KWH_PER_KM is not a valid number!")
    sys.exit(1)

log(f"Latest SOC received from shell: {round(INITIAL_SOC_PCT*100, 2)}%")
log(f"Latest 10 drives average efficiency received from shell: {round(EFF_KWH_PER_KM, 3)} kWh/km")
log(f"Latest charging efficiency received from shell: {round(CHARGE_EFF*100, 2)}%")

SOC_MIN_PCT = float(os.getenv("SOC_MIN_PCT"))
SOC_MAX_PCT = float(os.getenv("SOC_MAX_PCT"))
LAT = float(os.getenv("LAT"))
LON = float(os.getenv("LON"))
trips_json = os.getenv("TRIPS")
trips = pd.DataFrame(json.loads(trips_json))
token_id = os.getenv("SOLAX_TOKEN_ID")
wifi_sn = os.getenv("SOLAX_WIFI_SN")
carnot_apikey = os.getenv("CARNOT_APIKEY")
carnot_username = os.getenv("CARNOT_USERNAME")

# Adjust
now_slot = pd.Timestamp.now(tz=TZ).floor("15min")
now_minutes = now_slot.hour * 60 + now_slot.minute
today_wday = now_slot.day_name().lower()

if IS_HOME:
    log(f"⚡ Car is home → shifting trip times")
    
    # Only affect trips for today
    for i, t in trips.iterrows():
        if t["day"].lower() != today_wday:
            continue
        
        # Parse away_start and away_end as minutes since midnight
        start_h, start_m = map(int, str(t["away_start"]).split(":"))
        end_h, end_m     = map(int, str(t["away_end"]).split(":"))
        start_minutes = start_h * 60 + start_m
        end_minutes   = end_h * 60 + end_m

        # If current slot is within this trip, shift the start
        if start_minutes <= now_minutes < end_minutes:
            new_start_minutes = now_minutes + 15
            new_start_h = new_start_minutes // 60
            new_start_m = new_start_minutes % 60
            new_start = f"{new_start_h:02d}:{new_start_m:02d}"
            log(f"Trip on {t['day']} shifted: away_start {t['away_start']} → {new_start}")
            trips.at[i, "away_start"] = new_start

        # --- NEW: If car is home early, shift away_end to now ---
        # If now is within the away period, and car is home, set away_end to now
        if start_minutes < now_minutes < end_minutes:
            new_end_minutes = now_minutes
            new_end_h = new_end_minutes // 60
            new_end_m = new_end_minutes % 60
            new_end = f"{new_end_h:02d}:{new_end_m:02d}"
            log(f"Trip on {t['day']} shifted: away_end {t['away_end']} → {new_end} (car home early)")
            trips.at[i, "away_end"] = new_end

# --- Utility Functions ---

def _fetch_open_meteo_with_retries(
    url: str,
    value_path: list,
    attempts: int = 3,
    sleep_sec: int = 2
) -> tuple:
    """
    Fetch Open-Meteo JSON and pull a numeric array at value_path (list of keys).
    Retries up to `attempts` times. Returns (times, values) or (None, None) on failure.
    """
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            j = r.json()
            node = j
            for k in value_path[:-1]:
                node = node[k]
            values = node[value_path[-1]]
            time_key = value_path[0]
            times = j[time_key]["time"]
            if values is None or len(values) == 0:
                raise ValueError("Empty values from Open-Meteo")
            log(f"✅ Open-Meteo success for {time_key} on attempt {attempt}")
            return times, values
        except Exception as e:
            log(f"Open-Meteo fetch failed (attempt {attempt}/{attempts}): {e}")
            time.sleep(sleep_sec)
    log("⚠️ Open-Meteo total failure for URL:", url)
    return None, None

def _align_gti_to_quarters(
    times: list,
    values: list,
    tz: str,
    repeat_to_quarters: bool = False
) -> pd.Series:
    """
    Return quarter-resolution Series aligned to given times.
    If repeat_to_quarters=True (hourly input), repeat each hour 4×.
    """
    ts = pd.to_datetime(times).tz_localize(tz)
    arr = np.array(values, dtype=float)
    if repeat_to_quarters:
        ts_q = ts.repeat(4) + pd.to_timedelta(np.tile([0, 15, 30, 45], len(ts)), unit="m")
        arr_q = arr.repeat(4)
    else:
        ts_q = ts
        arr_q = arr
    return pd.Series(arr_q, index=ts_q)

def override_with_inverter(
    df: pd.DataFrame,
    tz: str,
    token_id: str,
    wifi_sn: str,
    attempts: int = 3,
    sleep_sec: int = 2
) -> pd.DataFrame:
    """
    Try to override current 15-min slot with inverter data (SolaxCloud).
    Retries up to `attempts` times. Returns the modified DataFrame.
    """
    url = "https://global.solaxcloud.com/api/v2/dataAccess/realtimeInfo/get"
    headers = {"tokenId": token_id, "Content-Type": "application/json"}
    payload = {"wifiSn": wifi_sn}
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "acpower" in data.get("result", {}):
                ac_power_w = float(data["result"]["acpower"])
                solar_kwh_now = ac_power_w / 1000.0 * 0.25
                now_slot = pd.Timestamp.now(tz=tz).floor("15min")
                mask = df["datetime_local"] == now_slot
                if mask.any():
                    old_val = df.loc[mask, "solar_energy"].values[0]
                    df.loc[mask, "solar_energy"] = solar_kwh_now
                    log(
                        f"✅ Overrode solar_energy at {now_slot}: "
                        f"{old_val:.3f} → {solar_kwh_now:.3f} kWh (from inverter)"
                    )
                else:
                    log(f"⚠️ Current slot {now_slot} not in df timeline")
                return df
            else:
                raise ValueError("Inverter API returned no data or missing acpower")
        except Exception as e:
            log(f"⚠️ Inverter fetch failed (attempt {attempt}/{attempts}): {e}")
            if attempt < attempts:
                time.sleep(sleep_sec)
    log("⚠️ Inverter override failed after all retries")
    return df

def fetch_dk1_prices_dkk(attempts: int = 3) -> pd.DataFrame:
    """
    Fetches DK1 spot prices from Nordpool (today and, if available, tomorrow).
    Returns DataFrame with columns: date, price, source.
    """
    today = datetime.now().date()
    now_cet = pd.Timestamp.now(tz=TZ)
    fetch_tomorrow = now_cet.hour > 12 or (now_cet.hour == 12 and now_cet.minute >= 45)
    log(
        f"{'🟢' if fetch_tomorrow else '🟡'} It is {now_cet.strftime('%H:%M %Z')} → "
        f"{'Nordpool tomorrow data should be available.' if fetch_tomorrow else 'Too early, skipping tomorrow fetch.'}"
    )
    try:
        from nordpool import elspot
        p = elspot.Prices(currency="DKK")
        dfs = []
        for offset in range(1 + int(fetch_tomorrow)):
            date_str = (today + timedelta(days=offset)).strftime("%Y-%m-%d")
            rows = None
            for attempt in range(attempts):
                try:
                    data = p.hourly(end_date=date_str, areas=["DK1"])
                    values = data["areas"]["DK1"]["values"]
                    rows = [
                        {
                            "date": pd.to_datetime(v["start"], utc=True),
                            "price": (v["value"] / 10.0) * 1.25,
                            "source": "Nordpool"
                        }
                        for v in values if v["value"] is not None
                    ]
                    log(f"✅ Nordpool success for {date_str} on attempt {attempt+1}")
                    break
                except Exception as e:
                    log(f"Nordpool fetch failed {date_str} (attempt {attempt+1}/{attempts}): {e}")
                    time.sleep(2)
            if rows is not None:
                dfs.append(pd.DataFrame(rows))
            else:
                log(f"⚠️ Nordpool prices not yet available for {date_str}, skipping")
        if not dfs:
            raise RuntimeError("No Nordpool data available at all")
        df = pd.concat(dfs, ignore_index=True)
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"Nordpool failed after retries: {e}")

def fetch_github_forecast_dkk(
    github_url: str = "https://raw.githubusercontent.com/solmoller/Spotprisprognose/refs/heads/main/DK1.json",
    attempts: int = 3,
    sleep_sec: int = 2
) -> pd.DataFrame:
    """
    Fetches DK1 spot price forecast from GitHub.
    Returns DataFrame with columns: date, price, source.
    """
    df_github = pd.DataFrame(columns=["date", "price", "source"])
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(github_url, timeout=30)
            r.raise_for_status()
            j = r.json()
            df_github_temp = pd.DataFrame(list(j.items()), columns=["date", "price"])
            df_github_temp["date"] = pd.to_datetime(df_github_temp["date"], utc=True)
            df_github_temp["source"] = "GitHub"
            df_github = df_github_temp[["date", "price", "source"]]
            log(f"✅ GitHub forecast success ({len(df_github)} hours) on attempt {attempt}")
            break
        except Exception as e:
            log(f"⚠️ GitHub forecast fetch failed (attempt {attempt}/{attempts}): {e}")
            if attempt < attempts:
                time.sleep(sleep_sec)
            else:
                log("❌ Github: All attempts failed. Returning empty DataFrame.")
    return df_github

def fetch_carnot_forecast_dkk(
    carnot_url: str = "https://openapi.carnot.dk/openapi/get_predict",
    apikey: str = "YOUR_API_KEY",
    username: str = "YOUR_USERNAME",
    daysahead: int = 3,
    attempts: int = 3,
    sleep_sec: int = 2
) -> pd.DataFrame:
    """
    Fetches DK1 spot price forecast from Carnot API.
    Returns DataFrame with columns: date, price, source.
    """
    df_carnot = pd.DataFrame(columns=["date", "price", "source"])
    for attempt in range(1, attempts + 1):
        try:
            headers = {
                "accept": "application/json",
                "apikey": apikey,
                "username": username,
            }
            params = {"daysahead": daysahead, "energysource": "spotprice", "region": "dk1"}
            r = requests.get(carnot_url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            df_carnot_temp = pd.DataFrame(data["predictions"])
            df_carnot_temp["date"] = pd.to_datetime(df_carnot_temp["utctime"], utc=True)
            df_carnot_temp["price"] = df_carnot_temp["prediction"] / 10.0 * 1.25
            df_carnot_temp["source"] = "Carnot"
            df_carnot = df_carnot_temp[["date", "price", "source"]]
            log(f"✅ Carnot forecast success ({len(df_carnot)} hours) on attempt {attempt}")
            break
        except Exception as e:
            log(f"⚠️ Carnot forecast fetch failed (attempt {attempt}/{attempts}): {e}")
            if attempt < attempts:
                time.sleep(sleep_sec)
            else:
                log("❌ Carnot: All attempts failed. Returning empty DataFrame.")
    return df_carnot

def fetch_combined_forecast(
    source: str = "combined",
    apikey: str = "YOUR_API_KEY",
    username: str = "YOUR_USERNAME"
) -> pd.DataFrame:
    """
    Fetch price forecast(s) depending on selected source.
    - "github": only Github spot price forecast
    - "carnot": only Carnot forecast
    - "combined": Github prioritized, Carnot appended (default)
    """
    if source == "github":
        forecast = fetch_github_forecast_dkk()
        log("🔮 Using Github forecast only")
    elif source == "carnot":
        forecast = fetch_carnot_forecast_dkk(apikey=apikey, username=username, daysahead=7, attempts=3)
        log("🔮 Using Carnot forecast only")
    elif source == "combined":
        github = fetch_github_forecast_dkk()
        carnot = fetch_carnot_forecast_dkk(apikey=apikey, username=username, daysahead=7, attempts=3)
        last_github = github["date"].max()
        future_carnot = carnot[carnot["date"] > last_github]
        forecast = pd.concat([github, future_carnot], ignore_index=True).sort_values("date").reset_index(drop=True)
        log("🔮 Using Github forecast (prioritized), Carnot appended")
    else:
        raise ValueError(f"Invalid forecast source: {source}")
    return forecast.reset_index(drop=True)

def combine_actuals_and_forecast(
    prices_actual: pd.DataFrame,
    prices_forecast: pd.DataFrame,
    tz: str = TZ
) -> pd.DataFrame:
    """
    Combines actual and forecasted prices, filtering from current hour and forward.
    """
    last_actual = prices_actual["date"].max()
    future = prices_forecast[prices_forecast["date"] > last_actual]
    df = pd.concat([prices_actual, future], ignore_index=True).sort_values("date").reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC").floor("15min") - timedelta(hours=2)
    df = df[df["date"] >= now]
    return df.reset_index(drop=True)

# --- Data Fetching ---
prices_actual = fetch_dk1_prices_dkk()
prices_forecast = fetch_combined_forecast(source="combined", apikey=carnot_apikey, username=carnot_username)
prices = combine_actuals_and_forecast(prices_actual=prices_actual, prices_forecast=prices_forecast, tz=TZ)
prices = prices.sort_values("date").reset_index(drop=True)

# --- Main Optimization Function ---

def optimize_ev_charging(
    trips: pd.DataFrame,
    prices: pd.DataFrame,
    battery_kwh: float = BATTERY_KWH,
    soc_min_pct: float = SOC_MIN_PCT,
    soc_max_pct: float = SOC_MAX_PCT,
    charger_kw: float = CHARGER_KW,
    charger_min_a: float = CHARGER_MIN_A,
    charger_volt: float = CHARGER_VOLT,
    phases: int = PHASES,
    eff_kwh_per_km: float = EFF_KWH_PER_KM,
    initial_soc_pct: float = INITIAL_SOC_PCT,
    solar_eff: float = SOLAR_EFF,
    panel_area: float = PANEL_AREA,
    panel_eff: float = PANEL_EFF,
    systemtarif: float = SYSTEMTARIF,
    nettarif_tso: float = NETTARIF_TSO,
    elafgift: float = ELAFGIFT,
    looad_tillaeg: float = LOOAD_TILLAEG,
    lat: float = LAT,
    lon: float = LON,
    tilt: float = TILT,
    azimuth: float = AZIMUTH,
    tz: str = TZ,
    charge_eff: float = CHARGE_EFF,
    refusion: float = REFUSION
) -> pd.DataFrame:
    """
    Optimize EV charging schedule given trips, prices, and system parameters.
    Returns DataFrame with optimal schedule and cost breakdown.
    """
    # --- Prices preprocessing ---
    assert {"date", "price"}.issubset(prices.columns)
    if not is_datetime64_any_dtype(prices["date"]):
        prices["date"] = pd.to_datetime(prices["date"], utc=True)
    prices = prices.sort_values("date").reset_index(drop=True)

    # --- Battery & charger parameters ---
    CHARGER_MIN_KW = (charger_min_a * charger_volt * math.sqrt(phases)) / 1000.0
    SOC_MIN = battery_kwh * soc_min_pct
    SOC_MAX = battery_kwh * soc_max_pct
    SOC0    = battery_kwh * initial_soc_pct
    FLAT_ADDERS = systemtarif + nettarif_tso + elafgift + looad_tillaeg

    # --- Build timeline & prices ---
    df = pd.DataFrame({"datetime_utc": prices["date"]})
    df["datetime_local"] = df["datetime_utc"].dt.tz_convert(tz)
    df["wday_label"] = df["datetime_local"].dt.day_name().str.lower()
    df["hour_local"] = df["datetime_local"].dt.hour
    df["minute_local"] = df["datetime_local"].dt.minute
    df["spot_kr_kwh"] = prices["price"] / 100.0

    h = df["hour_local"].values
    dso = np.full(len(df), 0.12763)
    dso[(h >= 0) & (h < 6)] = 0.08512
    dso[(h >= 6) & (h < 17)] = 0.12763
    dso[(h >= 17) & (h < 21)] = 0.33200
    df["dso_tariff"] = dso
    df["total_price_kr_kwh"] = df["spot_kr_kwh"] + FLAT_ADDERS + df["dso_tariff"]

    # --- Expand to 15-min resolution ---
    N = len(df)
    df_q = df.loc[df.index.repeat(4)].copy().reset_index(drop=True)
    df_q["datetime_local"] = df_q["datetime_local"] + pd.to_timedelta(np.tile([0,15,30,45], N), unit="m")
    df_q["hour_local"] = df_q["datetime_local"].dt.hour
    df_q["minute_local"] = df_q["datetime_local"].dt.minute
    df_q["wday_label"] = df_q["datetime_local"].dt.day_name().str.lower()

    df = df_q

    # Alternativ now = pd.Timestamp.now(tz=tz)
    now = pd.Timestamp.now(tz=tz).floor("15min")

    # filter from current hour and forward
    df = df.loc[df["datetime_local"] >= now].copy()
    df.reset_index(drop=True, inplace=True)
    H = len(df)
    assert df.index.min() == 0, "Index not starting at 0 after reset!"

    # --- Parse trip times (accept HH:MM) ---
    trips = trips.copy()
    for col in ["away_start", "away_end"]:
        if trips[col].dtype == object:
            trips[col] = pd.to_datetime(trips[col], format="%H:%M").dt.time

    # --- Availability ---
    available = np.ones(H, dtype=int)
    for _, t in trips.iterrows():
        idx_day = np.where(df["wday_label"].values == t["day"].lower())[0]
        start_minutes = t["away_start"].hour * 60 + t["away_start"].minute
        end_minutes   = t["away_end"].hour * 60 + t["away_end"].minute
        minutes_of_day = df["hour_local"].values[idx_day] * 60 + df["minute_local"].values[idx_day]
        mask = (minutes_of_day >= start_minutes) & (minutes_of_day < end_minutes)
        available[idx_day[mask]] = 0
    df["available"] = available

    # --- Solar irradiance (Open-Meteo) with retries & fallback ---
    start_date = df["datetime_local"].min().strftime("%Y-%m-%d")
    end_date   = df["datetime_local"].max().strftime("%Y-%m-%d")

    base = "https://api.open-meteo.com/v1/forecast"
    common = (
        f"?latitude={lat}&longitude={lon}"
        f"&tilt={tilt}&azimuth={azimuth}"
        f"&start_date={start_date}&end_date={end_date}"
        "&timezone=Europe/Copenhagen"
    )

    # Try 15-minute first
    url_15 = f"{base}{common}&minutely_15=global_tilted_irradiance_instant"
    t15, v15 = _fetch_open_meteo_with_retries(
        url_15,
        ["minutely_15", "global_tilted_irradiance_instant"],
        attempts=3, sleep_sec=2
    )

    # Then try hourly (as fallback and/or for filling gaps)
    url_h = f"{base}{common}&hourly=global_tilted_irradiance_instant"
    th, vh = _fetch_open_meteo_with_retries(
        url_h,
        ["hourly", "global_tilted_irradiance_instant"],
        attempts=3, sleep_sec=2
    )

    if t15 is None and th is None:
        raise RuntimeError("Open-Meteo irradiance unavailable after retries (both 15-min and hourly).")

    # Build quarter-resolution series for whichever we have
    ser_15 = _align_gti_to_quarters(t15, v15, tz, repeat_to_quarters=False) if t15 is not None else None
    ser_h  = _align_gti_to_quarters(th,  vh,  tz, repeat_to_quarters=True)  if th  is not None else None

    # Prefer 15-min; fill gaps with hourly if available; else just hourly.
    if ser_15 is not None and ser_h is not None:
        ser_q = ser_15.combine_first(ser_h)
        source_used = "minutely_15 + hourly fallback"
    elif ser_15 is not None:
        ser_q = ser_15
        source_used = "minutely_15"
    else:
        ser_q = ser_h
        source_used = "hourly (upsampled to 15-min)"

    # Align to model timeline (15-min local)
    irr_q_vals = ser_q.reindex(df["datetime_local"]).values
    if np.isnan(irr_q_vals).all():
        raise RuntimeError("Irradiance alignment error: all NaN after reindexing to timeline.")

    # Convert W/m² → kWh per 15-min slot: (W/m² / 1000) * area * panel_eff * solar_eff * 0.25h
    solar_energy_q = (irr_q_vals / 1000.0) * panel_area * panel_eff * solar_eff * 0.25

    df["irradiance"]   = irr_q_vals
    df["solar_energy"] = solar_energy_q

    log(f"✅ Using Open-Meteo irradiance source: {source_used}")

    # Override current slot with inverter data if possible
    df = override_with_inverter(df, tz, token_id, wifi_sn)

    # --- Trip energy vector ---
    trip_energy_vec = np.zeros(H)
    for _, t in trips.iterrows():
        need_kwh = float(t["trip_kwh"]) if pd.notna(t["trip_kwh"]) else float(t["distance_km"]) * eff_kwh_per_km
        dep_minutes = t["away_start"].hour * 60 + t["away_start"].minute
        idx_dep = df.index[
            (df["wday_label"].values == t["day"].lower()) &
            ((df["hour_local"].values * 60 + df["minute_local"].values) == dep_minutes)
        ]
        log(f"trip {t['day']} {t['away_start']} -> matched pos(s): {idx_dep.tolist()}")
        log(f"datetime at matched pos(s): {df.loc[idx_dep, 'datetime_local'].tolist()}")
        if len(idx_dep) >= 1:
            trip_energy_vec[idx_dep[0]] += need_kwh
        if SOC_MIN + need_kwh > SOC_MAX:
            raise RuntimeError(f"Trip on {t['day']} {t['away_start']} infeasible (need {need_kwh:.1f} kWh + reserve)")

    # --- Build MILP ---
    cap_per_quarter = charger_kw * 0.25
    min_per_quarter = CHARGER_MIN_KW * 0.25
    prob = pulp.LpProblem("ev_charging_opt", pulp.LpMinimize)
    grid  = pulp.LpVariable.dicts("grid",  range(H), lowBound=0, cat=pulp.LpContinuous)
    solar = pulp.LpVariable.dicts("solar", range(H), lowBound=0, cat=pulp.LpContinuous)
    first_trip_idx = np.where(trip_energy_vec > 0)[0]
    soc = {}
    for h in range(H):
        if h == 0:
            soc[h] = pulp.LpVariable(f"soc_{h}", lowBound=SOC0, upBound=SOC_MAX, cat=pulp.LpContinuous)
        elif h < first_trip_idx[0]:
            soc[h] = pulp.LpVariable(f"soc_{h}", lowBound=SOC0, upBound=SOC_MAX, cat=pulp.LpContinuous)
        else:
            soc[h] = pulp.LpVariable(f"soc_{h}", lowBound=SOC_MIN, upBound=SOC_MAX, cat=pulp.LpContinuous)
    z     = pulp.LpVariable.dicts("z",     range(H), cat=pulp.LpBinary)
    prices_k = df["total_price_kr_kwh"].values
    prob += pulp.lpSum(grid[h] * float(prices_k[h]) - refusion * solar[h] for h in range(H))

    # SOC dynamics with charging efficiency
    for h in range(H):
        if h == 0:
            # soc0 + charge_eff*(grid+solar) - trip = soc[0]
            prob += soc[h] - charge_eff * (grid[h] + solar[h]) == (SOC0 - float(trip_energy_vec[h]))
        else:
            # soc[h-1] + charge_eff*(grid+solar) - trip = soc[h]
            prob += soc[h] - soc[h-1] - charge_eff * (grid[h] + solar[h]) == (-float(trip_energy_vec[h]))

    avail = df["available"].values.astype(float)
    solar_cap = df["solar_energy"].values
    for h in range(H):
        prob += grid[h] + solar[h] <= (cap_per_quarter * avail[h]) * z[h]
        prob += solar[h] <= solar_cap[h] * z[h]
        prob += grid[h] + solar[h] >= min_per_quarter * z[h]
    trip_rows = np.where(trip_energy_vec > 0)[0]
    for h in trip_rows:
        if h > 0:
            prob += soc[h-1] >= SOC_MIN + float(trip_energy_vec[h])

    solver = pulp.PULP_CBC_CMD(msg=False)
    res_status = prob.solve(solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"MILP not optimal. Status: {pulp.LpStatus[prob.status]}")

    # --- Results ---
    grid_opt  = np.array([pulp.value(grid[h])  for h in range(H)])
    solar_opt = np.array([pulp.value(solar[h]) for h in range(H)])
    soc_opt   = np.array([pulp.value(soc[h])   for h in range(H)])

    # Derived “stored in battery” (post-losses)
    grid_to_batt  = grid_opt  * charge_eff
    solar_to_batt = solar_opt * charge_eff

    df_out = pd.DataFrame({
        "datetime_local": df["datetime_local"],
        "weekday": df["wday_label"].values,
        "hour": df["hour_local"].values,
        "minute": df["minute_local"].values,
        "price_kr_per_kwh": np.round(df["total_price_kr_kwh"].values, 5),
        "available": df["available"].values,
        "trip_kwh_at_departure": np.round(trip_energy_vec, 3),

        # DRAWN (same column names as before for backward compatibility)
        "grid_charge_kwh":  np.round(grid_opt, 4),
        "solar_charge_kwh": np.round(solar_opt, 4),
        "total_charge_kwh": np.round(grid_opt + solar_opt, 4),

        # STORED in battery (new columns)
        "grid_to_batt_kwh":  np.round(grid_to_batt, 4),
        "solar_to_batt_kwh": np.round(solar_to_batt, 4),
        "total_to_batt_kwh": np.round(grid_to_batt + solar_to_batt, 4),

        # Current draw → amps (based on drawn power)
        "amp": np.round((((grid_opt + solar_opt) / 0.25) * 1000) / (math.sqrt(phases) * charger_volt), 0),

        "irradiance": df["irradiance"].values,
        "soc_kwh": np.round(soc_opt, 3),

        # Cost (only grid is paid here)
        "cost_kr": np.round(grid_opt * df["total_price_kr_kwh"].values, 4),
    })

    # Add charge_eff so it’s visible downstream if needed
    df_out["charge_eff"] = charge_eff

    return df_out

df_out = optimize_ev_charging(
    trips,
    prices,
    BATTERY_KWH, SOC_MIN_PCT, SOC_MAX_PCT,
    CHARGER_KW, CHARGER_MIN_A, CHARGER_VOLT, PHASES,
    EFF_KWH_PER_KM, INITIAL_SOC_PCT,
    SOLAR_EFF, PANEL_AREA, PANEL_EFF,
    SYSTEMTARIF, NETTARIF_TSO, ELAFGIFT, LOOAD_TILLAEG,
    LAT, LON, TILT, AZIMUTH,
    TZ,
    CHARGE_EFF,
    REFUSION
)


# SoC before/after around charging/trip events (use stored energy, not drawn)
stored_this_slot = df_out["total_to_batt_kwh"].values
trip_this_slot   = df_out["trip_kwh_at_departure"].values

soc_kwh_before = np.where(
    (df_out["grid_charge_kwh"].values + df_out["solar_charge_kwh"].values) > 0,
    df_out["soc_kwh"].values - stored_this_slot,   # remove what was stored to get 'before'
    df_out["soc_kwh"].values + trip_this_slot,     # add back the trip to get 'before'
)

df_out["soc_kwh_before"] = soc_kwh_before
df_out["soc_pct_before"] = np.round((df_out["soc_kwh_before"].values / BATTERY_KWH) * 100.0, 1)
df_out["soc_pct_after"]  = np.round((df_out["soc_kwh"].values / BATTERY_KWH) * 100.0, 1)

# naive effective price per kWh drawn (cost / drawn energy)
# (NaN if nothing drawn that slot)
df_out["effective_price_kr_per_kwh_drawn"] = (
    df_out["cost_kr"] / df_out["total_charge_kwh"].replace(0, np.nan)
)

# If you want to apply REFUSION credit for solar used (value you gave as REFUSION [kr/kWh]),
# incorporate it as a negative cost for solar_kWh used (visual only; the solver was not
# informed of this). REFUSION variable in your script is in kr/kWh? If REFUSION is kr/kWh:
df_out["effective_net_cost_kr"] = df_out["cost_kr"] - (df_out["solar_charge_kwh"] * REFUSION)
df_out["effective_price_kr_per_kwh_drawn_with_refusion"] = (
    df_out["effective_net_cost_kr"] / df_out["total_charge_kwh"].replace(0, np.nan)
)

mask_events = (
    (df_out["trip_kwh_at_departure"].values > 0) |
    (df_out["grid_charge_kwh"].values > 0) |
    (df_out["solar_charge_kwh"].values > 0)
)

log("\n=== Optimal Charging & Trip Events (15-min) ===")
header = (
    f"{'datetime_local':<16} | {'weekday':<9} | {'hour':<2} | {'minute':<2} | {'irradiance':<10} | "
    f"{'price_kr/kWh':>12} | {'eff_price_kr/kWh':>12} | {'eff_price_kr_ref/kWh':>12} | {'grid_kWh':>8} | {'solar_kWh':>9} | {'total_kwh':>9} | "
    f"{'amp':>3} | {'trip_kWh':>8} | {'soc_kwh':>7} | {'soc_%_before':>12} | {'soc_%_after':>11}"
)
log(header)
log("-" * len(header))

for _, row in df_out.loc[mask_events].iterrows():
    log(
        f"{row['datetime_local']:%Y-%m-%d %H:%M} | "
        f"{row['weekday']:<9} | "
        f"{int(row['hour']):<4d} | "
        f"{int(row['minute']):<6d} | "
        f"{row['irradiance']:>10.0f} | "
        f"{row['price_kr_per_kwh']:>12.2f} | "
        f"{row['effective_price_kr_per_kwh_drawn']:>16.2f} | "
        f"{row['effective_price_kr_per_kwh_drawn_with_refusion']:>20.2f} | "
        f"{row['grid_charge_kwh']:>8.2f} | "
        f"{row['solar_charge_kwh']:>9.2f} | "
        f"{row['total_charge_kwh']:>9.2f} | "
        f"{int(row['amp']):>3d} | "
        f"{row['trip_kwh_at_departure']:>8.2f} | "
        f"{row['soc_kwh']:>7.2f} | "
        f"{row['soc_pct_before']:>12.1f} | "
        f"{row['soc_pct_after']:>11.1f}"
    )

# Totals (drawn vs stored)
total_cost = float(df_out["cost_kr"].sum())

from_grid_drawn  = float(df_out["grid_charge_kwh"].sum())
from_solar_drawn = float(df_out["solar_charge_kwh"].sum())
total_drawn      = float(df_out["total_charge_kwh"].sum())

from_grid_stored  = float(df_out["grid_to_batt_kwh"].sum())
from_solar_stored = float(df_out["solar_to_batt_kwh"].sum())
total_stored      = float(df_out["total_to_batt_kwh"].sum())

# Effective cost (your logic kept: refusion applied to solar 'drawn')
effective_cost = total_cost - from_solar_drawn * REFUSION

# Averages per kWh drawn and per kWh stored (both useful)
avg_cost_per_kWh_drawn  = (total_cost / total_drawn) if total_drawn > 0 else float("nan")
avg_cost_per_kWh_stored = (total_cost / total_stored) if total_stored > 0 else float("nan")

avg_eff_per_kWh_drawn  = (effective_cost / total_drawn) if total_drawn > 0 else float("nan")
avg_eff_per_kWh_stored = (effective_cost / total_stored) if total_stored > 0 else float("nan")


df_out["date"] = df_out["datetime_local"].dt.date
df_out["weekday"] = df_out["datetime_local"].dt.day_name()

daily_summary = df_out.groupby(["date", "weekday"]).agg(
    grid_drawn_kWh=("grid_charge_kwh", "sum"),
    solar_drawn_kWh=("solar_charge_kwh", "sum"),
    total_drawn_kWh=("total_charge_kwh", "sum"),
    grid_stored_kWh=("grid_to_batt_kwh", "sum"),
    solar_stored_kWh=("solar_to_batt_kwh", "sum"),
    total_stored_kWh=("total_to_batt_kwh", "sum"),
    trip_kWh=("trip_kwh_at_departure", "sum"),
    cost=("cost_kr", "sum"),
).reset_index()

# SoC start/end
soc_start = (
    df_out.sort_values("datetime_local")
    .groupby("date")["soc_pct_before"]
    .first()
)
soc_end = (
    df_out.sort_values("datetime_local")
    .groupby("date")["soc_pct_after"]
    .last()
)

daily_summary["soc_start"] = daily_summary["date"].map(soc_start)
daily_summary["soc_end"] = daily_summary["date"].map(soc_end)

# Effective cost (refusion on solar drawn)
daily_summary["effective_cost"] = daily_summary["cost"] - daily_summary["solar_drawn_kWh"] * REFUSION

# Average costs per kWh (drawn vs stored)
daily_summary["cost_per_kWh_drawn"]  = daily_summary["cost"] / daily_summary["total_drawn_kWh"].replace(0, np.nan)
daily_summary["cost_per_kWh_stored"] = daily_summary["cost"] / daily_summary["total_stored_kWh"].replace(0, np.nan)

daily_summary["eff_cost_per_kWh_drawn"]  = daily_summary["effective_cost"] / daily_summary["total_drawn_kWh"].replace(0, np.nan)
daily_summary["eff_cost_per_kWh_stored"] = daily_summary["effective_cost"] / daily_summary["total_stored_kWh"].replace(0, np.nan)


# log
log("\n=== Daily Summary ===")
header = (
    f"{'date':<10} | {'weekday':<9} | {'grid_kWh':>8} | {'solar_kWh':>8} | "
    f"{'total_kWh':>8} | {'trip_kWh':>8} | {'soc_start%':>9} | {'soc_end%':>7} | "
    f"{'cost':>8} | {'eff_cost':>10} | {'avg_cost':>9} | {'avg_eff':>9}"
)
log(header)
log("-" * len(header))

for _, row in daily_summary.iterrows():
    log(
        f"{row['date']} | "
        f"{row['weekday']:<9} | "
        f"{row['grid_drawn_kWh']:8.2f} | "
        f"{row['solar_drawn_kWh']:9.2f} | "
        f"{row['total_drawn_kWh']:9.2f} | "
        f"{row['trip_kWh']:8.2f} | "
        f"{row['soc_start']:10.1f} | "
        f"{row['soc_end']:8.1f} | "
        f"{row['cost']:8.2f} | "
        f"{row['effective_cost']:10.2f} | "
        f"{row['cost_per_kWh_drawn']:9.2f} | "
        f"{row['eff_cost_per_kWh_drawn']:9.2f}"
    )

log(
    f"Total cost: {total_cost:.2f} kr. "
    f"Total effective cost: {effective_cost:.2f} kr. "
    f"Energy drawn: {total_drawn:.2f} kWh ({from_grid_drawn:.2f} grid, {from_solar_drawn:.2f} solar). "
    f"Energy stored: {total_stored:.2f} kWh ({from_grid_stored:.2f} grid, {from_solar_stored:.2f} solar). "
    f"Avg cost: {avg_cost_per_kWh_drawn:.2f} kr/kWh drawn, {avg_cost_per_kWh_stored:.2f} kr/kWh stored. "
    f"Eff. avg: {avg_eff_per_kWh_drawn:.2f} (drawn), {avg_eff_per_kWh_stored:.2f} (stored)."
)

# --- Notification Logic ---
# Use ~/repos/ev-charge-opt/tmp
STATE_FILE = os.path.expanduser("~/repos/ev-charge-opt/tmp/ev_charging_state.json")

def load_last_amp():
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            return state.get("last_amp", 0)
    except FileNotFoundError:
        return 0

def save_last_amp(amp):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)  # ensure tmp folder exists
    with open(STATE_FILE, "w") as f:
        json.dump({"last_amp": amp}, f)

def should_notify(current_amp: int, last_amp: int):
    """
    Notification rules:
    - last_amp == 0 and current_amp > 0 → charging started
    - last_amp > 0 and current_amp == 0 → charging stopped
    - last_amp > 0 and current_amp > 0 and current_amp != last_amp → amps changed
    """
    if last_amp == 0 and current_amp > 0:
        return True, f"🔌 Charging started at {current_amp}A"

    if last_amp > 0 and current_amp == 0:
        return True, "⏹️ Charging stopped"

    if last_amp > 0 and current_amp > 0 and current_amp != last_amp:
        return True, f"⚡ Amps changed from {last_amp}A to {current_amp}A"

    return False, "No change"

from datetime import time

QUIET_START = time(0, 0)    # 00:00
QUIET_END   = time(6, 45)   # 06:45

def in_quiet_hours(now: pd.Timestamp) -> bool:
    return QUIET_START <= now.time() <= QUIET_END

now_slot = pd.Timestamp.now(tz=TZ).floor("15min")
current_row = df_out.iloc[0]
current_amp = int(current_row["amp"])

last_amp = load_last_amp()
notify, reason = should_notify(current_amp, last_amp)

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Email Notification ---
def send_email_notification(subject: str, body: str, sender: str, recipient: str, smtp_server: str, smtp_port: int, username: str, password: str):
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # secure connection
            server.login(username, password)
            server.sendmail(sender, recipient, msg.as_string())
        log("📧 Email notification sent.")
    except Exception as e:
        log(f"⚠️ Failed to send email: {e}")

def format_charge_plan_simple(df, mask_events, max_rows=24):
    """
    Returns a string summary of the charge plan (first max_rows rows with events).
    Columns: weekday, hour, minute, amps, effective_price_kr_per_kwh_drawn_with_refusion, soc_pct_before, soc_pct_after
    """
    cols = [
        "weekday", "hour", "minute", "amp",
        "effective_price_kr_per_kwh_drawn_with_refusion",
        "soc_pct_before", "soc_pct_after"
    ]
    df_short = df.loc[mask_events, cols].copy().head(max_rows)
    lines = []
    header = (
        f"{'weekday':<9} | {'hour':>2} | {'minute':>2} | {'amp':>4} | "
        f"{'eff_price_kr/kWh':>20} | {'SoC% before':>11} | {'SoC% after':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df_short.iterrows():
        lines.append(
            f"{row['weekday']:<9} | "
            f"{int(row['hour']):2d} | "
            f"{int(row['minute']):2d} | "
            f"{int(row['amp']):4d} | "
            f"{row['effective_price_kr_per_kwh_drawn_with_refusion']:20.2f} | "
            f"{row['soc_pct_before']:11.1f} | "
            f"{row['soc_pct_after']:10.1f}"
        )
    return "\n".join(lines)

if notify:
    save_last_amp(current_amp)

    if in_quiet_hours(now_slot):
        log(f"🌙 Quiet hours ({now_slot.time()}), no email sent ({reason})")
    else:
        subject = f"EV Charging Alert: {current_amp}A at {now_slot.strftime('%H:%M')}"
        body = (
            f"{reason} at {now_slot}.\n\n"
            f"Amps: {current_amp}\n"
            f"Grid: {current_row['grid_charge_kwh']:.2f} kWh\n"
            f"Solar: {current_row['solar_charge_kwh']:.2f} kWh\n"
            f"SoC before: {current_row['soc_pct_before']:.1f}%\n"
            f"SoC after:  {current_row['soc_pct_after']:.1f}%\n"
        )

        # Add charge plan table (simple version)
        body += "\n\n=== Optimal Charging & Trip Events (15-min) ===\n"
        body += format_charge_plan_simple(df_out, mask_events, max_rows=24)

        send_email_notification(
            subject=subject,
            body=body,
            sender=os.getenv("EMAIL_SENDER"),
            recipient=os.getenv("EMAIL_RECIPIENT"),
            smtp_server=os.getenv("SMTP_SERVER"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            username=os.getenv("SMTP_USER"),
            password=os.getenv("SMTP_PASS"),
        )
else:
    log(f"ℹ️ No email sent ({reason})")
