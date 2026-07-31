import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pulp
import requests
from pandas.api.types import is_datetime64_any_dtype

from ev_charge_opt.history import (
    HistoryStore,
    fetch_solax_current_quarter_kwh,
    prepare_total_prices_15m,
)


def _fetch_open_meteo_with_retries(url: str, value_path: list, log, attempts: int = 5, sleep_sec: int = 2):
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

    log(f"⚠️ Open-Meteo total failure for URL: {url}")
    return None, None


def _align_to_quarters(times: list, values: list, tz: str, repeat_to_quarters: bool = False) -> pd.Series:
    ts = pd.to_datetime(times, errors="coerce").tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    arr = np.array(values, dtype=float)

    valid_mask = ~ts.isna()
    ts = ts[valid_mask]
    arr = arr[valid_mask]

    if repeat_to_quarters:
        ts_q = ts.repeat(4) + pd.to_timedelta(np.tile([0, 15, 30, 45], len(ts)), unit="m")
        arr_q = arr.repeat(4)
    else:
        ts_q = ts
        arr_q = arr

    return pd.Series(arr_q, index=ts_q)


def _build_solar_series(df: pd.DataFrame, cfg, log):
    start_date = df["datetime_local"].min().strftime("%Y-%m-%d")
    end_date = df["datetime_local"].max().strftime("%Y-%m-%d")

    base = "https://api.open-meteo.com/v1/forecast"
    common = (
        f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
        f"&tilt={cfg['tilt']}&azimuth={cfg['azimuth']}"
        f"&start_date={start_date}&end_date={end_date}"
        "&timezone=Europe/Copenhagen"
    )

    t15, v15 = _fetch_open_meteo_with_retries(
        f"{base}{common}&minutely_15=global_tilted_irradiance_instant",
        ["minutely_15", "global_tilted_irradiance_instant"],
        log,
    )
    th, vh = _fetch_open_meteo_with_retries(
        f"{base}{common}&hourly=global_tilted_irradiance_instant",
        ["hourly", "global_tilted_irradiance_instant"],
        log,
    )

    if t15 is None and th is None:
        raise RuntimeError("Open-Meteo irradiance unavailable after retries (both 15-min and hourly).")

    ser_15 = _align_to_quarters(t15, v15, cfg["tz"], repeat_to_quarters=False) if t15 is not None else None
    ser_h = _align_to_quarters(th, vh, cfg["tz"], repeat_to_quarters=True) if th is not None else None

    if ser_15 is not None and ser_h is not None:
        ser_q = ser_15.combine_first(ser_h)
        source_used = "minutely_15 + hourly fallback"
    elif ser_15 is not None:
        ser_q = ser_15
        source_used = "minutely_15"
    else:
        ser_q = ser_h
        source_used = "hourly (upsampled to 15-min)"

    irr_q_vals = ser_q.reindex(df["datetime_local"]).values
    if np.isnan(irr_q_vals).all():
        raise RuntimeError("Irradiance alignment error: all NaN after reindexing to timeline.")

    ttemp15, vtemp15 = _fetch_open_meteo_with_retries(
        f"{base}{common}&minutely_15=temperature_2m",
        ["minutely_15", "temperature_2m"],
        log,
    )
    ttemph, vtemph = _fetch_open_meteo_with_retries(
        f"{base}{common}&hourly=temperature_2m",
        ["hourly", "temperature_2m"],
        log,
    )

    ser_temp_15 = _align_to_quarters(ttemp15, vtemp15, cfg["tz"], repeat_to_quarters=False) if ttemp15 is not None else None
    ser_temp_h = _align_to_quarters(ttemph, vtemph, cfg["tz"], repeat_to_quarters=True) if ttemph is not None else None

    if ser_temp_15 is not None and ser_temp_h is not None:
        ser_temp_q = ser_temp_15.combine_first(ser_temp_h)
    elif ser_temp_15 is not None:
        ser_temp_q = ser_temp_15
    elif ser_temp_h is not None:
        ser_temp_q = ser_temp_h
    else:
        ser_temp_q = None

    if ser_temp_q is None:
        temp_q_vals = np.full_like(irr_q_vals, cfg["panel_temp_ref_c"], dtype=float)
    else:
        temp_q_vals = ser_temp_q.reindex(df["datetime_local"]).values
        if np.isnan(temp_q_vals).all():
            temp_q_vals = np.full_like(irr_q_vals, cfg["panel_temp_ref_c"], dtype=float)
        else:
            temp_q_vals = np.where(np.isnan(temp_q_vals), cfg["panel_temp_ref_c"], temp_q_vals)

    panel_temp_q = temp_q_vals + ((cfg["noct_c"] - 20.0) / 800.0) * irr_q_vals
    panel_eff_q = cfg["panel_eff"] * (1.0 - cfg["panel_temp_beta"] * (panel_temp_q - cfg["panel_temp_ref_c"]))
    panel_eff_q = np.clip(panel_eff_q, 0.0, 1.0)

    solar_energy_q = np.minimum(
        (irr_q_vals / 1000.0) * cfg["panel_area"] * panel_eff_q * cfg["solar_eff"] * 0.25,
        cfg["solar_max_kwh"] * 0.25,
    )

    log(f"✅ Using Open-Meteo irradiance source: {source_used}")
    return irr_q_vals, temp_q_vals, panel_temp_q, panel_eff_q, solar_energy_q


def _parse_trip_time(s):
    if pd.isna(s) or s is None:
        return None
    if isinstance(s, str):
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).time()
            except ValueError:
                pass
        try:
            return pd.to_datetime(s).time()
        except Exception:
            pass
    if hasattr(s, "hour"):
        return s
    return None


def _persist_price_history(prices: pd.DataFrame, cfg: dict, store: HistoryStore) -> None:
    # Persist only Nordpool actuals; exclude forecast sources from history tables.
    prices_for_history = prices
    if "source" in prices.columns:
        prices_for_history = prices[prices["source"].eq("Nordpool")].copy()

    if prices_for_history.empty:
        return

    prices_15m_total = prepare_total_prices_15m(
        prices_15m=prices_for_history,
        tz=cfg["tz"],
        systemtarif=cfg["systemtarif"],
        nettarif_tso=cfg["nettarif_tso"],
        elafgift=cfg["elafgift"],
        tillaeg=cfg["tillaeg"],
    )
    store.save_total_price_15m(prices_15m_total)


def optimize_ev_charging(
    trips: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: dict,
    runtime: dict,
    store: HistoryStore,
    log,
) -> pd.DataFrame:
    assert {"date", "price"}.issubset(prices.columns)
    if not is_datetime64_any_dtype(prices["date"]):
        prices["date"] = pd.to_datetime(prices["date"], utc=True)
    prices = prices.sort_values("date").reset_index(drop=True)

    _persist_price_history(prices, cfg, store)

    charger_min_kw = (cfg["charger_min_a"] * cfg["charger_volt"] * math.sqrt(cfg["phases"])) / 1000.0
    soc_min = cfg["battery_kwh"] * cfg["soc_min_pct"]
    soc_max = cfg["battery_kwh"] * cfg["soc_max_pct"]
    soc0 = cfg["battery_kwh"] * runtime["initial_soc_pct"]
    flat_adders = cfg["systemtarif"] + cfg["nettarif_tso"] + cfg["elafgift"] + cfg["tillaeg"]
    weekday_order = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    df = pd.DataFrame({"datetime_utc": prices["date"], "price": prices["price"]})
    df["price_source"] = prices["source"].values if "source" in prices.columns else "unknown"
    df["datetime_local"] = df["datetime_utc"].dt.tz_convert(cfg["tz"]).dt.floor("15min")
    df = df.sort_values("datetime_local").drop_duplicates(subset=["datetime_local"], keep="last").reset_index(drop=True)
    df["wday_label"] = df["datetime_local"].dt.day_name().str.lower()
    df["hour_local"] = df["datetime_local"].dt.hour
    df["minute_local"] = df["datetime_local"].dt.minute
    df["spot_kr_kwh"] = df["price"] / 100.0

    h = df["hour_local"].values
    df_dates = df["datetime_local"].dt.date
    cutover = pd.Timestamp("2026-04-01").date()
    dso = np.zeros(len(df))

    mask_old = df_dates < cutover
    dso[mask_old & (h >= 0) & (h < 6)] = 0.070375
    dso[mask_old & (h >= 6) & (h < 17)] = 0.21125
    dso[mask_old & (h >= 17) & (h < 21)] = 0.63375
    dso[mask_old & (h >= 21) & (h < 24)] = 0.21125

    mask_new = df_dates >= cutover
    dso[mask_new & (h >= 0) & (h < 6)] = 0.070375
    dso[mask_new & (h >= 6) & (h < 17)] = 0.105625
    dso[mask_new & (h >= 17) & (h < 21)] = 0.274625
    dso[mask_new & (h >= 21) & (h < 24)] = 0.105625

    df["total_price_kr_kwh"] = df["spot_kr_kwh"] + flat_adders + dso

    now = pd.Timestamp.now(tz=cfg["tz"]).floor("15min")
    df = df.loc[df["datetime_local"] >= now].copy().reset_index(drop=True)
    horizon = len(df)

    trips = trips.copy()
    for col in ["away_start", "away_end"]:
        if col in trips.columns:
            trips[col] = trips[col].apply(_parse_trip_time)

    def _slot_index(day_label: str, minutes_of_day: int):
        matches = np.where(
            (df["wday_label"].values == day_label)
            & ((df["hour_local"].values * 60 + df["minute_local"].values) == minutes_of_day)
        )[0]
        return int(matches[0]) if len(matches) else None

    available = np.ones(horizon, dtype=int)
    for _, t in trips.iterrows():
        if pd.isna(t["away_start"]) or pd.isna(t["away_end"]):
            continue
        idx_day = np.where(df["wday_label"].values == t["day"].lower())[0]
        start_minutes = t["away_start"].hour * 60 + t["away_start"].minute
        end_minutes = t["away_end"].hour * 60 + t["away_end"].minute
        minutes_of_day = df["hour_local"].values[idx_day] * 60 + df["minute_local"].values[idx_day]
        mask = (minutes_of_day >= start_minutes) & (minutes_of_day < end_minutes)
        available[idx_day[mask]] = 0
    df["available"] = available

    irr, temp, panel_temp, panel_eff, solar_energy = _build_solar_series(df, cfg, log)
    df["irradiance"] = irr
    df["temperature_2m"] = temp
    df["panel_temp_c"] = panel_temp
    df["panel_efficiency"] = panel_eff
    df["solar_energy"] = solar_energy

    solax = fetch_solax_current_quarter_kwh(cfg["token_id"], cfg["wifi_sn"], cfg["tz"], log)
    if solax is not None:
        now_slot, solar_kwh_now = solax
        mask = df["datetime_local"] == now_slot
        if mask.any():
            old_val = df.loc[mask, "solar_energy"].values[0]
            df.loc[mask, "solar_energy"] = solar_kwh_now
            log(f"✅ Overrode solar_energy at {now_slot}: {old_val:.3f} -> {solar_kwh_now:.3f} kWh (from inverter)")
        store.save_solax_solar_kwh_15m(now_slot, solar_kwh_now)

    hard_soc_max_vec = np.full(horizon, soc_max)
    trip_soc_max_vec = np.full(horizon, np.nan)
    trip_energy_vec = np.zeros(horizon)
    sc_energy_vec = np.zeros(horizon)
    trip_departures = []
    trip_requirements = []
    trip_events = []

    for _, t in trips.iterrows():
        if "distance_km" not in t:
            continue

        need_kwh = float(t["trip_kwh"]) if pd.notna(t.get("trip_kwh")) else float(t["distance_km"]) * runtime["eff_kwh_per_km"]
        if "supercharge_kwh" in t and pd.notna(t["supercharge_kwh"]):
            need_kwh -= float(t["supercharge_kwh"])

        dep_minutes = t["away_start"].hour * 60 + t["away_start"].minute
        idx_dep = df.index[
            (df["wday_label"].values == t["day"].lower())
            & ((df["hour_local"].values * 60 + df["minute_local"].values) == dep_minutes)
        ]
        if len(idx_dep) < 1:
            continue

        h_dep = idx_dep[0]
        trip_energy_vec[h_dep] += need_kwh
        trip_departures.append((h_dep, max(0.0, need_kwh)))
        trip_requirements.append((h_dep, need_kwh, t["day"], t["away_start"]))

        h_end = h_dep
        if pd.notna(t.get("away_end")):
            end_minutes = int(t["away_end"].hour * 60 + t["away_end"].minute)
            end_day = str(t["day"]).lower()
            if end_minutes < dep_minutes:
                end_day = [day for day, idx in weekday_order.items() if idx == ((weekday_order[end_day] + 1) % 7)][0]
            end_idx = _slot_index(end_day, end_minutes)
            if end_idx is not None:
                h_end = end_idx

        if "supercharge_kwh" in t and pd.notna(t["supercharge_kwh"]):
            sc_energy_vec[h_dep] += float(t["supercharge_kwh"])

        trip_events.append(
            {
                "h_dep": h_dep,
                "h_end": h_end,
                "trip_max_kwh": cfg["battery_kwh"] * float(t["max_soc_pct"])
                if "max_soc_pct" in t and pd.notna(t["max_soc_pct"])
                else None,
            }
        )

    trip_events.sort(key=lambda item: item["h_dep"])

    last_trip_end_idx = 0
    for event in trip_events:
        trip_max_kwh = event["trip_max_kwh"]
        if trip_max_kwh is not None:
            existing_trip_caps = trip_soc_max_vec[last_trip_end_idx : event["h_dep"] + 1]
            trip_soc_max_vec[last_trip_end_idx : event["h_dep"] + 1] = np.where(
                np.isnan(existing_trip_caps), trip_max_kwh, np.minimum(existing_trip_caps, trip_max_kwh)
            )

        last_trip_end_idx = max(last_trip_end_idx, event["h_end"])

    hard_soc_max_vec = np.where(np.isnan(trip_soc_max_vec), hard_soc_max_vec, trip_soc_max_vec)

    for h_dep, need_kwh, day, away_start in trip_requirements:
        if soc_min + need_kwh > hard_soc_max_vec[h_dep]:
            raise RuntimeError(f"Trip on {day} {away_start} infeasible (need {need_kwh:.1f} kWh + reserve)")

    soft_soc_window_slots = int(round(max(0.0, cfg["soft_soc_window_hours"]) * 4.0))
    soft_soc_min_window_slots = int(round(max(0.0, cfg["soft_soc_min_window_hours"]) * 4.0))

    soft_extra_cap_vec = np.zeros(horizon)
    for h_dep, dep_trip_kwh in trip_departures:
        if dep_trip_kwh <= 0.0 or soft_soc_window_slots <= 0 or h_dep <= 0:
            continue
        start_idx = max(0, h_dep - soft_soc_window_slots)
        soft_extra_cap_vec[start_idx:h_dep] = np.maximum(soft_extra_cap_vec[start_idx:h_dep], dep_trip_kwh)

    soft_soc_abs_kwh = cfg["battery_kwh"] * min(1.0, max(cfg["soc_max_pct"], cfg["soft_soc_abs_max_pct"]))
    soc_ub_vec = np.minimum(hard_soc_max_vec + soft_extra_cap_vec, soft_soc_abs_kwh)
    soc_ub_vec = np.maximum(soc_ub_vec, hard_soc_max_vec)

    soft_min_relax_vec = np.zeros(horizon)
    for h_dep, dep_trip_kwh in trip_departures:
        if dep_trip_kwh <= 0.0 or soft_soc_min_window_slots <= 0:
            continue
        end_idx = min(horizon, h_dep + soft_soc_min_window_slots + 1)
        soft_min_relax_vec[h_dep:end_idx] = np.maximum(soft_min_relax_vec[h_dep:end_idx], dep_trip_kwh)

    soft_soc_abs_min_kwh = cfg["battery_kwh"] * min(cfg["soc_min_pct"], max(0.0, cfg["soft_soc_abs_min_pct"]))
    soc_lb_vec = np.maximum(soc_min - soft_min_relax_vec, soft_soc_abs_min_kwh)
    soc_lb_vec = np.minimum(soc_lb_vec, soc_min)

    cap_per_quarter = cfg["charger_kw"] * 0.25
    min_per_quarter = charger_min_kw * 0.25

    prob = pulp.LpProblem("ev_charging_opt", pulp.LpMinimize)
    grid = pulp.LpVariable.dicts("grid", range(horizon), lowBound=0, cat=pulp.LpContinuous)
    solar = pulp.LpVariable.dicts("solar", range(horizon), lowBound=0, cat=pulp.LpContinuous)
    z = pulp.LpVariable.dicts("z", range(horizon), cat=pulp.LpBinary)
    s = pulp.LpVariable.dicts("start", range(horizon), cat=pulp.LpBinary)

    first_trip_idx = np.where(trip_energy_vec > 0)[0]
    soc = {}
    for h_i in range(horizon):
        low = soc0 if len(first_trip_idx) == 0 or h_i < first_trip_idx[0] else float(soc_lb_vec[h_i])
        soc[h_i] = pulp.LpVariable(f"soc_{h_i}", lowBound=low, upBound=float(soc_ub_vec[h_i]), cat=pulp.LpContinuous)

    prices_k = df["total_price_kr_kwh"].values
    prob += pulp.lpSum(grid[h_i] * float(prices_k[h_i]) - cfg["refusion"] * solar[h_i] for h_i in range(horizon)) + 0.001 * pulp.lpSum(
        s[h_i] for h_i in range(horizon)
    )

    for h_i in range(horizon):
        if h_i == 0:
            prob += soc[h_i] - runtime["charge_eff"] * (grid[h_i] + solar[h_i]) == (soc0 - float(trip_energy_vec[h_i]))
        else:
            prob += soc[h_i] - soc[h_i - 1] - runtime["charge_eff"] * (grid[h_i] + solar[h_i]) == (-float(trip_energy_vec[h_i]))

    avail = df["available"].values.astype(float)
    solar_cap = df["solar_energy"].values
    for h_i in range(horizon):
        prob += grid[h_i] + solar[h_i] <= (cap_per_quarter * avail[h_i]) * z[h_i]
        prob += solar[h_i] <= solar_cap[h_i] * z[h_i]
        prob += grid[h_i] + solar[h_i] >= min_per_quarter * z[h_i]

    prob += s[0] >= z[0]
    for h_i in range(1, horizon):
        prob += s[h_i] >= z[h_i] - z[h_i - 1]

    trip_rows = np.where(trip_energy_vec > 0)[0]
    for h_i in trip_rows:
        if h_i > 0:
            prob += soc[h_i - 1] >= float(soc_lb_vec[h_i]) + float(trip_energy_vec[h_i])

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"MILP not optimal. Status: {pulp.LpStatus[prob.status]}")

    grid_opt = np.array([pulp.value(grid[h_i]) for h_i in range(horizon)])
    solar_opt = np.array([pulp.value(solar[h_i]) for h_i in range(horizon)])
    soc_opt = np.array([pulp.value(soc[h_i]) for h_i in range(horizon)])

    grid_to_batt = grid_opt * runtime["charge_eff"]
    solar_to_batt = solar_opt * runtime["charge_eff"]

    df_out = pd.DataFrame(
        {
            "datetime_local": df["datetime_local"],
            "weekday": df["wday_label"].values,
            "hour": df["hour_local"].values,
            "minute": df["minute_local"].values,
            "price_kr_per_kwh": np.round(df["total_price_kr_kwh"].values, 5),
            "available": df["available"].values,
            "trip_kwh_at_departure": np.round(trip_energy_vec, 3),
            "sc_kwh": np.round(sc_energy_vec, 3),
            "grid_charge_kwh": np.round(grid_opt, 4),
            "solar_charge_kwh": np.round(solar_opt, 4),
            "total_charge_kwh": np.round(grid_opt + solar_opt, 4),
            "grid_to_batt_kwh": np.round(grid_to_batt, 4),
            "solar_to_batt_kwh": np.round(solar_to_batt, 4),
            "total_to_batt_kwh": np.round(grid_to_batt + solar_to_batt, 4),
            "amp": np.round((((grid_opt + solar_opt) / 0.25) * 1000) / (math.sqrt(cfg["phases"]) * cfg["charger_volt"]), 0),
            "irradiance": df["irradiance"].values,
            "soc_kwh": np.round(soc_opt, 3),
            "hard_soc_min_kwh": np.round(np.full(horizon, soc_min), 3),
            "soft_soc_min_relax_kwh": np.round(soft_min_relax_vec, 3),
            "soc_lower_bound_kwh": np.round(soc_lb_vec, 3),
            "hard_soc_max_kwh": np.round(hard_soc_max_vec, 3),
            "soft_soc_extra_cap_kwh": np.round(soft_extra_cap_vec, 3),
            "soc_upper_bound_kwh": np.round(soc_ub_vec, 3),
            "cost_kr": np.round(grid_opt * df["total_price_kr_kwh"].values, 4),
        }
    )

    return df_out
