import os
import sys
from datetime import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ev_charge_opt.history import HistoryStore
from ev_charge_opt.logging_utils import RunLogger
from ev_charge_opt.notifications import send_email_notification
from ev_charge_opt.optimizer import optimize_ev_charging
from ev_charge_opt.pricing import combine_actuals_and_forecast, fetch_dk1_prices_dkk, fetch_epex_forecast_dkk
from ev_charge_opt.runtime import AppConfig, load_env_inputs, parse_runtime_inputs, shift_active_trip_windows
from ev_charge_opt.state import load_state, save_state, should_notify


QUIET_START = time(0, 0)
QUIET_END = time(6, 45)


def in_quiet_hours(now: pd.Timestamp) -> bool:
    return QUIET_START <= now.time() <= QUIET_END


def _format_event_log(df_out: pd.DataFrame, log) -> None:
    stored_this_slot = df_out["total_to_batt_kwh"].values
    trip_this_slot = df_out["trip_kwh_at_departure"].values

    soc_kwh_before = np.where(
        (df_out["grid_charge_kwh"].values + df_out["solar_charge_kwh"].values) > 0,
        df_out["soc_kwh"].values - stored_this_slot,
        df_out["soc_kwh"].values + trip_this_slot,
    )

    df_out["soc_kwh_before"] = soc_kwh_before
    df_out["soc_pct_before"] = np.round((df_out["soc_kwh_before"].values / 75.0) * 100.0, 1)
    df_out["soc_pct_after"] = np.round((df_out["soc_kwh"].values / 75.0) * 100.0, 1)
    df_out["effective_price_kr_per_kwh_drawn"] = df_out["cost_kr"] / df_out["total_charge_kwh"].replace(0, np.nan)

    mask_events = (
        (df_out["trip_kwh_at_departure"].values > 0)
        | (df_out["grid_charge_kwh"].values > 0)
        | (df_out["solar_charge_kwh"].values > 0)
    )

    log("\n=== Charging & Trip Events (Phone View) ===")
    header = f"{'dt':<11} {'wd':<3} {'A':>2} {'SoC b->a':>9} {'g/s/t kWh':>14} {'trip':>5} {'kr':>5} {'eff':>5}"
    log(header)
    log("-" * len(header))

    for _, row in df_out.loc[mask_events].iterrows():
        eff_price = row["effective_price_kr_per_kwh_drawn"]
        eff_str = f"{eff_price:>5.2f}" if pd.notna(eff_price) else "    -"
        log(
            f"{row['datetime_local']:%m-%d %H:%M} "
            f"{str(row['weekday'])[:3]:<3} "
            f"{int(row['amp']):>2d} "
            f"{row['soc_pct_before']:>4.1f}->{row['soc_pct_after']:<4.1f} "
            f"{row['grid_charge_kwh']:>4.2f}/{row['solar_charge_kwh']:>4.2f}/{row['total_charge_kwh']:>4.2f} "
            f"{row['trip_kwh_at_departure']:>5.2f} "
            f"{row['price_kr_per_kwh']:>5.2f} "
            f"{eff_str}"
        )


def main(argv=None) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, ".env.local"))
    argv = argv or sys.argv

    logger = RunLogger()
    log = logger.log

    runtime = parse_runtime_inputs(argv, log)
    log(f"Latest SOC received from shell: {round(runtime.initial_soc_pct * 100, 2)}%")
    log(f"Latest 10 drives average efficiency received from shell: {round(runtime.eff_kwh_per_km, 3)} kWh/km")
    log(f"Latest charging efficiency received from shell: {round(runtime.charge_eff * 100, 2)}%")

    app_cfg = AppConfig()
    env, trips = load_env_inputs(log)
    trips = shift_active_trip_windows(trips, runtime.is_home, app_cfg.tz, log)

    store = HistoryStore(
        db_name=env["tm_db_name"],
        db_user=env["tm_db_user"],
        db_schema=env["tm_db_schema"],
        db_container=env["tm_db_container"],
        tz=app_cfg.tz,
        log=log,
    )
    store.ensure_history_tables()

    prices_actual = fetch_dk1_prices_dkk(app_cfg.tz, log)
    prices_forecast = fetch_epex_forecast_dkk(log)
    prices = combine_actuals_and_forecast(prices_actual=prices_actual, prices_forecast=prices_forecast)

    cfg = {
        "battery_kwh": app_cfg.battery_kwh,
        "charger_kw": app_cfg.charger_kw,
        "charger_min_a": app_cfg.charger_min_a,
        "charger_volt": app_cfg.charger_volt,
        "phases": app_cfg.phases,
        "solar_eff": app_cfg.solar_eff,
        "panel_area": app_cfg.panel_area,
        "panel_eff": app_cfg.panel_eff,
        "noct_c": app_cfg.noct_c,
        "panel_temp_beta": app_cfg.panel_temp_beta,
        "panel_temp_ref_c": app_cfg.panel_temp_ref_c,
        "solar_max_kwh": app_cfg.solar_max_kwh,
        "systemtarif": app_cfg.systemtarif,
        "nettarif_tso": app_cfg.nettarif_tso,
        "elafgift": app_cfg.elafgift,
        "tillaeg": app_cfg.tillaeg,
        "refusion": app_cfg.refusion,
        "tilt": app_cfg.tilt,
        "azimuth": app_cfg.azimuth,
        "tz": app_cfg.tz,
        "soc_min_pct": env["soc_min_pct"],
        "soc_max_pct": env["soc_max_pct"],
        "lat": env["lat"],
        "lon": env["lon"],
        "token_id": env["token_id"],
        "wifi_sn": env["wifi_sn"],
        "soft_soc_window_hours": env["soft_soc_window_hours"],
        "soft_soc_abs_max_pct": env["soft_soc_abs_max_pct"],
        "soft_soc_min_window_hours": env["soft_soc_min_window_hours"],
        "soft_soc_abs_min_pct": env["soft_soc_abs_min_pct"],
    }
    runtime_dict = {
        "initial_soc_pct": runtime.initial_soc_pct,
        "eff_kwh_per_km": runtime.eff_kwh_per_km,
        "charge_eff": runtime.charge_eff,
    }

    try:
        df_out = optimize_ev_charging(trips=trips, prices=prices, cfg=cfg, runtime=runtime_dict, store=store, log=log)
    except RuntimeError as e:
        send_email_notification(
            subject="❌ EV Charging Optimization Failed",
            body=str(e),
            sender=os.getenv("EMAIL_SENDER"),
            recipient=os.getenv("EMAIL_RECIPIENT"),
            smtp_server=os.getenv("SMTP_SERVER"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            username=os.getenv("SMTP_USER"),
            password=os.getenv("SMTP_PASS"),
            log=log,
        )
        return 1

    _format_event_log(df_out, log)

    total_cost = float(df_out["cost_kr"].sum())
    total_drawn = float(df_out["total_charge_kwh"].sum())
    from_grid_drawn = float(df_out["grid_charge_kwh"].sum())
    from_solar_drawn = float(df_out["solar_charge_kwh"].sum())
    total_stored = float(df_out["total_to_batt_kwh"].sum())
    from_grid_stored = float(df_out["grid_to_batt_kwh"].sum())
    from_solar_stored = float(df_out["solar_to_batt_kwh"].sum())
    avg_drawn = (total_cost / total_drawn) if total_drawn > 0 else float("nan")
    avg_stored = (total_cost / total_stored) if total_stored > 0 else float("nan")

    log(
        f"Total cost: {total_cost:.2f} kr. "
        f"Energy drawn: {total_drawn:.2f} kWh ({from_grid_drawn:.2f} grid, {from_solar_drawn:.2f} solar). "
        f"Energy stored: {total_stored:.2f} kWh ({from_grid_stored:.2f} grid, {from_solar_stored:.2f} solar). "
        f"Avg cost: {avg_drawn:.2f} kr/kWh drawn, {avg_stored:.2f} kr/kWh stored."
    )

    now_slot = pd.Timestamp.now(tz=app_cfg.tz).floor("15min")
    current_row = df_out.iloc[0]
    current_amp = int(current_row["amp"])

    state = load_state()
    last_amp = int(state.get("last_amp", 0))
    notify, reason = should_notify(current_amp, last_amp)

    save_state({"last_amp": current_amp, "target_soc": float(current_row["soc_pct_after"] / 100.0)})

    forced_send = now_slot.time().hour == 21 and now_slot.time().minute == 0
    if notify or forced_send:
        if in_quiet_hours(now_slot):
            log(f"🌙 Quiet hours ({now_slot.time()}), no email sent ({reason})")
        else:
            body = (
                f"{reason} at {now_slot}.\n\n"
                f"Amps: {current_amp}\n"
                f"Grid: {current_row['grid_charge_kwh']:.2f} kWh\n"
                f"Solar: {current_row['solar_charge_kwh']:.2f} kWh\n"
                f"SoC before: {current_row['soc_pct_before']:.1f}%\n"
                f"SoC after:  {current_row['soc_pct_after']:.1f}%\n"
            )
            if logger.lines:
                body += "\n\n=== Status & Log ===\n" + "\n".join(logger.lines)

            send_email_notification(
                subject=f"EV Charging Alert: {current_amp}A at {now_slot.strftime('%H:%M')}",
                body=body,
                sender=os.getenv("EMAIL_SENDER"),
                recipient=os.getenv("EMAIL_RECIPIENT"),
                smtp_server=os.getenv("SMTP_SERVER"),
                smtp_port=int(os.getenv("SMTP_PORT", "587")),
                username=os.getenv("SMTP_USER"),
                password=os.getenv("SMTP_PASS"),
                log=log,
            )
    else:
        log(f"ℹ️ No email sent ({reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
