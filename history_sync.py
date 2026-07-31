import os

from dotenv import load_dotenv

from ev_charge_opt.history import (
    HistoryStore,
    fetch_solax_current_quarter_kwh,
    prepare_total_prices_15m,
)
from ev_charge_opt.logging_utils import RunLogger
from ev_charge_opt.pricing import fetch_dk1_prices_dkk
from ev_charge_opt.runtime import AppConfig, load_env_inputs


def main() -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(repo_root, ".env.local"))

    logger = RunLogger()
    log = logger.log

    app_cfg = AppConfig()
    env, _ = load_env_inputs(log)

    store = HistoryStore(
        db_name=env["tm_db_name"],
        db_user=env["tm_db_user"],
        db_schema=env["tm_db_schema"],
        db_container=env["tm_db_container"],
        tz=app_cfg.tz,
        log=log,
    )

    store.ensure_history_tables()

    prices_actual = fetch_dk1_prices_dkk(app_cfg.tz, log, resolution_minutes=env["price_resolution_minutes"])
    prices_15m_total = prepare_total_prices_15m(
        prices_15m=prices_actual,
        tz=app_cfg.tz,
        systemtarif=app_cfg.systemtarif,
        nettarif_tso=app_cfg.nettarif_tso,
        elafgift=app_cfg.elafgift,
        tillaeg=app_cfg.tillaeg,
    )
    store.save_total_price_15m(prices_15m_total)

    solax = fetch_solax_current_quarter_kwh(
        token_id=env["token_id"],
        wifi_sn=env["wifi_sn"],
        tz=app_cfg.tz,
        log=log,
    )
    if solax is not None:
        slot_local, solar_kwh = solax
        store.save_solax_solar_kwh_15m(slot_local, solar_kwh)

    store.save_charging_session_summary()

    log("✅ Independent history sync completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
