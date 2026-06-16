import os

from dotenv import load_dotenv

from ev_charge_opt.history import HistoryStore, fetch_solax_current_quarter_kwh
from ev_charge_opt.logging_utils import RunLogger
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

    solax = fetch_solax_current_quarter_kwh(
        token_id=env["token_id"],
        wifi_sn=env["wifi_sn"],
        tz=app_cfg.tz,
        log=log,
    )
    if solax is None:
        return 1

    slot_local, solar_kwh = solax
    store.save_solax_solar_kwh_15m(slot_local, solar_kwh)
    store.save_charging_session_summary()

    log(f"✅ Solar-only sync completed for {slot_local} (quarter estimate avg updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
