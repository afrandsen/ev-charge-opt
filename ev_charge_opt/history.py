import re
import subprocess
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests


class HistoryStore:
    def __init__(
        self,
        db_name: str,
        db_user: str,
        db_schema: str,
        db_container: str,
        tz: str,
        log,
    ) -> None:
        self.db_name = db_name
        self.db_user = db_user
        self.db_schema = db_schema
        self.db_container = db_container
        self.tz = tz
        self.log = log

    def _to_sql_local_timestamp(self, ts: pd.Timestamp) -> str:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(self.tz).tz_localize(None)
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _valid_pg_identifier(name: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))

    def _history_table(self, base_table: str) -> Optional[str]:
        if not self._valid_pg_identifier(self.db_schema):
            self.log("⚠️ Direct Postgres write failed: TM_DB_SCHEMA must be a valid Postgres identifier")
            return None
        if not self._valid_pg_identifier(base_table):
            self.log("⚠️ Direct Postgres write failed: invalid history table identifier")
            return None
        return f"{self.db_schema}.{base_table}"

    def _run_psql(self, sql: str) -> bool:
        if not self.db_container:
            self.log("⚠️ Docker exec DB write failed: TM_DB_CONTAINER is not set")
            return False

        cmd = [
            "docker",
            "exec",
            "-i",
            self.db_container,
            "psql",
            "-U",
            self.db_user,
            self.db_name,
            "-v",
            "ON_ERROR_STOP=1",
        ]
        try:
            subprocess.run(cmd, input=sql, text=True, capture_output=True, check=True)
            return True
        except Exception as e:
            self.log(f"⚠️ Docker exec DB write failed: {e}")
            return False

    def ensure_history_tables(self) -> bool:
        solax_table = self._history_table("ev_charge_opt_solax_ac_power_15m")
        nordpool_table = self._history_table("ev_charge_opt_nordpool_spot_price_15m")
        charging_session_table = self._history_table("ev_charge_opt_charging_session_summary")
        if not solax_table or not nordpool_table or not charging_session_table:
            return False

        sql = f"""
        CREATE SCHEMA IF NOT EXISTS {self.db_schema};

        CREATE TABLE IF NOT EXISTS {solax_table} (
            slot_local timestamptz PRIMARY KEY,
            solar_kwh_now double precision NOT NULL,
            sample_count integer NOT NULL DEFAULT 1,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {nordpool_table} (
            slot_local timestamptz PRIMARY KEY,
            spot_price_kr_per_kwh double precision,
            total_price_kr_per_kwh double precision NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {charging_session_table} (
            charging_process_id bigint PRIMARY KEY,
            start_date timestamp without time zone,
            end_date timestamp without time zone,
            charged_kwh double precision NOT NULL,
            solar_kwh double precision NOT NULL,
            grid_kwh double precision NOT NULL,
            session_cost_kr double precision NOT NULL,
            cost_kr_per_kwh double precision,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );

        ALTER TABLE {nordpool_table}
        ADD COLUMN IF NOT EXISTS spot_price_kr_per_kwh double precision;

        ALTER TABLE {solax_table}
        ADD COLUMN IF NOT EXISTS sample_count integer NOT NULL DEFAULT 1;
        """
        return self._run_psql(sql)

    def save_solax_solar_kwh_15m(self, slot_local: pd.Timestamp, solar_kwh_now: float) -> bool:
        solax_table = self._history_table("ev_charge_opt_solax_ac_power_15m")
        if not solax_table:
            return False

        slot_local_sql = self._to_sql_local_timestamp(slot_local)
        value = float(solar_kwh_now)
        sql = f"""
        INSERT INTO {solax_table} (slot_local, solar_kwh_now, sample_count)
        VALUES ('{slot_local_sql}'::timestamp AT TIME ZONE '{self.tz}', {value}, 1)
        ON CONFLICT (slot_local) DO UPDATE
        SET solar_kwh_now = (
                ({solax_table}.solar_kwh_now * {solax_table}.sample_count)
                + EXCLUDED.solar_kwh_now
            ) / ({solax_table}.sample_count + 1),
            sample_count = {solax_table}.sample_count + 1,
            updated_at = now();
        """
        return self._run_psql(sql)

    def save_total_price_15m(self, expanded_prices: pd.DataFrame) -> bool:
        nordpool_table = self._history_table("ev_charge_opt_nordpool_spot_price_15m")
        if not nordpool_table:
            return False

        if expanded_prices.empty:
            self.log("⚠️ No total prices available to persist")
            return False

        values_sql = []
        for row in expanded_prices.itertuples(index=False):
            slot_local_sql = self._to_sql_local_timestamp(pd.Timestamp(row.slot_local))
            spot_price = float(row.spot_price_kr_per_kwh)
            total_price = float(row.total_price_kr_per_kwh)
            values_sql.append(f"('{slot_local_sql}'::timestamp AT TIME ZONE '{self.tz}', {spot_price}, {total_price})")

        sql = f"""
        INSERT INTO {nordpool_table} (slot_local, spot_price_kr_per_kwh, total_price_kr_per_kwh)
        VALUES
        """ + ",\n".join(values_sql) + """
        ON CONFLICT (slot_local) DO UPDATE
        SET spot_price_kr_per_kwh = EXCLUDED.spot_price_kr_per_kwh,
            total_price_kr_per_kwh = EXCLUDED.total_price_kr_per_kwh,
            updated_at = now();
        """
        return self._run_psql(sql)

    def save_charging_session_summary(self) -> bool:
        charging_session_table = self._history_table("ev_charge_opt_charging_session_summary")
        nordpool_table = self._history_table("ev_charge_opt_nordpool_spot_price_15m")
        solax_table = self._history_table("ev_charge_opt_solax_ac_power_15m")
        if not charging_session_table or not nordpool_table or not solax_table:
            return False

        sql = f"""
        INSERT INTO {charging_session_table} (
            charging_process_id,
            start_date,
            end_date,
            charged_kwh,
            solar_kwh,
            grid_kwh,
            session_cost_kr,
            cost_kr_per_kwh
        )
        WITH charge_deltas AS (
            SELECT
                c.charging_process_id,
                c.date,
                greatest(
                    c.charge_energy_added
                    - lag(c.charge_energy_added) OVER (
                        PARTITION BY c.charging_process_id
                        ORDER BY c.date
                    ),
                    0
                ) AS delta_kwh
            FROM charges c
            WHERE c.charge_energy_added IS NOT NULL
        ),

        avg_efficiency AS (
            SELECT
                SUM(charge_energy_used::numeric) / NULLIF(SUM(charge_energy_added), 0) AS ratio
            FROM charging_processes
            WHERE charge_energy_used IS NOT NULL
              AND charge_energy_added IS NOT NULL
              AND charge_energy_added > 0
              AND address_id = 1
        ),

        charge_buckets AS (
            SELECT
                cd.charging_process_id,
                cp.start_date,
                cp.end_date,
                date_trunc('hour', cd.date)
                    + floor(extract(minute FROM cd.date) / 15) * interval '15 minutes' AS bucket,

                sum(cd.delta_kwh)
                    * (
                        COALESCE(
                            cp.charge_energy_used::numeric / NULLIF(cp.charge_energy_added, 0),
                            ae.ratio
                        )
                    ) AS charge_kwh

            FROM charge_deltas cd
            JOIN charging_processes cp
                ON cp.id = cd.charging_process_id
               AND cp.address_id = 1
            CROSS JOIN avg_efficiency ae
            WHERE cd.delta_kwh IS NOT NULL
            GROUP BY
                cd.charging_process_id,
                cp.start_date,
                cp.end_date,
                bucket,
                cp.charge_energy_added,
                cp.charge_energy_used,
                ae.ratio
        )

        SELECT
            cb.charging_process_id,
            cb.start_date,
            coalesce(max(cb.end_date), max(cb.bucket) + interval '15 minutes') AS end_date,

            round(sum(cb.charge_kwh)::numeric, 4) AS charged_kwh,

            round(
                sum(coalesce(s.solar_kwh_now, 0))::numeric,
                4
            ) AS solar_kwh,

            round(
                sum(
                    greatest(
                        cb.charge_kwh - coalesce(s.solar_kwh_now, 0),
                        0
                    )
                )::numeric,
                4
            ) AS grid_kwh,

            round(
                sum(
                    greatest(
                        cb.charge_kwh - coalesce(s.solar_kwh_now, 0),
                        0
                    ) * p.total_price_kr_per_kwh
                )::numeric,
                4
            ) AS session_cost_kr,

            round(
                (
                    sum(
                        greatest(
                            cb.charge_kwh - coalesce(s.solar_kwh_now, 0),
                            0
                        ) * p.total_price_kr_per_kwh
                    )
                    /
                    nullif(sum(cb.charge_kwh), 0)
                )::numeric,
                4
            ) AS cost_kr_per_kwh
        FROM charge_buckets cb

        JOIN {nordpool_table} p
            ON p.slot_local = cb.bucket

        LEFT JOIN {solax_table} s
            ON s.slot_local = cb.bucket

        GROUP BY
            cb.charging_process_id,
            cb.start_date
        ON CONFLICT (charging_process_id) DO UPDATE
        SET start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            charged_kwh = EXCLUDED.charged_kwh,
            solar_kwh = EXCLUDED.solar_kwh,
            grid_kwh = EXCLUDED.grid_kwh,
            session_cost_kr = EXCLUDED.session_cost_kr,
            cost_kr_per_kwh = EXCLUDED.cost_kr_per_kwh,
            updated_at = now();
        """
        return self._run_psql(sql)


def prepare_total_prices_hourly(
    prices_hourly: pd.DataFrame,
    tz: str,
    systemtarif: float,
    nettarif_tso: float,
    elafgift: float,
    tillaeg: float,
) -> pd.DataFrame:
    if prices_hourly.empty:
        return pd.DataFrame(columns=["date", "spot_kr_per_kwh", "total_price_kr_per_kwh"])

    df = prices_hourly[["date", "price"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["datetime_local"] = df["date"].dt.tz_convert(tz)
    df["spot_kr_per_kwh"] = df["price"] / 100.0

    h = df["datetime_local"].dt.hour.values
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

    flat_adders = systemtarif + nettarif_tso + elafgift + tillaeg
    df["total_price_kr_per_kwh"] = df["spot_kr_per_kwh"] + dso + flat_adders
    return df[["date", "spot_kr_per_kwh", "total_price_kr_per_kwh"]]


def expand_hourly_total_prices_to_quarters(prices_hourly_total: pd.DataFrame, tz: str) -> pd.DataFrame:
    if prices_hourly_total.empty:
        return pd.DataFrame(columns=["slot_local", "spot_price_kr_per_kwh", "total_price_kr_per_kwh"])

    df = prices_hourly_total[["date", "spot_kr_per_kwh", "total_price_kr_per_kwh"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    repeated = df.loc[df.index.repeat(4)].copy().reset_index(drop=True)
    repeated["slot_local"] = (
        repeated["date"].dt.tz_convert(tz)
        + pd.to_timedelta(np.tile([0, 15, 30, 45], len(df)), unit="m")
    ).dt.tz_localize(None)
    repeated.rename(columns={"spot_kr_per_kwh": "spot_price_kr_per_kwh"}, inplace=True)
    return repeated[["slot_local", "spot_price_kr_per_kwh", "total_price_kr_per_kwh"]]


def fetch_solax_current_quarter_kwh(
    token_id: str,
    wifi_sn: str,
    tz: str,
    log,
    attempts: int = 5,
    sleep_sec: int = 2,
) -> Optional[Tuple[pd.Timestamp, float]]:
    if not token_id or not wifi_sn:
        log("⚠️ Solax credentials missing, skipping inverter history sync")
        return None

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
                log(f"✅ Solax realtime success on attempt {attempt}: {solar_kwh_now:.3f} kWh")
                return now_slot, solar_kwh_now
            raise ValueError("Inverter API returned no data or missing acpower")
        except Exception as e:
            log(f"⚠️ Solax realtime fetch failed (attempt {attempt}/{attempts}): {e}")
            if attempt < attempts:
                time.sleep(sleep_sec)

    log("⚠️ Solax realtime sync failed after all retries")
    return None
