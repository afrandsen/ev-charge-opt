import time
from datetime import datetime, timedelta

import pandas as pd
import requests


def _validate_quarter_grid(df: pd.DataFrame, source: str) -> None:
    if df.empty:
        return

    ts = pd.to_datetime(df["date"], utc=True)
    bad_minute = ~ts.dt.minute.isin([0, 15, 30, 45])
    if bad_minute.any():
        bad = ts.loc[bad_minute].iloc[0]
        raise RuntimeError(f"{source} returned non-15m timestamp: {bad}")

    dupes = ts.duplicated()
    if dupes.any():
        dup = ts.loc[dupes].iloc[0]
        raise RuntimeError(f"{source} returned duplicate timestamp: {dup}")


def fetch_dk1_prices_dkk(tz: str, log, attempts: int = 5) -> pd.DataFrame:
    today = datetime.now().date()
    now_cet = pd.Timestamp.now(tz=tz)
    fetch_tomorrow = now_cet.hour > 12 or (now_cet.hour == 12 and now_cet.minute >= 45)
    log(
        f"{'🟢' if fetch_tomorrow else '🟡'} It is {now_cet.strftime('%H:%M %Z')} -> "
        f"{'Nordpool tomorrow data should be available.' if fetch_tomorrow else 'Too early, skipping tomorrow fetch.'}"
    )

    from nordpool import elspot

    p = elspot.Prices(currency="DKK")
    dfs = []
    for offset in range(1 + int(fetch_tomorrow)):
        date_str = (today + timedelta(days=offset)).strftime("%Y-%m-%d")
        rows = None
        for attempt in range(attempts):
            try:
                data = p.hourly(end_date=date_str, areas=["DK1"], resolution=15)
                values = data["areas"]["DK1"]["values"]
                rows = [
                    {
                        "date": pd.to_datetime(v["start"], utc=True),
                        "price": (v["value"] / 10.0) * 1.25,
                        "source": "Nordpool",
                    }
                    for v in values
                    if v["value"] is not None
                ]
                log(f"✅ Nordpool success for {date_str} on attempt {attempt+1}")
                break
            except Exception as e:
                if "resolution" in str(e).lower():
                    raise RuntimeError(
                        "Nordpool client does not support 15-minute resolution. "
                        "Upgrade the nordpool package version."
                    ) from e
                log(f"Nordpool fetch failed {date_str} (attempt {attempt+1}/{attempts}): {e}")
                time.sleep(2)
        if rows is not None:
            dfs.append(pd.DataFrame(rows))
        else:
            log(f"⚠️ Nordpool prices not yet available for {date_str}, skipping")

    if not dfs:
        raise RuntimeError("No Nordpool data available")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    _validate_quarter_grid(df, source="Nordpool")
    return df


def fetch_eur_dkk_exchange_rate(log, attempts: int = 5, sleep_sec: int = 1) -> float:
    exchange_url = "https://api.exchangerate-api.com/v4/latest/EUR"
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(exchange_url, timeout=10)
            r.raise_for_status()
            data = r.json()
            rate = data["rates"]["DKK"]
            log(f"✅ EUR/DKK exchange rate: {rate}")
            return rate
        except Exception as e:
            log(f"⚠️ Exchange rate fetch failed (attempt {attempt}/{attempts}): {e}")
            if attempt < attempts:
                time.sleep(sleep_sec)

    log("❌ Exchange rate fetch failed. Using fallback rate of 7.47")
    return 7.47


def fetch_epex_forecast_dkk(log, attempts: int = 5, sleep_sec: int = 2) -> pd.DataFrame:
    df_epex = pd.DataFrame(columns=["date", "price", "source"])
    exchange_rate = fetch_eur_dkk_exchange_rate(log)

    epex_url = "https://epexpredictor.batzill.com/prices"
    params = {
        "hours": -1,
        "surcharge": 0,
        "taxPercent": 0,
        "region": "DK1",
        "evaluation": False,
        "unit": "EUR_PER_MWH",
        "hourly": False,
        "timezone": "Europe/Copenhagen",
    }

    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(epex_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            df_epex_temp = pd.DataFrame(data["prices"])
            df_epex_temp["date"] = pd.to_datetime(df_epex_temp["startsAt"], utc=True)
            df_epex_temp["price"] = (df_epex_temp["total"] / 10) * exchange_rate * 1.25
            df_epex_temp["source"] = "EPEX"
            df_epex = df_epex_temp[["date", "price", "source"]]
            _validate_quarter_grid(df_epex, source="EPEX")
            log(f"✅ EPEX forecast success ({len(df_epex)} slots) on attempt {attempt}")
            break
        except Exception as e:
            log(f"⚠️ EPEX forecast fetch failed (attempt {attempt}/{attempts}): {e}")
            if attempt < attempts:
                time.sleep(sleep_sec)
            else:
                log("❌ EPEX: All attempts failed. Returning empty DataFrame.")

    return df_epex


def combine_actuals_and_forecast(prices_actual: pd.DataFrame, prices_forecast: pd.DataFrame) -> pd.DataFrame:
    last_actual = prices_actual["date"].max()
    future = prices_forecast[prices_forecast["date"] > last_actual]
    df = pd.concat([prices_actual, future], ignore_index=True).sort_values("date").reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC").floor("15min") - timedelta(hours=2)
    df = df[df["date"] >= now]
    _validate_quarter_grid(df, source="Combined price timeline")
    return df.reset_index(drop=True)
