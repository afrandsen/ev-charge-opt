import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from ev_charge_opt.history import fetch_solax_current_quarter_kwh
from ev_charge_opt.optimizer import _align_to_quarters, _build_solar_series, optimize_ev_charging
from ev_charge_opt.runtime import shift_active_trip_windows


TZ = "Europe/Copenhagen"


def local_day_slots(day):
    start = pd.Timestamp(day, tz=TZ)
    end = start + pd.DateOffset(days=1)
    return pd.date_range(start, end, freq="15min", inclusive="left")


@contextmanager
def frozen_clock(module, instant):
    with patch(f"{module}.pd", wraps=pd) as fake_pd:
        fake_pd.Timestamp.now.side_effect = lambda tz: instant.tz_convert(tz)
        yield


class DstHandlingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "tz": TZ,
            "charger_min_a": 6,
            "charger_volt": 400,
            "phases": 3,
            "charger_kw": 11,
            "battery_kwh": 75,
            "soc_min_pct": 0.2,
            "soc_max_pct": 0.8,
            "systemtarif": 0.1,
            "nettarif_tso": 0.1,
            "elafgift": 0.1,
            "tillaeg": 0.1,
            "refusion": 0.0,
            "soft_soc_window_hours": 3,
            "soft_soc_min_window_hours": 1,
            "soft_soc_abs_min_pct": 0.1,
            "soft_soc_abs_max_pct": 1.0,
            "token_id": "",
            "wifi_sn": "",
            "lat": 56.0,
            "lon": 10.0,
            "tilt": 30,
            "azimuth": 0,
            "panel_temp_ref_c": 25.0,
            "noct_c": 45.0,
            "panel_eff": 0.2,
            "panel_temp_beta": 0.004,
            "panel_area": 20.0,
            "solar_eff": 0.9,
            "solar_max_kwh": 5.0,
        }
        self.runtime = {
            "initial_soc_pct": 0.4,
            "charge_eff": 0.9285,
            "eff_kwh_per_km": 0.144,
        }

    def weather_response(self, mode):
        def fetch(url, value_path, log):
            query = parse_qs(urlparse(url).query)
            self.assertEqual(query["timezone"], ["UTC"])
            if mode == "hourly" and value_path[0] == "minutely_15":
                return None, None
            start = pd.Timestamp(query["start_date"][0], tz="UTC")
            end = pd.Timestamp(query["end_date"][0], tz="UTC") + pd.Timedelta(days=1)
            freq = "h" if value_path[0] == "hourly" else "15min"
            times = pd.date_range(start, end, freq=freq, inclusive="left")
            if mode == "mixed" and value_path[0] == "minutely_15":
                times = times[::2]
            values = times.hour * 10.0 + times.minute
            if value_path[-1] == "temperature_2m":
                values = np.full(len(times), 10.0)
            return times.tz_localize(None).strftime("%Y-%m-%dT%H:%M").tolist(), list(values)

        return fetch

    def optimize(self, slots, now, weather_mode=None):
        prices = pd.DataFrame({"date": slots.tz_convert("UTC"), "price": 100.0, "source": "EPEX"})
        if weather_mode is None:
            solar_patch = patch(
                "ev_charge_opt.optimizer._build_solar_series",
                side_effect=lambda frame, cfg, log: tuple(np.zeros(len(frame)) for _ in range(5)),
            )
        else:
            solar_patch = patch(
                "ev_charge_opt.optimizer._fetch_open_meteo_with_retries",
                side_effect=self.weather_response(weather_mode),
            )
        with frozen_clock("ev_charge_opt.optimizer", now), solar_patch, patch(
            "ev_charge_opt.optimizer.fetch_solax_current_quarter_kwh", return_value=None
        ):
            return optimize_ev_charging(pd.DataFrame(), prices, self.cfg, self.runtime, Mock(), Mock())

    def test_optimizer_preserves_complete_local_days(self):
        for day, count in (("2026-10-25", 100), ("2026-03-29", 92), ("2026-10-22", 96)):
            with self.subTest(day=day):
                slots = local_day_slots(day)
                result = self.optimize(slots, slots[0])
                actual = pd.DatetimeIndex(result["datetime_local"])
                self.assertEqual(len(actual), count)
                self.assertTrue(actual.is_unique)
                self.assertFalse(actual.hasnans)
                pd.testing.assert_index_equal(actual, slots, check_names=False)
                self.assertTrue(np.isfinite(result.select_dtypes(include="number")).all().all())
                if count == 100:
                    offsets = {stamp.strftime("%z") for stamp in actual if stamp.hour == 2}
                    self.assertEqual(offsets, {"+0200", "+0100"})
                elif count == 92:
                    self.assertNotIn(2, actual.hour)

    def test_optimizer_floors_current_instant_without_changing_offset(self):
        for instant in (
            "2026-10-25T00:07:00Z",
            "2026-10-25T01:07:00Z",
            "2026-03-29T00:57:00Z",
            "2026-03-29T01:07:00Z",
            "2026-10-22T10:07:00Z",
        ):
            with self.subTest(instant=instant):
                now = pd.Timestamp(instant)
                slots = local_day_slots(now.tz_convert(TZ).strftime("%Y-%m-%d"))
                result = self.optimize(slots, now)
                expected = now.floor("15min").tz_convert(TZ)
                self.assertEqual(result["datetime_local"].iloc[0], expected)
                self.assertEqual(len(result), len(slots[slots >= expected]))

    def test_runtime_and_solax_use_correct_current_quarter(self):
        for instant in (
            "2026-10-25T00:07:00Z",
            "2026-10-25T01:07:00Z",
            "2026-03-29T00:57:00Z",
            "2026-03-29T01:07:00Z",
            "2026-10-22T10:07:00Z",
        ):
            with self.subTest(instant=instant):
                now = pd.Timestamp(instant)
                expected = now.floor("15min").tz_convert(TZ)
                trips = pd.DataFrame([{
                    "day": expected.day_name().lower(),
                    "away_start": expected.strftime("%H:%M"),
                    "away_end": "23:45",
                }])
                with frozen_clock("ev_charge_opt.runtime", now):
                    result = shift_active_trip_windows(trips, True, TZ, Mock())
                next_minutes = expected.hour * 60 + expected.minute + 15
                self.assertEqual(result.loc[0, "away_start"], f"{next_minutes // 60:02d}:{next_minutes % 60:02d}")

                response = Mock()
                response.json.return_value = {"success": True, "result": {"acpower": 2000}}
                with frozen_clock("ev_charge_opt.history", now), patch(
                    "ev_charge_opt.history.requests.post", return_value=response
                ) as post:
                    result = fetch_solax_current_quarter_kwh("token", "serial", TZ, Mock(), sleep_sec=0)
                self.assertIsNotNone(result)
                self.assertEqual(result[0], expected)
                self.assertEqual(result[0].strftime("%z"), expected.strftime("%z"))
                self.assertEqual(result[1], 0.5)
                post.assert_called_once()

    def test_weather_alignment_covers_local_days_and_both_repeated_hours(self):
        for day in ("2026-10-25", "2026-03-29", "2026-10-22"):
            for mode in ("15min", "hourly", "mixed"):
                with self.subTest(day=day, mode=mode):
                    slots = local_day_slots(day)
                    frame = pd.DataFrame({"datetime_local": slots})
                    with patch(
                        "ev_charge_opt.optimizer._fetch_open_meteo_with_retries",
                        side_effect=self.weather_response(mode),
                    ):
                        arrays = _build_solar_series(frame, self.cfg, Mock())
                    for values in arrays:
                        self.assertEqual(len(values), len(slots))
                        self.assertTrue(np.isfinite(values).all())
                    utc_slots = slots.tz_convert("UTC")
                    expected = utc_slots.hour * 10.0
                    if mode == "15min":
                        expected = expected + utc_slots.minute
                    elif mode == "mixed":
                        expected = expected + np.where(utc_slots.minute % 30 == 0, utc_slots.minute, 0)
                    np.testing.assert_allclose(arrays[0], expected)
                    if day == "2026-10-25":
                        repeated = arrays[0][(slots.hour == 2) & (slots.minute == 0)]
                        np.testing.assert_allclose(repeated, [0.0, 10.0])

    def test_hourly_alignment_preserves_distinct_absolute_slots(self):
        series = _align_to_quarters(
            ["2026-10-25T00:00", "2026-10-25T01:00"], [10.0, 20.0], TZ, repeat_to_quarters=True
        )
        self.assertEqual(len(series), 8)
        self.assertTrue(series.index.is_unique)
        self.assertTrue((series.index.hour == 2).all())
        np.testing.assert_allclose(series.values, [10.0] * 4 + [20.0] * 4)
        self.assertTrue((series.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all())

    def test_optimizer_with_real_solar_alignment_at_both_transitions(self):
        for day in ("2026-10-25", "2026-03-29"):
            for mode in ("15min", "hourly"):
                with self.subTest(day=day, mode=mode):
                    slots = local_day_slots(day)
                    result = self.optimize(slots, slots[0], weather_mode=mode)
                    pd.testing.assert_index_equal(
                        pd.DatetimeIndex(result["datetime_local"]), slots, check_names=False
                    )
                    self.assertTrue(np.isfinite(result.select_dtypes(include="number")).all().all())

    def test_optimizer_floors_non_quarter_price_timestamps(self):
        slots = local_day_slots("2026-10-25")
        result = self.optimize(slots + pd.Timedelta(minutes=7), slots[0])
        pd.testing.assert_index_equal(pd.DatetimeIndex(result["datetime_local"]), slots, check_names=False)


if __name__ == "__main__":
    unittest.main()