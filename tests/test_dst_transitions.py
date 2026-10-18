import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ev_charge_opt import history, optimizer, pricing, runtime


class _SlotsPrepared(Exception):
    pass


class DstTransitionTests(unittest.TestCase):
    tz = "Europe/Copenhagen"

    def _clock(self, now):
        clock = Mock(wraps=pd)
        clock.Timestamp = Mock(wraps=pd.Timestamp)
        clock.Timestamp.now.return_value = pd.Timestamp(now).tz_convert("UTC")
        return clock

    def _prepare_slots(self, prices, now):
        cfg = {
            "tz": self.tz,
            "charger_min_a": 6,
            "charger_volt": 230,
            "phases": 3,
            "battery_kwh": 75,
            "soc_min_pct": 0.2,
            "soc_max_pct": 1.0,
            "systemtarif": 0.1,
            "nettarif_tso": 0.2,
            "elafgift": 0.3,
            "tillaeg": 0.4,
        }
        clock = self._clock(now)
        store = Mock()
        original = prices.copy(deep=True)
        with (
            patch.object(optimizer, "pd", clock),
            patch.object(optimizer, "_persist_price_history") as persist,
            patch.object(optimizer, "_build_solar_series", side_effect=_SlotsPrepared) as solar,
            self.assertRaises(_SlotsPrepared),
        ):
            optimizer.optimize_ev_charging(
                trips=pd.DataFrame(),
                prices=prices,
                cfg=cfg,
                runtime={"initial_soc_pct": 0.5},
                store=store,
                log=Mock(),
            )
        clock.Timestamp.now.assert_called_once_with(tz="UTC")
        persist.assert_called_once()
        pd.testing.assert_frame_equal(persist.call_args.args[0], original)
        pd.testing.assert_frame_equal(prices, original)
        self.assertEqual(store.mock_calls, [])
        return solar.call_args.args[0]

    def _prices(self, dates):
        return pd.DataFrame({"date": dates, "price": np.arange(len(dates)) + 100.0, "source": "EPEX"})

    def test_both_repeated_hours_round_to_distinct_slots(self):
        dates = pd.to_datetime(["2026-10-25T00:07:00Z", "2026-10-25T01:07:00Z"])
        slots = self._prepare_slots(self._prices(dates), "2026-10-24T21:00:00Z")

        self.assertEqual(
            slots["datetime_local"].map(lambda stamp: stamp.isoformat()).tolist(),
            ["2026-10-25T02:00:00+02:00", "2026-10-25T02:00:00+01:00"],
        )
        pd.testing.assert_index_equal(pd.DatetimeIndex(slots["datetime_utc"]), dates.floor("15min"), check_names=False)
        self.assertTrue(slots["datetime_local"].is_unique)
        self.assertEqual(slots["price"].tolist(), [100.0, 101.0])
        self.assertEqual(slots["price_source"].tolist(), ["EPEX", "EPEX"])

    def test_local_days_preserve_all_real_quarters(self):
        for day, expected_count in [("2026-10-25", 100), ("2026-03-29", 92), ("2026-10-18", 96)]:
            with self.subTest(day=day):
                start = pd.Timestamp(day, tz=self.tz)
                end = start + pd.DateOffset(days=1)
                dates = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="15min", inclusive="left")
                slots = self._prepare_slots(self._prices(dates), start.isoformat())

                self.assertEqual(len(slots), expected_count)
                self.assertTrue(slots["datetime_utc"].is_unique)
                self.assertTrue(slots["datetime_utc"].is_monotonic_increasing)
                self.assertTrue(slots["datetime_local"].is_unique)
                self.assertTrue(slots["datetime_utc"].diff().dropna().eq(pd.Timedelta(minutes=15)).all())
                self.assertEqual(slots["price"].tolist(), (np.arange(expected_count) + 100.0).tolist())
                self.assertTrue(slots["price_source"].eq("EPEX").all())
                repeated_hour = slots.loc[slots["hour_local"] == 2, "datetime_local"]
                if expected_count == 100:
                    self.assertEqual(len(repeated_hour), 8)
                    self.assertEqual(set(repeated_hour.dt.strftime("%z")), {"+0200", "+0100"})
                elif expected_count == 92:
                    self.assertTrue(repeated_hour.empty)

    def test_now_filter_preserves_correct_occurrence_of_repeated_hour(self):
        dates = pd.date_range("2026-10-25T00:00:00Z", periods=12, freq="15min")
        for now, expected_start in [("2026-10-25T00:22:00Z", 1), ("2026-10-25T01:22:00Z", 5)]:
            with self.subTest(now=now):
                slots = self._prepare_slots(self._prices(dates), now)
                pd.testing.assert_index_equal(
                    pd.DatetimeIndex(slots["datetime_utc"]), dates[expected_start:], check_names=False
                )

    def test_epex_offsets_preserve_both_repeated_hours(self):
        response = Mock()
        response.json.return_value = {
            "prices": [
                {"startsAt": "2026-10-25T02:00:00+02:00", "total": 100.0},
                {"startsAt": "2026-10-25T02:00:00+01:00", "total": 200.0},
            ]
        }
        with (
            patch.object(pricing.requests, "get", return_value=response) as get,
            patch.object(pricing, "fetch_eur_dkk_exchange_rate", return_value=7.48),
        ):
            prices = pricing.fetch_epex_forecast_dkk(log=Mock(), attempts=1)

        get.assert_called_once()
        self.assertFalse(get.call_args.kwargs["params"]["hourly"])
        self.assertEqual(prices["date"].tolist(), list(pd.to_datetime(["2026-10-25T00:00:00Z", "2026-10-25T01:00:00Z"])))
        np.testing.assert_allclose(prices["price"], [93.5, 187.0])
        self.assertEqual(prices["source"].tolist(), ["EPEX", "EPEX"])

    def test_solax_now_slot_keeps_correct_offset(self):
        for now in ["2026-10-25T00:07:00Z", "2026-10-25T01:07:00Z"]:
            with self.subTest(now=now):
                clock = self._clock(now)
                response = Mock()
                response.json.return_value = {"success": True, "result": {"acpower": 2000}}
                with (
                    patch.object(history, "pd", clock),
                    patch.object(history.requests, "post", return_value=response) as post,
                ):
                    result = history.fetch_solax_current_quarter_kwh("test-token", "test-inverter", self.tz, Mock(), attempts=1)
                self.assertIsNotNone(result)
                slot, energy = result
                self.assertEqual(slot.isoformat(), pd.Timestamp(now).floor("15min").tz_convert(self.tz).isoformat())
                self.assertEqual(energy, 0.5)
                clock.Timestamp.now.assert_called_once_with(tz="UTC")
                post.assert_called_once()

    def test_trip_shift_works_in_both_repeated_hours(self):
        trips = pd.DataFrame([{"day": "sunday", "away_start": "02:00", "away_end": "04:00"}])
        for now in ["2026-10-25T00:22:00Z", "2026-10-25T01:22:00Z"]:
            with self.subTest(now=now):
                clock = self._clock(now)
                with patch.object(runtime, "pd", clock):
                    shifted = runtime.shift_active_trip_windows(trips, True, self.tz, Mock())
                self.assertEqual(shifted.loc[0, "away_start"], "02:30")
                self.assertEqual(trips.loc[0, "away_start"], "02:00")
                clock.Timestamp.now.assert_called_once_with(tz="UTC")


if __name__ == "__main__":
    unittest.main()