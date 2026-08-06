import unittest

import numpy as np

from ev_charge_opt.optimizer import _build_soft_soc_bounds


class SoftSocBoundsTests(unittest.TestCase):
    def test_soft_min_starts_at_arrival_and_soft_max_stays_at_departure(self):
        horizon = 20
        cfg = {
            "battery_kwh": 75,
            "soc_max_pct": 1.0,
            "soft_soc_window_hours": 3,
            "soft_soc_abs_max_pct": 1.0,
            "soc_min_pct": 0.2,
            "soft_soc_min_window_hours": 1,
            "soft_soc_abs_min_pct": 0.1,
        }
        hard_soc_max_vec = np.full(horizon, 15.0)
        trip_departures = [(4, 10.0)]
        trip_events = [
            {
                "h_dep": 4,
                "h_end": 16,
                "trip_kwh": 10.0,
                "trip_max_kwh": None,
            }
        ]

        soc_lb_vec, soc_ub_vec, soft_min_relax_vec, soft_extra_cap_vec = _build_soft_soc_bounds(
            horizon=horizon,
            cfg=cfg,
            soc_min=15.0,
            hard_soc_max_vec=hard_soc_max_vec,
            trip_departures=trip_departures,
            trip_events=trip_events,
        )

        self.assertTrue(np.allclose(soft_extra_cap_vec[:4], 10.0))
        self.assertEqual(soft_extra_cap_vec[4], 0.0)

        self.assertTrue(np.allclose(soft_min_relax_vec[:16], 0.0))
        self.assertTrue(np.allclose(soft_min_relax_vec[16:], 10.0))

        self.assertEqual(soc_lb_vec[15], 15.0)
        self.assertEqual(soc_lb_vec[16], 7.5)
        self.assertTrue(np.all(soc_ub_vec >= hard_soc_max_vec))


if __name__ == "__main__":
    unittest.main()