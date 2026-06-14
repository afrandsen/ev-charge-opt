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

charge_buckets AS (
    SELECT
        cd.charging_process_id,
        date_trunc('hour', cd.date)
            + floor(extract(minute FROM cd.date) / 15) * interval '15 minutes' AS bucket,

        sum(cd.delta_kwh)
            * (
                cp.charge_energy_used::numeric
                / nullif(cp.charge_energy_added, 0)
            ) AS charge_kwh

    FROM charge_deltas cd
    JOIN charging_processes cp
        ON cp.id = cd.charging_process_id
    WHERE cd.delta_kwh IS NOT NULL
    GROUP BY
        cd.charging_process_id,
        bucket,
        cp.charge_energy_added,
        cp.charge_energy_used
)

SELECT
    cb.charging_process_id,

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

JOIN history.ev_charge_opt_nordpool_spot_price_15m p
    ON p.slot_local = cb.bucket

LEFT JOIN history.ev_charge_opt_solax_ac_power_15m s
    ON s.slot_local = cb.bucket

GROUP BY cb.charging_process_id

ORDER BY cb.charging_process_id;