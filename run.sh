#!/bin/bash

# Load environment variables from .env.local (LAT, LON, etc.)
source ~/repos/ev-charge-opt/.env.local

# Check if car is within 2000 m of home
IS_HOME=$(docker exec teslamate-database-1 \
          psql -U teslamate teslamate -t -c \
          "SELECT (earth_distance(ll_to_earth(latitude, longitude), ll_to_earth(${LAT}, ${LON})) <= 2000) AS is_home FROM positions WHERE car_id = 1 ORDER BY date DESC LIMIT 1;" | xargs)

if [ "$IS_HOME" = "t" ]; then
    echo "⚡ Car is home within 2000 m → fetching SOC and running ev-charge-opt"

      # Fetch the latest SOC from TeslaMate database inside Docker
      SOC=$(docker exec teslamate-database-1 \
            psql -U teslamate teslamate -t -c \
            "SELECT battery_level FROM positions ORDER BY date DESC LIMIT 1;" | xargs)

      # Fetch charging efficiency from TeslaMate database inside Docker
      CHARGE_EFF=$(docker exec teslamate-database-1 \
                   psql -U teslamate teslamate -t -c \
                   "SELECT SUM(charge_energy_added) / SUM(GREATEST(charge_energy_added, charge_energy_used)) AS charging_efficiency_percent FROM charging_processes WHERE car_id = 1 AND charge_energy_added > 0.01 LIMIT 1;" | xargs)

      # Fetch efficiency from TeslaMate database inside Docker
      EFF_KWH_PER_KM=$(docker exec teslamate-database-1 \
                       psql -U teslamate teslamate -t -c \
                       "SELECT AVG((start_rated_range_km - end_rated_range_km) * car.efficiency / NULLIF(distance,0)) AS wh_per_km_7day_avg FROM drives JOIN cars car ON car.id = drives.car_id WHERE start_date >= NOW() - INTERVAL '7 days' AND distance > 10 AND start_rated_range_km IS NOT NULL AND end_rated_range_km IS NOT NULL AND (start_rated_range_km - end_rated_range_km) > 0 LIMIT 1;" | xargs)

      source ~/repos/ev-charge-opt/venv/bin/activate

      # Call ev-charge-opt.py with the fetched SOC
      python ~/repos/ev-charge-opt/ev-charge-opt.py "$SOC" "$IS_HOME" "$EFF_KWH_PER_KM" "$CHARGE_EFF"
else
    echo "🚗 Car is not home → skipping ev-charge-opt"
fi