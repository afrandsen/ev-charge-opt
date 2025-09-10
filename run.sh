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

      source ~/repos/ev-charge-opt/venv/bin/activate

      # Call ev-charge-opt.py with the fetched SOC
      python ~/repos/ev-charge-opt/ev-charge-opt.py "$SOC" "$IS_HOME"
else
    echo "🚗 Car is not home → skipping ev-charge-opt"
fi