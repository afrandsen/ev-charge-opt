#!/bin/bash

# Load environment variables from .env.local (LAT, LON, etc.)
source ~/repos/ev-charge-opt/.env.local

# Check if car is within 2000 m of home
IS_HOME=$(docker exec teslamate-database-1 \
          psql -U teslamate teslamate -t -c \
          "SELECT (earth_distance(ll_to_earth(latitude, longitude), ll_to_earth(${LAT}, ${LON})) <= 2000) AS is_home FROM positions WHERE car_id = 1 ORDER BY date DESC LIMIT 1;" | xargs)

if [ "$IS_HOME" = "t" ]; then
    echo "⚡ Car is home within 2000 m → running alarm"

    # Fetch if car is charging from TeslaMate database inside Docker
    CHARGING=$(docker exec teslamate-database-1 \
                   psql -U teslamate teslamate -t -c \
                   "SELECT (SELECT COUNT(*) FROM charging_processes WHERE car_id = 1 AND end_date IS NULL) > 0;" | xargs)
    
    if [ "$CHARGING" = "f" ]; then
        echo "🔔 Car is not charging"

        source ~/repos/ev-charge-opt/venv/bin/activate

        # Call alarm.py with the fetched charging state
        python ~/repos/ev-charge-opt/alarm.py "$CHARGING"
    
    else
        echo "✅ Car is charging → skipping alarm"
    fi
else
    echo "🚗 Car is not home → skipping alarm"
fi