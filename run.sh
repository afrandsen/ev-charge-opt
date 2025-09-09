#!/bin/bash
echo "Script started..."

# Fetch the latest SOC from TeslaMate database inside Docker
SOC=$(docker exec teslamate-database-1 \
      psql -U teslamate teslamate -t -c \
      "SELECT battery_level FROM positions ORDER BY date DESC LIMIT 1;" | xargs)

echo "SOC fetched: $SOC"

source ~/repos/ev-charge-opt/venv/bin/activate

# Call ev-charge-opt.py with the fetched SOC
~/repos/ev-charge-opt/venv/bin/python ~/repos/ev-charge-opt/ev-charge-opt.py "$SOC"