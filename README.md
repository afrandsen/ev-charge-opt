# ev-charge-opt
Charge Plan with Global Cost Minimiser

## Project Structure

- `ev_charge_opt/`
	- `main.py`: main optimizer app entrypoint
	- `optimizer.py`: optimization model + weather/solar preprocessing
	- `pricing.py`: Nordpool and forecast data fetch/merge
	- `history.py`: Postgres history persistence + Solax realtime fetch
	- `runtime.py`: runtime/config parsing and trip window adjustments
	- `state.py`: persisted charging state for alerting
	- `notifications.py`: email notification helper
	- `logging_utils.py`: simple run logger
- `ev-charge-opt.py`: backward-compatible wrapper that runs `ev_charge_opt.main`
- `history_sync.py`: independent history sync (Solax + spot/total prices + charging session summaries)
- `solar_sync.py`: Solax-only sync (updates quarter-slot solar estimate and refreshes charging session summaries)
- `get_trips.py`: fetches and parses calendar trips into `TRIPS` env var
- `run.sh`: orchestration script; always runs history sync, runs optimizer only if EV is home
- `run_history.sh`: run only the independent history sync
- `run_solar.sh`: run only the Solax solar sync (intended for 5-minute cadence)
- `alarm.py`, `alarm.sh`: not-charging alarm flow

## Usage
Clone the repository, `pip install -r requirements.txt` in a dedicated Python environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Main Flow

- `./run.sh`
	- If car is home: runs trip refresh + optimizer (optimizer also persists Solax and spot/total price history).
	- If car is away: skips optimizer and runs independent history sync.

### History-Only Flow

- `./run_history.sh`
	- Runs Solax + price history sync and charging session summary sync without running the optimizer.

### Solar-Only Flow (Optional 5-Minute Sampling)

- `./run_solar.sh`
	- Fetches Solax realtime data and updates a running average estimate for the current 15-minute slot.
	- Recomputes charging session summaries using the latest solar estimate.