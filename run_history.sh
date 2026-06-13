#!/bin/bash

source ~/repos/ev-charge-opt/.env.local
source ~/repos/ev-charge-opt/venv/bin/activate

python ~/repos/ev-charge-opt/history_sync.py
