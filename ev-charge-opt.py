#!/usr/bin/env python3
"""Backward-compatible wrapper for the modular EV charge optimizer."""

from ev_charge_opt.main import main


if __name__ == "__main__":
    raise SystemExit(main())
