"""Offline / privacy environment flags (F17)."""
from __future__ import annotations

import os


def apply_offline_env() -> None:
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "0")
    os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")
