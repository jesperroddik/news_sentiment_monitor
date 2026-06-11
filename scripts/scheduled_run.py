"""Launcher for the Windows scheduled task ("DanishNewsSentimentPipeline").

Runs the full pipeline once (fetch -> store -> sentiment -> IPTC classify) and
appends all output to ``logs/pipeline.log``. Designed to be run by ``pythonw.exe``
so there is no console window — under ``pythonw`` ``sys.stdout``/``sys.stderr`` are
otherwise ``None`` and any ``print()`` would raise, so we point them at the log
file before doing anything else.

    pythonw scripts/scheduled_run.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LOG_PATH = ROOT / "logs" / "pipeline.log"


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as logf:
        sys.stdout = logf
        sys.stderr = logf
        print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} — scheduled pipeline run =====")
        try:
            import fetch

            fetch.run_pipeline()
        except Exception:
            import traceback

            traceback.print_exc()
        logf.flush()


if __name__ == "__main__":
    main()
