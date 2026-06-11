"""Launcher for the Windows scheduled task ("DanishNewsSentimentPipeline").

Runs the full pipeline once (fetch -> store -> sentiment -> IPTC classify) and
appends all output to ``logs/pipeline.log``. Designed to be run by ``pythonw.exe``
so there is no console window — under ``pythonw`` ``sys.stdout``/``sys.stderr`` are
otherwise ``None`` and any ``print()`` would raise, so we point them at the log
file before doing anything else.

On failure it (1) logs the traceback + a ``RUN FAILED`` footer, (2) pops a Windows
toast notification, and (3) exits non-zero so Task Scheduler's History flags the
run as failed. On success it logs a ``RUN OK`` footer and exits 0.

    pythonw scripts/scheduled_run.py
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LOG_PATH = ROOT / "logs" / "pipeline.log"


def _notify_failure(detail: str) -> None:
    """Pop a desktop toast on failure. Never let a notification problem mask the
    real error, and never let it change the exit code."""
    try:
        from winotify import Notification

        toast = Notification(
            app_id="Danish News Pipeline",
            title="Pipeline run failed",
            msg=f"{detail}\nSee logs/pipeline.log",
            launch=str(LOG_PATH),  # clicking the toast opens the log
        )
        toast.show()
    except Exception:
        traceback.print_exc()


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as logf:
        sys.stdout = logf
        sys.stderr = logf
        print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} — scheduled pipeline run =====")
        try:
            import fetch

            inserted = fetch.run_pipeline()
        except Exception as exc:
            traceback.print_exc()
            print(f"RUN FAILED: {exc!r}")
            logf.flush()
            _notify_failure(repr(exc))
            return 1
        print(f"RUN OK: {inserted} new article(s)")
        logf.flush()
        return 0


if __name__ == "__main__":
    sys.exit(main())
