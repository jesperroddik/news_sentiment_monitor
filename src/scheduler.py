"""Scheduler — runs the full pipeline every 2 hours.

    python src/scheduler.py

Triggers fetch -> sentiment -> IPTC classify on an interval,
plus one immediate run on startup. Ctrl-C to stop.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler

import db
import fetch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("scheduler")

INTERVAL_HOURS = 2
RETENTION_HOUR_UTC = 3  # daily retention sweep, off-cycle from the 2h pipeline


def _job() -> None:
    log.info("Pipeline run starting")
    try:
        fetch.run_pipeline()
        log.info("Pipeline run finished")
    except Exception:
        # Never let a single failure kill the scheduler.
        log.exception("Pipeline run failed")


def _retention_job() -> None:
    log.info("Retention sweep starting")
    try:
        deleted = db.purge_old_articles()
        log.info("Retention sweep finished — %d article(s) deleted", deleted)
    except Exception:
        # A failed sweep must not kill the scheduler; it retries next day.
        log.exception("Retention sweep failed")


def main() -> None:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        _job,
        trigger="interval",
        hours=INTERVAL_HOURS,
        id="news_pipeline",
    )
    scheduler.add_job(
        _retention_job,
        trigger="cron",
        hour=RETENTION_HOUR_UTC,
        id="retention_sweep",
    )
    log.info("Running pipeline once on startup, then every %d hours", INTERVAL_HOURS)
    log.info("Retention sweep scheduled daily at %02d:00 UTC", RETENTION_HOUR_UTC)
    _job()  # immediate first run
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")


if __name__ == "__main__":
    main()
