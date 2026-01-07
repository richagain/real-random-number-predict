"""Scheduled tasks for automatic data updates."""

import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.scraper.data_loader import DataLoader
from app.database.db import get_sync_session

logger = logging.getLogger(__name__)

# Singapore timezone
SGT = pytz.timezone("Asia/Singapore")

# Scheduler instance
scheduler = AsyncIOScheduler(timezone=SGT)


async def scheduled_scrape():
    """Scrape latest TOTO results after each draw."""
    logger.info("Running scheduled scrape for new TOTO results...")

    try:
        session = get_sync_session()
        loader = DataLoader(session)

        # Only scrape the first page (latest 50 results)
        result = await loader.populate_from_scraper(max_pages=1)

        if result.get("success"):
            inserted = result.get("inserted", 0)
            total = result.get("total_in_db", 0)
            logger.info(f"Scheduled scrape complete: {inserted} new draws, {total} total in database")
        else:
            logger.warning(f"Scheduled scrape failed: {result.get('message', 'Unknown error')}")

        session.close()

    except Exception as e:
        logger.error(f"Error in scheduled scrape: {e}")


def setup_scheduler():
    """Configure and start the scheduler.

    TOTO draws occur:
    - Monday at 6:30pm SGT
    - Thursday at 6:30pm SGT

    Results are usually available 15-30 minutes after the draw.
    We schedule scraping at 7:00pm SGT on draw days.
    """
    # Schedule for Monday at 7:00pm SGT
    scheduler.add_job(
        scheduled_scrape,
        CronTrigger(day_of_week="mon", hour=19, minute=0, timezone=SGT),
        id="monday_scrape",
        name="Monday TOTO Scrape",
        replace_existing=True,
    )

    # Schedule for Thursday at 7:00pm SGT
    scheduler.add_job(
        scheduled_scrape,
        CronTrigger(day_of_week="thu", hour=19, minute=0, timezone=SGT),
        id="thursday_scrape",
        name="Thursday TOTO Scrape",
        replace_existing=True,
    )

    # Also run a daily check at 8:00am SGT to catch any missed updates
    scheduler.add_job(
        scheduled_scrape,
        CronTrigger(hour=8, minute=0, timezone=SGT),
        id="daily_check",
        name="Daily Data Check",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started with TOTO scraping jobs")

    # Log next run times
    for job in scheduler.get_jobs():
        logger.info(f"  {job.name}: next run at {job.next_run_time}")


def shutdown_scheduler():
    """Shutdown the scheduler gracefully."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shutdown complete")


def get_scheduler_status() -> dict:
    """Get current scheduler status and job info."""
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })

    return {
        "running": scheduler.running,
        "timezone": str(SGT),
        "jobs": jobs,
    }
