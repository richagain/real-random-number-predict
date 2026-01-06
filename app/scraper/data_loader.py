"""Data loader for populating database from scraped results."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.database.db import get_sync_session, init_db_sync
from app.database.models import Draw
from app.scraper.singapore_pools import TOTOScraper

logger = logging.getLogger(__name__)


class DataLoader:
    """Load scraped TOTO data into database."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_sync_session()
        self.scraper = TOTOScraper()

    def get_latest_draw_number(self) -> int:
        """Get the latest draw number in database."""
        result = self.session.execute(
            select(func.max(Draw.draw_number))
        ).scalar()
        return result or 0

    def get_total_draws(self) -> int:
        """Get total number of draws in database."""
        return self.session.execute(
            select(func.count(Draw.id))
        ).scalar() or 0

    def insert_draw(self, draw_data: dict) -> Optional[Draw]:
        """Insert a single draw into database."""
        # Check if draw already exists
        existing = self.session.execute(
            select(Draw).where(Draw.draw_number == draw_data["draw_number"])
        ).scalar_one_or_none()

        if existing:
            return None

        numbers = sorted(draw_data["winning_numbers"])

        draw = Draw(
            draw_number=draw_data["draw_number"],
            draw_date=draw_data["draw_date"],
            num1=numbers[0],
            num2=numbers[1],
            num3=numbers[2],
            num4=numbers[3],
            num5=numbers[4],
            num6=numbers[5],
            additional_number=draw_data["additional_number"],
            group1_prize=draw_data.get("group1_prize"),
            group1_winners=draw_data.get("group1_winners"),
        )

        self.session.add(draw)
        return draw

    def bulk_insert(self, draws_data: list[dict], batch_size: int = 100) -> int:
        """Bulk insert draws into database."""
        inserted = 0

        for i, draw_data in enumerate(draws_data):
            result = self.insert_draw(draw_data)
            if result:
                inserted += 1

            # Commit in batches
            if (i + 1) % batch_size == 0:
                self.session.commit()
                logger.info(f"Committed batch: {i + 1} processed, {inserted} inserted")

        # Final commit
        self.session.commit()
        return inserted

    async def populate_from_scraper(
        self,
        max_pages: Optional[int] = None,
        progress_callback=None
    ) -> dict:
        """Scrape and populate database with historical data."""
        start_time = datetime.now()

        logger.info("Starting data population...")

        # Get current state
        existing_count = self.get_total_draws()
        latest_draw = self.get_latest_draw_number()
        logger.info(f"Current database: {existing_count} draws, latest #{latest_draw}")

        # Scrape data
        def scrape_progress(page, total, count):
            if progress_callback:
                progress_callback(f"Scraping page {page}/{total}...")
            logger.info(f"Scraping page {page}/{total}: {count} results")

        results = await self.scraper.scrape_all_history(
            max_pages=max_pages,
            progress_callback=scrape_progress
        )

        if not results:
            return {
                "success": False,
                "message": "Failed to scrape data",
                "inserted": 0,
            }

        logger.info(f"Scraped {len(results)} draws, inserting into database...")

        # Insert into database
        inserted = self.bulk_insert(results)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "scraped": len(results),
            "inserted": inserted,
            "total_in_db": self.get_total_draws(),
            "elapsed_seconds": elapsed,
        }


async def populate_database(max_pages: Optional[int] = None):
    """Main function to populate database."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize database
    init_db_sync()

    # Create loader and populate
    loader = DataLoader()
    result = await loader.populate_from_scraper(max_pages=max_pages)

    print("\n" + "=" * 50)
    print("Data Population Complete")
    print("=" * 50)
    print(f"Scraped: {result.get('scraped', 0)} draws")
    print(f"Inserted: {result.get('inserted', 0)} new draws")
    print(f"Total in DB: {result.get('total_in_db', 0)} draws")
    print(f"Time: {result.get('elapsed_seconds', 0):.1f} seconds")

    return result


if __name__ == "__main__":
    # Run with limited pages for testing
    asyncio.run(populate_database(max_pages=3))
