"""Scraper for Singapore TOTO historical results."""

import asyncio
import re
from datetime import datetime
from typing import Optional
import logging

import httpx
from bs4 import BeautifulSoup

from app.config import TOTO_RESULTS_URL

logger = logging.getLogger(__name__)


class TOTOScraper:
    """Scraper for Singapore TOTO results from multiple sources."""

    # Primary source: Lottolyzer (has extensive history)
    LOTTOLYZER_BASE = "https://en.lottolyzer.com"
    LOTTOLYZER_HISTORY = f"{LOTTOLYZER_BASE}/history/singapore/toto"
    LOTTOLYZER_PER_PAGE = 50

    # Backup source: Official Singapore Pools
    SINGAPORE_POOLS_BASE = "https://www.singaporepools.com.sg"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            try:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                logger.error(f"Error fetching {url}: {e}")
                return None

    def parse_lottolyzer_page(self, html: str) -> list[dict]:
        """Parse a page of results from Lottolyzer."""
        soup = BeautifulSoup(html, "lxml")
        results = []

        # Find the main data table
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 4:
                    try:
                        # Extract draw number
                        draw_text = cells[0].get_text(strip=True)
                        draw_match = re.search(r'(\d+)', draw_text)
                        if not draw_match:
                            continue
                        draw_number = int(draw_match.group(1))

                        # Extract date
                        date_text = cells[1].get_text(strip=True)
                        draw_date = self._parse_date(date_text)
                        if not draw_date:
                            continue

                        # Extract winning numbers
                        numbers_text = cells[2].get_text(strip=True)
                        numbers = self._parse_numbers(numbers_text)
                        if len(numbers) != 6:
                            continue

                        # Extract additional number
                        additional_text = cells[3].get_text(strip=True)
                        additional_match = re.search(r'(\d+)', additional_text)
                        if not additional_match:
                            continue
                        additional_number = int(additional_match.group(1))

                        results.append({
                            "draw_number": draw_number,
                            "draw_date": draw_date,
                            "winning_numbers": sorted(numbers),
                            "additional_number": additional_number,
                        })
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing row: {e}")
                        continue

        return results

    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from various formats."""
        # Common formats from different sources
        formats = [
            "%Y-%m-%d",
            "%d %b %Y",
            "%d %B %Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
        ]

        # Clean up the date text
        date_text = date_text.strip()

        for fmt in formats:
            try:
                return datetime.strptime(date_text, fmt)
            except ValueError:
                continue

        # Try to extract date pattern
        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_text)
        if date_match:
            try:
                return datetime(int(date_match.group(1)),
                               int(date_match.group(2)),
                               int(date_match.group(3)))
            except ValueError:
                pass

        return None

    def _parse_numbers(self, numbers_text: str) -> list[int]:
        """Parse winning numbers from text."""
        # Find all numbers in the text
        numbers = re.findall(r'\d+', numbers_text)
        return [int(n) for n in numbers if 1 <= int(n) <= 49][:6]

    async def scrape_all_history(
        self,
        max_pages: Optional[int] = None,
        progress_callback=None
    ) -> list[dict]:
        """Scrape all historical TOTO results."""
        all_results = []
        page = 1

        # First, get total pages from page 1
        url = f"{self.LOTTOLYZER_HISTORY}/page/1/per-page/{self.LOTTOLYZER_PER_PAGE}/summary-view"
        html = await self.fetch_page(url)

        if not html:
            logger.error("Failed to fetch first page")
            return []

        # Try to extract total pages
        soup = BeautifulSoup(html, "lxml")
        total_pages = 1

        # Look for pagination info
        page_text = soup.get_text()
        page_match = re.search(r'(\d+)\s*/\s*(\d+)\s*pages?', page_text, re.IGNORECASE)
        if page_match:
            total_pages = int(page_match.group(2))
        else:
            # Fallback: estimate based on known data
            total_pages = 37  # Approximate based on research

        if max_pages:
            total_pages = min(total_pages, max_pages)

        logger.info(f"Scraping {total_pages} pages of TOTO history")

        # Parse first page
        results = self.parse_lottolyzer_page(html)
        all_results.extend(results)

        if progress_callback:
            progress_callback(1, total_pages, len(results))

        # Scrape remaining pages
        for page in range(2, total_pages + 1):
            url = f"{self.LOTTOLYZER_HISTORY}/page/{page}/per-page/{self.LOTTOLYZER_PER_PAGE}/summary-view"
            html = await self.fetch_page(url)

            if html:
                results = self.parse_lottolyzer_page(html)
                all_results.extend(results)

                if progress_callback:
                    progress_callback(page, total_pages, len(results))
            else:
                logger.warning(f"Failed to fetch page {page}")

            # Be respectful to the server
            await asyncio.sleep(0.5)

        # Remove duplicates based on draw_number
        seen = set()
        unique_results = []
        for result in all_results:
            if result["draw_number"] not in seen:
                seen.add(result["draw_number"])
                unique_results.append(result)

        # Sort by draw number
        unique_results.sort(key=lambda x: x["draw_number"])

        logger.info(f"Scraped {len(unique_results)} unique draws")
        return unique_results

    async def scrape_latest(self, count: int = 10) -> list[dict]:
        """Scrape only the most recent results."""
        url = f"{self.LOTTOLYZER_HISTORY}/page/1/per-page/{count}/summary-view"
        html = await self.fetch_page(url)

        if not html:
            return []

        results = self.parse_lottolyzer_page(html)
        return sorted(results, key=lambda x: x["draw_number"], reverse=True)[:count]


async def main():
    """Test the scraper."""
    scraper = TOTOScraper()

    def progress(page, total, count):
        print(f"Page {page}/{total}: {count} results")

    # Test with just first 2 pages
    results = await scraper.scrape_all_history(max_pages=2, progress_callback=progress)

    print(f"\nTotal results: {len(results)}")
    if results:
        print("\nLatest draw:")
        latest = results[-1]
        print(f"  Draw #{latest['draw_number']} on {latest['draw_date']}")
        print(f"  Numbers: {latest['winning_numbers']} + {latest['additional_number']}")


if __name__ == "__main__":
    asyncio.run(main())
