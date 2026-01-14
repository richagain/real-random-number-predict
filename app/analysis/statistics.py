"""Statistical analysis for TOTO number frequency and patterns."""

from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional
import numpy as np
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session

from app.database.models import Draw, NumberStatistics
from app.config import (
    TOTO_MIN_NUMBER, TOTO_MAX_NUMBER,
    HOT_NUMBERS_COUNT, COLD_NUMBERS_COUNT,
    DEFAULT_LOOKBACK_DRAWS
)


class NumberAnalyzer:
    """Analyze TOTO number frequencies and statistics."""

    def __init__(self, session: Session, max_draw_number: Optional[int] = None):
        self.session = session
        self.max_draw_number = max_draw_number
        self.all_numbers = list(range(TOTO_MIN_NUMBER, TOTO_MAX_NUMBER + 1))

    def get_all_draws(self, limit: Optional[int] = None) -> list[Draw]:
        """Get all draws from database, optionally limited.

        If max_draw_number is set, only returns draws before that draw number.
        This is used for backtesting to simulate historical predictions.
        """
        query = select(Draw).order_by(desc(Draw.draw_number))
        if self.max_draw_number:
            query = query.where(Draw.draw_number < self.max_draw_number)
        if limit:
            query = query.limit(limit)
        result = self.session.execute(query).scalars().all()
        return list(result)

    def get_total_draws(self) -> int:
        """Get total number of draws in database.

        If max_draw_number is set, only counts draws before that draw number.
        """
        query = select(func.count(Draw.id))
        if self.max_draw_number:
            query = query.where(Draw.draw_number < self.max_draw_number)
        return self.session.execute(query).scalar() or 0

    def calculate_frequency(
        self,
        draws: Optional[list[Draw]] = None,
        lookback: Optional[int] = None
    ) -> dict[int, dict]:
        """Calculate frequency statistics for each number."""
        if draws is None:
            draws = self.get_all_draws(limit=lookback)

        total_draws = len(draws)
        if total_draws == 0:
            return {}

        # Initialize counters
        main_counts = Counter()
        additional_counts = Counter()
        position_counts = defaultdict(Counter)  # position -> {number: count}

        # Latest appearance tracking
        last_seen = {}  # number -> (draw_number, date)

        for draw in draws:
            winning_numbers = draw.winning_numbers

            # Count main numbers
            for i, num in enumerate(winning_numbers):
                main_counts[num] += 1
                position_counts[i + 1][num] += 1

                if num not in last_seen or draw.draw_number > last_seen[num][0]:
                    last_seen[num] = (draw.draw_number, draw.draw_date)

            # Count additional number
            additional_counts[draw.additional_number] += 1
            if draw.additional_number not in last_seen or draw.draw_number > last_seen[draw.additional_number][0]:
                last_seen[draw.additional_number] = (draw.draw_number, draw.draw_date)

        # Get latest draw number for gap calculation
        latest_draw_num = max(d.draw_number for d in draws) if draws else 0

        # Build frequency dict for all numbers
        frequency = {}
        for num in self.all_numbers:
            main_count = main_counts.get(num, 0)
            additional_count = additional_counts.get(num, 0)
            total_count = main_count + additional_count

            # Calculate gap (draws since last appearance)
            if num in last_seen:
                last_draw_num, last_date = last_seen[num]
                gap = latest_draw_num - last_draw_num
            else:
                gap = total_draws  # Never appeared
                last_date = None

            frequency[num] = {
                "number": num,
                "total_appearances": total_count,
                "appearances_as_main": main_count,
                "appearances_as_additional": additional_count,
                "frequency_percentage": (total_count / (total_draws * 7)) * 100 if total_draws > 0 else 0,
                "main_frequency_percentage": (main_count / (total_draws * 6)) * 100 if total_draws > 0 else 0,
                "last_drawn_date": last_date,
                "draws_since_last": gap,
                "position_counts": {
                    pos: position_counts[pos].get(num, 0) for pos in range(1, 7)
                },
            }

        return frequency

    def get_hot_numbers(
        self,
        count: int = HOT_NUMBERS_COUNT,
        lookback: Optional[int] = DEFAULT_LOOKBACK_DRAWS
    ) -> list[int]:
        """Get the most frequently drawn numbers."""
        frequency = self.calculate_frequency(lookback=lookback)
        sorted_nums = sorted(
            frequency.items(),
            key=lambda x: x[1]["total_appearances"],
            reverse=True
        )
        return [num for num, _ in sorted_nums[:count]]

    def get_cold_numbers(
        self,
        count: int = COLD_NUMBERS_COUNT,
        lookback: Optional[int] = DEFAULT_LOOKBACK_DRAWS
    ) -> list[int]:
        """Get the least frequently drawn numbers (overdue numbers)."""
        frequency = self.calculate_frequency(lookback=lookback)
        sorted_nums = sorted(
            frequency.items(),
            key=lambda x: x[1]["total_appearances"]
        )
        return [num for num, _ in sorted_nums[:count]]

    def get_overdue_numbers(
        self,
        count: int = 10,
        lookback: Optional[int] = None
    ) -> list[int]:
        """Get numbers with longest gap since last appearance."""
        frequency = self.calculate_frequency(lookback=lookback)
        sorted_nums = sorted(
            frequency.items(),
            key=lambda x: x[1]["draws_since_last"],
            reverse=True
        )
        return [num for num, _ in sorted_nums[:count]]

    def calculate_sum_statistics(
        self,
        lookback: Optional[int] = None
    ) -> dict:
        """Calculate statistics for sum of winning numbers."""
        draws = self.get_all_draws(limit=lookback)

        if not draws:
            return {}

        sums = [sum(d.winning_numbers) for d in draws]

        return {
            "min": min(sums),
            "max": max(sums),
            "mean": np.mean(sums),
            "median": np.median(sums),
            "std": np.std(sums),
            "distribution": dict(Counter(sums)),
            "percentiles": {
                "10": np.percentile(sums, 10),
                "25": np.percentile(sums, 25),
                "50": np.percentile(sums, 50),
                "75": np.percentile(sums, 75),
                "90": np.percentile(sums, 90),
            }
        }

    def calculate_odd_even_distribution(
        self,
        lookback: Optional[int] = None
    ) -> dict:
        """Calculate odd/even number distribution."""
        draws = self.get_all_draws(limit=lookback)

        if not draws:
            return {}

        distributions = []
        for draw in draws:
            odd_count = sum(1 for n in draw.winning_numbers if n % 2 == 1)
            even_count = 6 - odd_count
            distributions.append((odd_count, even_count))

        counter = Counter(distributions)

        return {
            "distribution": {f"{o}O-{e}E": count for (o, e), count in counter.items()},
            "most_common": counter.most_common(3),
            "average_odd": np.mean([d[0] for d in distributions]),
            "average_even": np.mean([d[1] for d in distributions]),
        }

    def calculate_high_low_distribution(
        self,
        lookback: Optional[int] = None,
        midpoint: int = 25
    ) -> dict:
        """Calculate high/low number distribution."""
        draws = self.get_all_draws(limit=lookback)

        if not draws:
            return {}

        distributions = []
        for draw in draws:
            low_count = sum(1 for n in draw.winning_numbers if n <= midpoint)
            high_count = 6 - low_count
            distributions.append((low_count, high_count))

        counter = Counter(distributions)

        return {
            "distribution": {f"{l}L-{h}H": count for (l, h), count in counter.items()},
            "most_common": counter.most_common(3),
            "average_low": np.mean([d[0] for d in distributions]),
            "average_high": np.mean([d[1] for d in distributions]),
            "midpoint": midpoint,
        }

    def get_comprehensive_statistics(
        self,
        lookback: Optional[int] = DEFAULT_LOOKBACK_DRAWS
    ) -> dict:
        """Get comprehensive statistics for dashboard."""
        draws = self.get_all_draws(limit=lookback)
        total_draws = len(draws)

        if total_draws == 0:
            return {"error": "No draws in database"}

        frequency = self.calculate_frequency(draws=draws)

        # Get date range
        dates = [d.draw_date for d in draws]
        date_range = {
            "earliest": min(dates),
            "latest": max(dates),
        }

        return {
            "total_draws": total_draws,
            "date_range": date_range,
            "number_frequencies": list(frequency.values()),
            "hot_numbers": self.get_hot_numbers(lookback=lookback),
            "cold_numbers": self.get_cold_numbers(lookback=lookback),
            "overdue_numbers": self.get_overdue_numbers(lookback=lookback),
            "sum_statistics": self.calculate_sum_statistics(lookback=lookback),
            "odd_even_distribution": self.calculate_odd_even_distribution(lookback=lookback),
            "high_low_distribution": self.calculate_high_low_distribution(lookback=lookback),
        }

    def update_number_statistics_table(self) -> None:
        """Update the NumberStatistics table with current data."""
        frequency = self.calculate_frequency()
        total_draws = self.get_total_draws()

        for num, stats in frequency.items():
            # Get or create statistics record
            existing = self.session.execute(
                select(NumberStatistics).where(NumberStatistics.number == num)
            ).scalar_one_or_none()

            if existing:
                record = existing
            else:
                record = NumberStatistics(number=num)
                self.session.add(record)

            # Update fields
            record.total_appearances = stats["total_appearances"]
            record.appearances_as_main = stats["appearances_as_main"]
            record.appearances_as_additional = stats["appearances_as_additional"]
            record.last_drawn_date = stats["last_drawn_date"]
            record.draws_since_last = stats["draws_since_last"]

            # Position counts
            pos = stats["position_counts"]
            record.position1_count = pos.get(1, 0)
            record.position2_count = pos.get(2, 0)
            record.position3_count = pos.get(3, 0)
            record.position4_count = pos.get(4, 0)
            record.position5_count = pos.get(5, 0)
            record.position6_count = pos.get(6, 0)

            # Calculate scores
            if total_draws > 0:
                expected_freq = total_draws * 6 / 49
                record.hot_score = stats["appearances_as_main"] / expected_freq if expected_freq > 0 else 0
                record.cold_score = stats["draws_since_last"] / 10  # Normalize gap

        self.session.commit()
