"""Pattern detection for TOTO number sequences."""

from collections import Counter, defaultdict
from itertools import combinations
from typing import Optional
import numpy as np
from sqlalchemy.orm import Session

from app.database.models import Draw
from app.analysis.statistics import NumberAnalyzer


class PatternAnalyzer:
    """Analyze patterns in TOTO draws."""

    def __init__(self, session: Session):
        self.session = session
        self.number_analyzer = NumberAnalyzer(session)

    def find_consecutive_patterns(
        self,
        lookback: Optional[int] = None
    ) -> dict:
        """Find patterns of consecutive numbers in draws."""
        draws = self.number_analyzer.get_all_draws(limit=lookback)

        if not draws:
            return {}

        # Track consecutive pairs, triples, etc.
        consecutive_pairs = Counter()
        consecutive_triples = Counter()
        draws_with_consecutive = 0

        for draw in draws:
            numbers = draw.winning_numbers
            has_consecutive = False

            # Check for consecutive pairs
            for i in range(len(numbers) - 1):
                if numbers[i + 1] - numbers[i] == 1:
                    pair = (numbers[i], numbers[i + 1])
                    consecutive_pairs[pair] += 1
                    has_consecutive = True

                    # Check for triple
                    if i < len(numbers) - 2 and numbers[i + 2] - numbers[i + 1] == 1:
                        triple = (numbers[i], numbers[i + 1], numbers[i + 2])
                        consecutive_triples[triple] += 1

            if has_consecutive:
                draws_with_consecutive += 1

        total_draws = len(draws)
        return {
            "consecutive_pairs": dict(consecutive_pairs.most_common(20)),
            "consecutive_triples": dict(consecutive_triples.most_common(10)),
            "draws_with_consecutive": draws_with_consecutive,
            "consecutive_percentage": (draws_with_consecutive / total_draws * 100) if total_draws > 0 else 0,
            "total_draws_analyzed": total_draws,
        }

    def find_number_pairs(
        self,
        lookback: Optional[int] = None,
        min_occurrences: int = 5
    ) -> dict:
        """Find frequently occurring number pairs."""
        draws = self.number_analyzer.get_all_draws(limit=lookback)

        if not draws:
            return {}

        pair_counts = Counter()

        for draw in draws:
            numbers = draw.winning_numbers
            # Generate all possible pairs
            for pair in combinations(numbers, 2):
                pair_counts[pair] += 1

        # Filter by minimum occurrences
        frequent_pairs = {
            pair: count
            for pair, count in pair_counts.items()
            if count >= min_occurrences
        }

        return {
            "frequent_pairs": dict(Counter(frequent_pairs).most_common(30)),
            "total_unique_pairs": len(pair_counts),
            "pairs_above_threshold": len(frequent_pairs),
        }

    def find_number_triples(
        self,
        lookback: Optional[int] = None,
        min_occurrences: int = 3
    ) -> dict:
        """Find frequently occurring number triples."""
        draws = self.number_analyzer.get_all_draws(limit=lookback)

        if not draws:
            return {}

        triple_counts = Counter()

        for draw in draws:
            numbers = draw.winning_numbers
            for triple in combinations(numbers, 3):
                triple_counts[triple] += 1

        frequent_triples = {
            triple: count
            for triple, count in triple_counts.items()
            if count >= min_occurrences
        }

        return {
            "frequent_triples": dict(Counter(frequent_triples).most_common(20)),
            "total_unique_triples": len(triple_counts),
        }

    def analyze_gaps(self, lookback: Optional[int] = None) -> dict:
        """Analyze gaps between number appearances."""
        draws = self.number_analyzer.get_all_draws(limit=lookback)

        if not draws:
            return {}

        # Sort draws by draw number
        draws = sorted(draws, key=lambda d: d.draw_number)

        # Track last appearance for each number
        last_appearance = {}
        gap_history = defaultdict(list)

        for draw in draws:
            all_numbers = draw.winning_numbers + [draw.additional_number]

            for num in range(1, 50):
                if num in all_numbers:
                    if num in last_appearance:
                        gap = draw.draw_number - last_appearance[num]
                        gap_history[num].append(gap)
                    last_appearance[num] = draw.draw_number

        # Calculate gap statistics for each number
        gap_stats = {}
        for num in range(1, 50):
            gaps = gap_history[num]
            if gaps:
                gap_stats[num] = {
                    "min_gap": min(gaps),
                    "max_gap": max(gaps),
                    "avg_gap": np.mean(gaps),
                    "median_gap": np.median(gaps),
                    "current_gap": draws[-1].draw_number - last_appearance.get(num, 0)
                    if num in last_appearance else len(draws),
                }

        # Find numbers with unusually long current gaps
        avg_gaps = {num: stats["avg_gap"] for num, stats in gap_stats.items()}
        current_gaps = {num: stats["current_gap"] for num, stats in gap_stats.items()}

        overdue = []
        for num in range(1, 50):
            if num in gap_stats:
                if current_gaps[num] > avg_gaps[num] * 1.5:
                    overdue.append({
                        "number": num,
                        "current_gap": current_gaps[num],
                        "average_gap": avg_gaps[num],
                        "overdue_ratio": current_gaps[num] / avg_gaps[num]
                    })

        overdue.sort(key=lambda x: x["overdue_ratio"], reverse=True)

        return {
            "gap_statistics": gap_stats,
            "overdue_numbers": overdue[:15],
            "average_gap_all_numbers": np.mean(list(avg_gaps.values())) if avg_gaps else 0,
        }

    def analyze_day_patterns(self, lookback: Optional[int] = None) -> dict:
        """Analyze patterns based on day of week (Monday vs Thursday)."""
        draws = self.number_analyzer.get_all_draws(limit=lookback)

        if not draws:
            return {}

        monday_numbers = Counter()
        thursday_numbers = Counter()
        monday_count = 0
        thursday_count = 0

        for draw in draws:
            day = draw.draw_date.strftime("%A")
            numbers = draw.winning_numbers + [draw.additional_number]

            if day == "Monday":
                monday_count += 1
                for num in numbers:
                    monday_numbers[num] += 1
            elif day == "Thursday":
                thursday_count += 1
                for num in numbers:
                    thursday_numbers[num] += 1

        # Calculate day-specific hot numbers
        monday_hot = [num for num, _ in monday_numbers.most_common(10)]
        thursday_hot = [num for num, _ in thursday_numbers.most_common(10)]

        return {
            "monday_draws": monday_count,
            "thursday_draws": thursday_count,
            "monday_hot_numbers": monday_hot,
            "thursday_hot_numbers": thursday_hot,
            "monday_frequency": dict(monday_numbers),
            "thursday_frequency": dict(thursday_numbers),
        }

    def analyze_sum_patterns(self, lookback: Optional[int] = None) -> dict:
        """Analyze winning number sum patterns."""
        draws = self.number_analyzer.get_all_draws(limit=lookback)

        if not draws:
            return {}

        sums = [sum(d.winning_numbers) for d in draws]

        # Create sum ranges (buckets)
        min_possible = 1 + 2 + 3 + 4 + 5 + 6  # 21
        max_possible = 44 + 45 + 46 + 47 + 48 + 49  # 279

        # Create 10 equal-width buckets
        bucket_size = (max_possible - min_possible) // 10
        buckets = defaultdict(int)

        for s in sums:
            bucket = (s - min_possible) // bucket_size
            bucket_start = min_possible + bucket * bucket_size
            bucket_end = bucket_start + bucket_size - 1
            bucket_label = f"{bucket_start}-{bucket_end}"
            buckets[bucket_label] += 1

        # Find optimal sum range (where most wins occur)
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[1], reverse=True)

        return {
            "min_sum": min(sums),
            "max_sum": max(sums),
            "average_sum": np.mean(sums),
            "median_sum": np.median(sums),
            "std_sum": np.std(sums),
            "sum_distribution": dict(buckets),
            "optimal_range": sorted_buckets[0] if sorted_buckets else None,
            "theoretical_min": min_possible,
            "theoretical_max": max_possible,
        }

    def get_comprehensive_patterns(self, lookback: Optional[int] = None) -> dict:
        """Get all pattern analysis in one call."""
        return {
            "consecutive_patterns": self.find_consecutive_patterns(lookback),
            "number_pairs": self.find_number_pairs(lookback),
            "gap_analysis": self.analyze_gaps(lookback),
            "day_patterns": self.analyze_day_patterns(lookback),
            "sum_patterns": self.analyze_sum_patterns(lookback),
        }
