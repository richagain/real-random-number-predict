"""Backtesting engine for evaluating prediction strategies."""

import time
from datetime import datetime
from typing import Optional
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from app.database.models import Draw, BacktestResult
from app.analysis.predictor import TOTOPredictor


class Backtester:
    """Backtesting engine to evaluate prediction strategies against historical data."""

    AVAILABLE_STRATEGIES = ["hot", "cold", "balanced", "ml", "ensemble"]

    def __init__(self, session: Session):
        self.session = session

    def get_test_draws(self, num_draws: int = 1000) -> list[Draw]:
        """Get the draws to test against (most recent N draws)."""
        result = self.session.execute(
            select(Draw)
            .order_by(desc(Draw.draw_number))
            .limit(num_draws)
        )
        draws = list(result.scalars().all())
        # Return in chronological order (oldest first)
        return sorted(draws, key=lambda d: d.draw_number)

    def _count_matches(self, predicted: list[int], actual: list[int]) -> int:
        """Count how many predicted numbers match the actual winning numbers."""
        return len(set(predicted) & set(actual))

    def run(
        self,
        strategy: str,
        num_draws: int = 1000,
        save_result: bool = True
    ) -> BacktestResult:
        """Run backtest for a single strategy.

        Args:
            strategy: The prediction strategy to test
            num_draws: Number of recent draws to test against
            save_result: Whether to save the result to database

        Returns:
            BacktestResult with aggregated metrics
        """
        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {self.AVAILABLE_STRATEGIES}")

        start_time = time.time()

        # Get test draws
        test_draws = self.get_test_draws(num_draws)
        if len(test_draws) < 2:
            raise ValueError("Not enough draws in database for backtesting")

        # Initialize match counters
        match_counts = {i: 0 for i in range(7)}  # 0-6 matches
        total_matches = 0

        # For each draw, predict using only data before that draw
        for i, draw in enumerate(test_draws):
            # Skip first draw (no prior data)
            if i == 0:
                continue

            # Create predictor with cutoff at this draw
            predictor = TOTOPredictor(self.session, max_draw_number=draw.draw_number)

            try:
                # Generate prediction
                predictions = predictor.predict(strategy=strategy, count=1)
                if not predictions:
                    continue

                predicted_numbers = predictions[0]["numbers"]
                actual_numbers = draw.winning_numbers

                # Count matches
                matches = self._count_matches(predicted_numbers, actual_numbers)
                match_counts[matches] += 1
                total_matches += matches

            except Exception as e:
                # Skip this draw if prediction fails
                print(f"Warning: Prediction failed for draw {draw.draw_number}: {e}")
                continue

        execution_time = time.time() - start_time
        total_predictions = sum(match_counts.values())

        if total_predictions == 0:
            raise ValueError("No predictions could be generated")

        # Calculate metrics
        average_matches = total_matches / total_predictions if total_predictions > 0 else 0
        best_match = max(i for i, count in match_counts.items() if count > 0)

        # Create result object
        result = BacktestResult(
            strategy=strategy,
            start_draw=test_draws[0].draw_number,
            end_draw=test_draws[-1].draw_number,
            total_predictions=total_predictions,
            matches_0=match_counts[0],
            matches_1=match_counts[1],
            matches_2=match_counts[2],
            matches_3=match_counts[3],
            matches_4=match_counts[4],
            matches_5=match_counts[5],
            matches_6=match_counts[6],
            average_matches=average_matches,
            best_match=best_match,
            execution_time_seconds=execution_time,
        )

        if save_result:
            self.session.add(result)
            self.session.commit()
            self.session.refresh(result)

        return result

    def run_all_strategies(
        self,
        num_draws: int = 1000,
        save_results: bool = True
    ) -> list[BacktestResult]:
        """Run backtest for all available strategies.

        Args:
            num_draws: Number of recent draws to test against
            save_results: Whether to save results to database

        Returns:
            List of BacktestResult objects, one per strategy
        """
        results = []

        for strategy in self.AVAILABLE_STRATEGIES:
            try:
                result = self.run(
                    strategy=strategy,
                    num_draws=num_draws,
                    save_result=save_results
                )
                results.append(result)
            except Exception as e:
                print(f"Warning: Backtest failed for {strategy}: {e}")

        return results

    def get_saved_results(
        self,
        strategy: Optional[str] = None,
        limit: int = 20
    ) -> list[BacktestResult]:
        """Get previously saved backtest results.

        Args:
            strategy: Filter by strategy (optional)
            limit: Maximum number of results to return

        Returns:
            List of BacktestResult objects
        """
        query = select(BacktestResult).order_by(desc(BacktestResult.created_at))

        if strategy:
            query = query.where(BacktestResult.strategy == strategy)

        query = query.limit(limit)

        result = self.session.execute(query)
        return list(result.scalars().all())

    def get_result_by_id(self, result_id: int) -> Optional[BacktestResult]:
        """Get a specific backtest result by ID."""
        result = self.session.execute(
            select(BacktestResult).where(BacktestResult.id == result_id)
        )
        return result.scalar_one_or_none()

    def compare_strategies(self, results: list[BacktestResult]) -> dict:
        """Compare multiple backtest results and return summary.

        Args:
            results: List of BacktestResult objects to compare

        Returns:
            Dictionary with comparison metrics
        """
        if not results:
            return {}

        comparison = {
            "strategies": [],
            "best_average_matches": None,
            "best_hit_rate_3_plus": None,
        }

        best_avg = 0
        best_hit_rate = 0

        for result in results:
            strategy_data = {
                "strategy": result.strategy,
                "average_matches": round(result.average_matches, 3),
                "best_match": result.best_match,
                "hit_rate_1_plus": round(result.hit_rate_1_plus, 2),
                "hit_rate_2_plus": round(result.hit_rate_2_plus, 2),
                "hit_rate_3_plus": round(result.hit_rate_3_plus, 2),
                "total_predictions": result.total_predictions,
                "execution_time": round(result.execution_time_seconds, 2) if result.execution_time_seconds else None,
            }
            comparison["strategies"].append(strategy_data)

            if result.average_matches > best_avg:
                best_avg = result.average_matches
                comparison["best_average_matches"] = result.strategy

            if result.hit_rate_3_plus > best_hit_rate:
                best_hit_rate = result.hit_rate_3_plus
                comparison["best_hit_rate_3_plus"] = result.strategy

        # Sort by average matches
        comparison["strategies"].sort(key=lambda x: x["average_matches"], reverse=True)

        return comparison
