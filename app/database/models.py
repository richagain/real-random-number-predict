"""SQLAlchemy database models for TOTO data."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Draw(Base):
    """Model for storing TOTO draw results."""

    __tablename__ = "draws"

    id = Column(Integer, primary_key=True, autoincrement=True)
    draw_number = Column(Integer, unique=True, nullable=False, index=True)
    draw_date = Column(DateTime, nullable=False, index=True)

    # The 6 winning numbers (stored in ascending order)
    num1 = Column(Integer, nullable=False)
    num2 = Column(Integer, nullable=False)
    num3 = Column(Integer, nullable=False)
    num4 = Column(Integer, nullable=False)
    num5 = Column(Integer, nullable=False)
    num6 = Column(Integer, nullable=False)

    # Additional number
    additional_number = Column(Integer, nullable=False)

    # Prize information (optional)
    group1_prize = Column(Float, nullable=True)
    group1_winners = Column(Integer, nullable=True)
    group2_prize = Column(Float, nullable=True)
    group2_winners = Column(Integer, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Index for faster queries on number combinations
    __table_args__ = (
        Index('ix_all_numbers', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6'),
    )

    @property
    def winning_numbers(self) -> list[int]:
        """Return all 6 winning numbers as a sorted list."""
        return sorted([self.num1, self.num2, self.num3, self.num4, self.num5, self.num6])

    @property
    def all_numbers(self) -> list[int]:
        """Return all 7 numbers (6 winning + additional)."""
        return self.winning_numbers + [self.additional_number]

    def __repr__(self):
        nums = "-".join(str(n) for n in self.winning_numbers)
        return f"<Draw #{self.draw_number} ({self.draw_date.strftime('%Y-%m-%d')}): {nums} + {self.additional_number}>"


class Prediction(Base):
    """Model for storing generated predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Strategy used
    strategy = Column(String(50), nullable=False)  # hot, cold, balanced, ml, ensemble

    # Predicted numbers
    num1 = Column(Integer, nullable=False)
    num2 = Column(Integer, nullable=False)
    num3 = Column(Integer, nullable=False)
    num4 = Column(Integer, nullable=False)
    num5 = Column(Integer, nullable=False)
    num6 = Column(Integer, nullable=False)

    # Confidence score (0-1)
    confidence = Column(Float, nullable=True)

    # Target draw (if known)
    target_draw_number = Column(Integer, nullable=True)

    # Result tracking (filled in after draw)
    actual_draw_id = Column(Integer, nullable=True)
    matches = Column(Integer, nullable=True)  # How many numbers matched
    matched_additional = Column(Boolean, nullable=True)

    @property
    def predicted_numbers(self) -> list[int]:
        """Return predicted numbers as a sorted list."""
        return sorted([self.num1, self.num2, self.num3, self.num4, self.num5, self.num6])

    def __repr__(self):
        nums = "-".join(str(n) for n in self.predicted_numbers)
        return f"<Prediction ({self.strategy}): {nums}>"


class NumberStatistics(Base):
    """Model for caching number statistics."""

    __tablename__ = "number_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    number = Column(Integer, unique=True, nullable=False, index=True)

    # Frequency statistics
    total_appearances = Column(Integer, default=0)
    appearances_as_main = Column(Integer, default=0)  # In the 6 winning numbers
    appearances_as_additional = Column(Integer, default=0)

    # Recency
    last_drawn_date = Column(DateTime, nullable=True)
    last_drawn_number = Column(Integer, nullable=True)  # Draw number when last appeared
    draws_since_last = Column(Integer, default=0)  # Gap

    # Position frequency (how often appears in each position when sorted)
    position1_count = Column(Integer, default=0)
    position2_count = Column(Integer, default=0)
    position3_count = Column(Integer, default=0)
    position4_count = Column(Integer, default=0)
    position5_count = Column(Integer, default=0)
    position6_count = Column(Integer, default=0)

    # Calculated scores
    hot_score = Column(Float, default=0.0)  # Higher = more frequent
    cold_score = Column(Float, default=0.0)  # Higher = more overdue

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<NumberStats #{self.number}: {self.total_appearances} appearances>"


class BacktestResult(Base):
    """Model for storing backtest results."""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Backtest configuration
    strategy = Column(String(50), nullable=False)  # hot, cold, balanced, ml, ensemble
    start_draw = Column(Integer, nullable=False)
    end_draw = Column(Integer, nullable=False)
    total_predictions = Column(Integer, nullable=False)

    # Match distribution (count of predictions with 0, 1, 2, 3, 4, 5, 6 matches)
    matches_0 = Column(Integer, default=0)
    matches_1 = Column(Integer, default=0)
    matches_2 = Column(Integer, default=0)
    matches_3 = Column(Integer, default=0)
    matches_4 = Column(Integer, default=0)
    matches_5 = Column(Integer, default=0)
    matches_6 = Column(Integer, default=0)

    # Aggregate metrics
    average_matches = Column(Float, nullable=False)
    best_match = Column(Integer, nullable=False)
    execution_time_seconds = Column(Float, nullable=True)

    @property
    def match_distribution(self) -> dict[int, int]:
        """Return match distribution as a dictionary."""
        return {
            0: self.matches_0,
            1: self.matches_1,
            2: self.matches_2,
            3: self.matches_3,
            4: self.matches_4,
            5: self.matches_5,
            6: self.matches_6,
        }

    @property
    def hit_rate_1_plus(self) -> float:
        """Percentage of predictions with at least 1 match."""
        if self.total_predictions == 0:
            return 0.0
        hits = sum([self.matches_1, self.matches_2, self.matches_3,
                    self.matches_4, self.matches_5, self.matches_6])
        return (hits / self.total_predictions) * 100

    @property
    def hit_rate_2_plus(self) -> float:
        """Percentage of predictions with at least 2 matches."""
        if self.total_predictions == 0:
            return 0.0
        hits = sum([self.matches_2, self.matches_3, self.matches_4,
                    self.matches_5, self.matches_6])
        return (hits / self.total_predictions) * 100

    @property
    def hit_rate_3_plus(self) -> float:
        """Percentage of predictions with at least 3 matches."""
        if self.total_predictions == 0:
            return 0.0
        hits = sum([self.matches_3, self.matches_4, self.matches_5, self.matches_6])
        return (hits / self.total_predictions) * 100

    def __repr__(self):
        return f"<BacktestResult ({self.strategy}): {self.start_draw}-{self.end_draw}, avg={self.average_matches:.2f}>"
