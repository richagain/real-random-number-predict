"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DrawBase(BaseModel):
    """Base schema for draw data."""
    draw_number: int
    draw_date: datetime
    winning_numbers: list[int] = Field(..., min_length=6, max_length=6)
    additional_number: int = Field(..., ge=1, le=49)

    @field_validator('winning_numbers')
    @classmethod
    def validate_numbers(cls, v):
        if len(v) != 6:
            raise ValueError('Must have exactly 6 winning numbers')
        if not all(1 <= n <= 49 for n in v):
            raise ValueError('All numbers must be between 1 and 49')
        if len(set(v)) != 6:
            raise ValueError('All numbers must be unique')
        return sorted(v)


class DrawCreate(DrawBase):
    """Schema for creating a new draw."""
    group1_prize: Optional[float] = None
    group1_winners: Optional[int] = None


class DrawResponse(DrawBase):
    """Schema for draw response."""
    id: int
    group1_prize: Optional[float] = None
    group1_winners: Optional[int] = None

    class Config:
        from_attributes = True


class DrawListResponse(BaseModel):
    """Schema for paginated draw list."""
    draws: list[DrawResponse]
    total: int
    page: int
    page_size: int


class NumberFrequency(BaseModel):
    """Schema for number frequency statistics."""
    number: int
    total_appearances: int
    appearances_as_main: int
    appearances_as_additional: int
    last_drawn_date: Optional[datetime]
    draws_since_last: int
    hot_score: float
    cold_score: float
    frequency_percentage: float


class StatisticsResponse(BaseModel):
    """Schema for statistics response."""
    total_draws: int
    date_range: dict[str, datetime]
    number_frequencies: list[NumberFrequency]
    hot_numbers: list[int]
    cold_numbers: list[int]
    most_common_sums: list[dict]
    odd_even_distribution: dict[str, float]
    high_low_distribution: dict[str, float]


class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    strategy: str = Field(
        default="ensemble",
        description="Prediction strategy: hot, cold, balanced, ml, ensemble"
    )
    count: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of predictions to generate"
    )

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        valid_strategies = ['hot', 'cold', 'balanced', 'ml', 'ensemble']
        if v.lower() not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v.lower()


class PredictionResult(BaseModel):
    """Schema for a single prediction result."""
    numbers: list[int] = Field(..., min_length=6, max_length=6)
    strategy: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    predictions: list[PredictionResult]
    generated_at: datetime
    based_on_draws: int


class PatternAnalysis(BaseModel):
    """Schema for pattern analysis."""
    consecutive_pairs: dict[str, int]  # e.g., {"12-13": 5, "23-24": 3}
    sum_distribution: list[dict[str, int]]  # Sum ranges and counts
    gap_analysis: dict[int, int]  # Number -> current gap
    day_patterns: dict[str, dict]  # Monday/Thursday patterns


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    database: str
    total_draws: int
    last_draw_date: Optional[datetime]
