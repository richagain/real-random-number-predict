"""FastAPI route handlers."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_session
from app.database.models import Draw
from app.models.schemas import (
    DrawResponse, DrawListResponse, StatisticsResponse,
    PredictionRequest, PredictionResponse, HealthResponse
)


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(session: AsyncSession = Depends(get_session)):
    """Check API health and database status."""
    try:
        total = await session.execute(select(func.count(Draw.id)))
        total_draws = total.scalar() or 0

        latest = await session.execute(
            select(Draw.draw_date).order_by(desc(Draw.draw_number)).limit(1)
        )
        last_date = latest.scalar_one_or_none()

        return HealthResponse(
            status="healthy",
            database="connected",
            total_draws=total_draws,
            last_draw_date=last_date
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/draws", response_model=DrawListResponse)
async def list_draws(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session)
):
    """List historical draws with pagination."""
    # Get total count
    total_result = await session.execute(select(func.count(Draw.id)))
    total = total_result.scalar() or 0

    # Get paginated draws
    offset = (page - 1) * page_size
    result = await session.execute(
        select(Draw)
        .order_by(desc(Draw.draw_number))
        .offset(offset)
        .limit(page_size)
    )
    draws = result.scalars().all()

    return DrawListResponse(
        draws=[
            DrawResponse(
                id=d.id,
                draw_number=d.draw_number,
                draw_date=d.draw_date,
                winning_numbers=d.winning_numbers,
                additional_number=d.additional_number,
                group1_prize=d.group1_prize,
                group1_winners=d.group1_winners,
            )
            for d in draws
        ],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/draws/{draw_number}", response_model=DrawResponse)
async def get_draw(
    draw_number: int,
    session: AsyncSession = Depends(get_session)
):
    """Get a specific draw by number."""
    result = await session.execute(
        select(Draw).where(Draw.draw_number == draw_number)
    )
    draw = result.scalar_one_or_none()

    if not draw:
        raise HTTPException(status_code=404, detail="Draw not found")

    return DrawResponse(
        id=draw.id,
        draw_number=draw.draw_number,
        draw_date=draw.draw_date,
        winning_numbers=draw.winning_numbers,
        additional_number=draw.additional_number,
        group1_prize=draw.group1_prize,
        group1_winners=draw.group1_winners,
    )


@router.get("/statistics")
async def get_statistics(
    lookback: Optional[int] = Query(None, ge=1, le=5000),
    session: AsyncSession = Depends(get_session)
):
    """Get number frequency statistics."""
    from app.database.db import get_sync_session
    from app.analysis.statistics import NumberAnalyzer

    # Use sync session for complex analysis
    sync_session = get_sync_session()
    try:
        analyzer = NumberAnalyzer(sync_session)
        stats = analyzer.get_comprehensive_statistics(lookback=lookback)
        return stats
    finally:
        sync_session.close()


@router.get("/patterns")
async def get_patterns(
    lookback: Optional[int] = Query(None, ge=1, le=5000),
    session: AsyncSession = Depends(get_session)
):
    """Get pattern analysis."""
    from app.database.db import get_sync_session
    from app.analysis.patterns import PatternAnalyzer

    sync_session = get_sync_session()
    try:
        analyzer = PatternAnalyzer(sync_session)
        patterns = analyzer.get_comprehensive_patterns(lookback=lookback)
        return patterns
    finally:
        sync_session.close()


@router.post("/predict", response_model=PredictionResponse)
async def generate_prediction(
    request: PredictionRequest,
    session: AsyncSession = Depends(get_session)
):
    """Generate number predictions."""
    from app.database.db import get_sync_session
    from app.analysis.predictor import TOTOPredictor

    sync_session = get_sync_session()
    try:
        # Get total draws for context
        total_result = await session.execute(select(func.count(Draw.id)))
        total_draws = total_result.scalar() or 0

        predictor = TOTOPredictor(sync_session)
        predictions = predictor.predict(
            strategy=request.strategy,
            count=request.count
        )

        return PredictionResponse(
            predictions=[
                {
                    "numbers": p["numbers"],
                    "strategy": p["strategy"],
                    "confidence": p.get("confidence"),
                    "reasoning": p.get("reasoning"),
                }
                for p in predictions
            ],
            generated_at=datetime.now(),
            based_on_draws=total_draws
        )
    finally:
        sync_session.close()


@router.post("/predict/all")
async def generate_all_predictions(
    count: int = Query(1, ge=1, le=5),
    session: AsyncSession = Depends(get_session)
):
    """Generate predictions from all strategies."""
    from app.database.db import get_sync_session
    from app.analysis.predictor import TOTOPredictor

    sync_session = get_sync_session()
    try:
        predictor = TOTOPredictor(sync_session)
        all_predictions = predictor.predict_all_strategies(count_per_strategy=count)

        total_result = await session.execute(select(func.count(Draw.id)))
        total_draws = total_result.scalar() or 0

        return {
            "predictions": all_predictions,
            "generated_at": datetime.now().isoformat(),
            "based_on_draws": total_draws
        }
    finally:
        sync_session.close()


@router.post("/train")
async def train_model(
    epochs: int = Query(50, ge=10, le=200),
    session: AsyncSession = Depends(get_session)
):
    """Train the ML prediction model."""
    from app.database.db import get_sync_session
    from app.analysis.predictor import TOTOPredictor

    sync_session = get_sync_session()
    try:
        predictor = TOTOPredictor(sync_session)
        result = predictor.train_ml_model(epochs=epochs)
        return result
    finally:
        sync_session.close()


@router.post("/scrape")
async def trigger_scrape(
    max_pages: Optional[int] = Query(None, ge=1, le=50),
    session: AsyncSession = Depends(get_session)
):
    """Trigger data scraping and population."""
    from app.scraper.data_loader import DataLoader
    from app.database.db import get_sync_session

    sync_session = get_sync_session()
    try:
        loader = DataLoader(sync_session)
        result = await loader.populate_from_scraper(max_pages=max_pages)
        return result
    finally:
        sync_session.close()


@router.get("/scheduler")
async def get_scheduler_status():
    """Get scheduler status and next run times."""
    from app.scheduler import get_scheduler_status
    return get_scheduler_status()
