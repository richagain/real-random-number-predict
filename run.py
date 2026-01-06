#!/usr/bin/env python3
"""Application runner script."""

import argparse
import asyncio
import uvicorn


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run the FastAPI server."""
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


async def populate_data(max_pages: int = None):
    """Populate database with scraped data."""
    from app.scraper.data_loader import populate_database
    await populate_database(max_pages=max_pages)


async def train_model(epochs: int = 50):
    """Train the ML model."""
    from app.database.db import init_db_sync, get_sync_session
    from app.analysis.predictor import TOTOPredictor

    init_db_sync()
    session = get_sync_session()
    try:
        predictor = TOTOPredictor(session)
        result = predictor.train_ml_model(epochs=epochs)
        print(f"Training result: {result}")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Singapore TOTO Predictor")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run web server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    server_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    # Populate command
    populate_parser = subparsers.add_parser("populate", help="Populate database from web")
    populate_parser.add_argument("--max-pages", type=int, help="Maximum pages to scrape")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")

    args = parser.parse_args()

    if args.command == "server":
        run_server(
            host=args.host,
            port=args.port,
            reload=not args.no_reload
        )
    elif args.command == "populate":
        asyncio.run(populate_data(max_pages=args.max_pages))
    elif args.command == "train":
        asyncio.run(train_model(epochs=args.epochs))
    else:
        # Default: run server
        run_server()


if __name__ == "__main__":
    main()
