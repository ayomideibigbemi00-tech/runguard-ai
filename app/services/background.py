import asyncio
import logging
from app.services.predictions import resolve_due_predictions

logger = logging.getLogger("uvicorn.error")

async def prediction_resolver_loop():
    logger.info("Starting background prediction resolver loop...")
    while True:
        try:
            resolved = resolve_due_predictions()
            if resolved:
                logger.info(f"Resolved {len(resolved)} prediction(s).")
        except Exception as e:
            logger.error(f"Error in background resolver: {e}")
        await asyncio.sleep(60)