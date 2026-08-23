import asyncio
import logging
from app.services.predictions import resolve_due_predictions

logger = logging.getLogger("uvicorn.error")

async def prediction_resolver_loop():
    """
    Background task that runs every 60 seconds.
    Automatically resolves any predictions whose target time has passed.
    """
    logger.info("Starting background prediction resolver loop...")
    while True:
        try:
            # Try to resolve predictions for all users
            resolved = resolve_due_predictions()
            if resolved:
                logger.info(f"Resolved {len(resolved)} prediction(s).")
        except Exception as e:
            logger.error(f"Error in background resolver: {e}")
        
        # Wait 60 seconds before checking again
        await asyncio.sleep(60)