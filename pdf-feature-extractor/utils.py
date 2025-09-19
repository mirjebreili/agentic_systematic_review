"""
Helper functions and utilities.
"""

import hashlib
import logging
from pathlib import Path
import time
import functools
from typing import Any, Callable

def generate_paper_id(pdf_path: str) -> str:
    """Generate a unique and consistent ID from the PDF path, truncated to 63 chars for ChromaDB."""
    return hashlib.sha256(pdf_path.encode('utf-8')).hexdigest()[:63]

def setup_logging(log_level: str, log_file: Path):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level.upper(),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    # Suppress noisy logs from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1.0):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= retries:
                        logging.error(f"Function {func.__name__} failed after {retries} retries. Error: {e}")
                        raise
                    sleep_time = backoff_in_seconds * (2 ** (attempts - 1))
                    logging.warning(f"Attempt {attempts} failed for {func.__name__}. Retrying in {sleep_time:.2f} seconds. Error: {e}")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

def timer_decorator(func: Callable) -> Callable:
    """Decorator to measure and log the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' executed in {run_time:.4f} seconds")
        return result
    return wrapper
