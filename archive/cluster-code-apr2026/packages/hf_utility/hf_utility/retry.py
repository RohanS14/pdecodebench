"""Retry logic for HuggingFace API calls."""

from __future__ import annotations

import random
import time

API_TIMEOUT_WAIT = 300  # 5 minutes
CONFLICT_BASE_WAIT = 5  # seconds

_RETRYABLE_ERRORS = (
    "timeout", "429", "rate limit",
    "412", "precondition failed",
    "409", "conflict",
)


def _is_conflict_error(error_str: str) -> bool:
    return any(tok in error_str for tok in ("412", "precondition", "409", "conflict"))


def retry_on_timeout(func, *args, max_retries: int = 5, **kwargs):
    """Retry a function on API timeout, rate-limit, or 409/412 conflict."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if any(tok in error_str for tok in _RETRYABLE_ERRORS):
                if attempt < max_retries - 1:
                    if _is_conflict_error(error_str):
                        wait = CONFLICT_BASE_WAIT * (attempt + 1) + random.uniform(0, 5)
                        print(f"Manifest conflict (409/412), retrying in {wait:.1f}s "
                              f"(attempt {attempt + 1}/{max_retries})...")
                    else:
                        wait = API_TIMEOUT_WAIT
                        print(f"API timeout/rate limit, waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise
