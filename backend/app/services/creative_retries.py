"""Transient transport retries, separate from schema repair and quality revision."""
import time

from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_diagnostics import diagnostic_event


NETWORK_RETRY_DELAYS = (15, 30, 60)  # Three retries after the initial request.
MAX_RETRY_WAIT_SECONDS = 300


def request_with_retries(budget, *, before_attempt, on_failure, diagnostic_fields=None, **kwargs):
    for index in range(len(NETWORK_RETRY_DELAYS) + 1):
        budget.ensure_available()
        before_attempt(index)
        started = time.monotonic()
        try:
            return budget(**kwargs)
        except CreativeError as exc:
            if not exc.retryable:
                raise
            delay = max(NETWORK_RETRY_DELAYS[index], exc.retry_after or 0) if index < len(NETWORK_RETRY_DELAYS) else 0
            will_retry = index < len(NETWORK_RETRY_DELAYS) and delay <= MAX_RETRY_WAIT_SECONDS
            record = {"attempt": index + 1, "max_retries": len(NETWORK_RETRY_DELAYS),
                      "duration_ms": round((time.monotonic() - started) * 1000),
                      "error_code": exc.code, "http_status": exc.http_status, "error": str(exc),
                      "will_retry": will_retry, "wait_seconds": delay if will_retry else 0}
            on_failure(record)
            diagnostic_event("网络故障等待重试" if will_retry else "网络重试停止", level="WARNING", **(diagnostic_fields or {}), **record)
            if not will_retry:
                raise
            # Short slices make cancellation and total-time limits effective
            # during backoff, rather than waiting for the next model request.
            remaining = delay
            while remaining > 0:
                budget.ensure_available()
                before_attempt(None)
                chunk = min(1, remaining)
                time.sleep(chunk)
                remaining -= chunk
