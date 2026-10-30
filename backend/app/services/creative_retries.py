"""Transient transport retries, separate from schema repair and quality revision."""
import time

from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_diagnostics import diagnostic_event


NETWORK_RETRY_DELAYS = (15, 30, 60, 90, 120)  # Five retries after the initial request.
NETWORK_ATTEMPTS = len(NETWORK_RETRY_DELAYS) + 1  # Initial request plus retries.
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
                # Name the likely cause and the safe next step: the task keeps
                # its snapshot, so a short wait and the existing resume entry
                # point is cheaper than starting the whole run again.
                raise CreativeError(
                    f"{exc}已自动重试 {len(NETWORK_RETRY_DELAYS)} 次仍未成功，通常是模型服务或中转站超时、限流或网络抖动；"
                    "建议等待几分钟后使用「从失败阶段继续」，不要立刻重新生成。",
                    exc.code, retryable=exc.retryable, http_status=exc.http_status, retry_after=exc.retry_after,
                ) from exc
            # Short slices make cancellation and total-time limits effective
            # during backoff, rather than waiting for the next model request.
            remaining = delay
            while remaining > 0:
                budget.ensure_available()
                before_attempt(None)
                chunk = min(1, remaining)
                time.sleep(chunk)
                remaining -= chunk
