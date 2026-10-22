type Failure = { attempt: number; duration_ms: number; status: string; error: string };
type NetworkFailure = { attempt: number; max_retries: number; duration_ms: number; error: string;
  error_code: string; will_retry: boolean; wait_seconds: number; role?: string };

export function readNetworkAttempts(metadata: unknown): NetworkFailure[] {
  if (!metadata || typeof metadata !== "object") return [];
  const attempts = (metadata as Record<string, unknown>).network_attempts;
  if (!Array.isArray(attempts)) return [];
  return attempts.filter((value): value is NetworkFailure => Boolean(value &&
    typeof value.attempt === "number" && Number.isFinite(value.attempt) &&
    typeof value.max_retries === "number" && Number.isFinite(value.max_retries) &&
    typeof value.duration_ms === "number" && Number.isFinite(value.duration_ms) &&
    typeof value.wait_seconds === "number" && Number.isFinite(value.wait_seconds) &&
    typeof value.will_retry === "boolean" && typeof value.error_code === "string" && typeof value.error === "string"));
}

export function readFailedAttempts(metadata: unknown): Failure[] {
  if (!metadata || typeof metadata !== "object") return [];
  const attempts = (metadata as Record<string, unknown>).failed_attempts;
  if (!Array.isArray(attempts)) return [];
  return attempts.filter((value): value is Failure => Boolean(value &&
    typeof value.attempt === "number" && Number.isFinite(value.attempt) &&
    typeof value.duration_ms === "number" && Number.isFinite(value.duration_ms) &&
    typeof value.status === "string" && typeof value.error === "string"));
}
