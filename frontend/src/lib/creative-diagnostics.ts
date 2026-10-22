type Failure = { attempt: number; duration_ms: number; status: string; error: string };

export function readFailedAttempts(metadata: unknown): Failure[] {
  if (!metadata || typeof metadata !== "object") return [];
  const attempts = (metadata as Record<string, unknown>).failed_attempts;
  if (!Array.isArray(attempts)) return [];
  return attempts.filter((value): value is Failure => Boolean(value &&
    typeof value.attempt === "number" && Number.isFinite(value.attempt) &&
    typeof value.duration_ms === "number" && Number.isFinite(value.duration_ms) &&
    typeof value.status === "string" && typeof value.error === "string"));
}
