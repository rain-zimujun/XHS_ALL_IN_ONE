"""Persist validated batches inside a node; only return a complete merged output."""
from pydantic import ValidationError

from backend.app.services.creative_agents.title_generation import content_fingerprint
from backend.app.services.creative_diagnostics import diagnostic_event
from backend.app.services.creative_orchestrator import AgentRun, AgentValidationError


def run_batches(*, agent, payload, batches, metadata, prefix, version, validate, merge,
                checkpoint, revision=None, prepare_revision=None):
    revision_key = content_fingerprint(revision) if revision else None
    cache = metadata.get(f"{prefix}_batches")
    if (not isinstance(cache, dict) or cache.get("version") != version
            or cache.get("input_fingerprint") != payload["input_fingerprint"]
            or cache.get("revision_fingerprint") != revision_key or not isinstance(cache.get("results"), dict)):
        cache = {"version": version, "input_fingerprint": payload["input_fingerprint"],
                 "revision_fingerprint": revision_key, "results": {}}
    metadata[f"{prefix}_batches"] = cache
    known = {b["input_fingerprint"] for b in batches}
    cache["results"] = {k: v for k, v in cache["results"].items() if k in known}
    for batch in batches:
        key = batch["input_fingerprint"]
        if key in cache["results"]:
            try:
                output = agent.output_model.model_validate(cache["results"][key])
                validate(output, batch)
                if revision:
                    from backend.app.services.creative_agents.revision import ensure_targeted_revision
                    local = prepare_revision(revision, batch)
                    ensure_targeted_revision(agent.spec.stage, local["previous_output"], output.model_dump(mode="json"), local["target_ids"])
            except (ValidationError, AgentValidationError):
                cache["results"].pop(key)
    metadata.update({f"{prefix}_batch_count": len(batches), f"{prefix}_reused_batches": 0,
                     f"{prefix}_completed_batches": len(cache["results"])})
    results, traces = [], []
    for index, batch in enumerate(batches, 1):
        key = batch["input_fingerprint"]
        metadata[f"{prefix}_batch"] = {"index": index, "count": len(batches), "item_count": 1}
        if key in cache["results"]:
            results.append(agent.output_model.model_validate(cache["results"][key]))
            metadata[f"{prefix}_reused_batches"] += 1
            diagnostic_event("复用已完成批次", stage=agent.spec.stage, agent=agent.spec.name,
                             batch_index=index, batch_count=len(batches), item_count=1)
            checkpoint(f"{agent.spec.name} · 复用第 {index}/{len(batches)} 篇")
            continue
        checkpoint(f"{agent.spec.name} · 正在处理第 {index}/{len(batches)} 篇")
        run = agent.invoke(batch, validate_output=lambda r, current=batch: validate(r, current),
                           on_progress=lambda text: checkpoint(f"{text} · 第 {index}/{len(batches)} 篇"),
                           revision=prepare_revision(revision, batch) if revision else None)
        results.append(run.output)
        traces.extend(run.attempts)
        cache["results"][key] = run.output.model_dump(mode="json")
        metadata[f"{prefix}_completed_batches"] = len(cache["results"])
        checkpoint(f"{agent.spec.name} · 已完成 {index}/{len(batches)} 篇")
    return AgentRun(stage=agent.spec.stage, name=agent.spec.name, output=merge(results), attempts=tuple(traces))
