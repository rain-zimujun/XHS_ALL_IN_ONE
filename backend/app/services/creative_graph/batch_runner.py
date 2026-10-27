"""Persist validated batches inside a node; only return a complete merged output."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from contextlib import nullcontext

from pydantic import ValidationError

from backend.app.services.creative_agents.title_generation import content_fingerprint
from backend.app.services.creative_diagnostics import diagnostic_event
from backend.app.services.creative_orchestrator import AgentRun, AgentValidationError


def run_batches(*, agent, payload, batches, metadata, prefix, version, validate, merge,
                checkpoint, revision=None, prepare_revision=None, max_workers=1, state_lock=None):
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
    results_by_key, traces_by_key = {}, {}
    pending = []
    for index, batch in enumerate(batches, 1):
        key = batch["input_fingerprint"]
        metadata[f"{prefix}_batch"] = {"index": index, "count": len(batches), "item_count": 1}
        if key in cache["results"]:
            results_by_key[key] = agent.output_model.model_validate(cache["results"][key])
            metadata[f"{prefix}_reused_batches"] += 1
            diagnostic_event("复用已完成批次", stage=agent.spec.stage, agent=agent.spec.name,
                             batch_index=index, batch_count=len(batches), item_count=1)
            checkpoint(f"{agent.spec.name} · 复用第 {index}/{len(batches)} 篇")
            continue
        pending.append((index, batch))

    def process(index, batch):
        checkpoint(f"{agent.spec.name} · 正在处理第 {index}/{len(batches)} 篇")
        run = agent.invoke(batch, validate_output=lambda r, current=batch: validate(r, current),
                           on_progress=lambda text: checkpoint(f"{text} · 第 {index}/{len(batches)} 篇"),
                           revision=prepare_revision(revision, batch) if revision else None)
        return batch["input_fingerprint"], run

    first_error = None
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix=f"creative-{prefix}") as executor:
            futures = {}
            for index, batch in pending:
                context = copy_context()
                future = executor.submit(context.run, process, index, batch)
                futures[future] = (index, batch)
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    key, run = future.result()
                except Exception as exc:
                    first_error = first_error or exc
                    for queued in futures:
                        if queued is not future:
                            queued.cancel()
                    continue
                results_by_key[key] = run.output
                traces_by_key[key] = run.attempts
                index, _batch = futures[future]
                guard = state_lock if state_lock is not None else nullcontext()
                with guard:
                    cache["results"][key] = run.output.model_dump(mode="json")
                    metadata[f"{prefix}_completed_batches"] = len(cache["results"])
                    checkpoint(f"{agent.spec.name} · 已完成 {index}/{len(batches)} 篇")
    if first_error:
        raise first_error

    ordered_results = [results_by_key[batch["input_fingerprint"]] for batch in batches]
    traces = [trace for batch in batches for trace in traces_by_key.get(batch["input_fingerprint"], ())]
    return AgentRun(stage=agent.spec.stage, name=agent.spec.name, output=merge(ordered_results), attempts=tuple(traces))
