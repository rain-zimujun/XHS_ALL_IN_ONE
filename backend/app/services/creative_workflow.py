"""Persistable, bounded creative workflow; no model-selected tools or external actions.

Only verified outputs created inside this run are passed downstream. Old drafts
remain in stage history; invalidated reviews never authorize a new final result.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import partial

from backend.app.core.security import decrypt_text
from backend.app.schemas.creative import CreativeAnalysisOptions, CreativeAnalysisResult
from backend.app.services.creative_agents.sample_analysis import run_sample_analysis
from backend.app.services.creative_agents.reader_needs import run_reader_needs
from backend.app.services.creative_agents.topic_selection import run_topic_selection
from backend.app.services.creative_agents.topic_review import run_topic_review
from backend.app.services.creative_agents.title_generation import run_title_generation
from backend.app.services.creative_agents.title_review import run_title_review
from backend.app.services.creative_agents.copy_generation import run_copy_generation
from backend.app.services.creative_agents.copy_review import run_copy_review
from backend.app.services.creative_agents.image_planning import run_image_planning
from backend.app.services.creative_agents.final_review import run_final_review
from backend.app.services.creative_agents.revision import (
    MAX_QUALITY_REVISIONS, MAX_MODEL_REQUESTS, PipelineCancelled, RequestBudget, revision_completion, revision_items,
)
from backend.app.services.creative_analysis import CreativeError, owned_text_model, request_analysis
from backend.app.services.creative_orchestrator import AgentRunError
from backend.app.services.creative_pipeline import new_pipeline_state, utc_now, validate_pipeline_state
from backend.app.services.creative_diagnostics import diagnostic_event
from backend.app.services.creative_retries import request_with_retries


PRODUCERS = {"topic": run_topic_selection, "title": run_title_generation, "copy": run_copy_generation, "image": run_image_planning}
JUDGES = {"topic": run_topic_review, "title": run_title_review, "copy": run_copy_review}
PRODUCTION_ORDER = tuple(PRODUCERS)


class CreativeWorkflow:
    def __init__(self, *, search, sources, options, complete, on_state=None, max_requests=MAX_MODEL_REQUESTS):
        self.search, self.sources = search, sources
        self.options = CreativeAnalysisOptions.model_validate(options)
        self.on_state = on_state
        self.state = new_pipeline_state()
        self.state["max_requests"] = max_requests
        self.state["max_quality_revisions"] = MAX_QUALITY_REVISIONS
        self.current = "sample_analysis"
        self.outputs, self.reviews = {}, {}
        self.transport = complete
        self.budget = RequestBudget(self.recorded_transport, max_requests=max_requests)

    def save(self, activity=None):
        if activity:
            self.state["activity"] = activity
        self.state["updated_at"] = utc_now()
        self.state["request_count"] = self.budget.request_count
        validate_pipeline_state(self.state)
        if self.on_state:
            self.on_state(deepcopy(self.state))

    def complete(self, **kwargs):
        # Persist intent/count before network IO, and observe cancellation on both sides.
        self.save()
        raw = request_with_retries(self.budget, before_attempt=self._network_progress,
                                   on_failure=self._network_failure, diagnostic_fields={"stage": self.current}, **kwargs)
        self.save()
        return raw

    def _network_progress(self, retry):
        self.save(f"当前 Agent 正在进行网络重试 {retry}/3" if retry else None)

    def _network_failure(self, record):
        self.state["stages"][self.current]["metadata"].setdefault("network_attempts", []).append(record)
        self.save(f"网络异常，{record['wait_seconds']:g} 秒后重试 {record['attempt']}/3" if record["will_retry"] else "当前模型请求的网络重试已停止")

    def recorded_transport(self, **kwargs):
        self.save()  # RequestBudget has now counted the attempted request.
        return self.transport(**kwargs)

    def invoke(self, stage, fn, inputs, *, reviewer=False, revision=None):
        self.current = stage
        self.state["current_stage"] = stage
        item = self.state["stages"][stage]
        item["status"] = "running"
        self.state["status"] = "running"
        started = utc_now()
        attempt = {"attempt": len(item["attempts"]) + 1, "started_at": started, "finished_at": None, "status": "running", "error": None}
        item["attempts"].append(attempt)
        self.save()
        completion = self.complete
        if revision:
            completion = revision_completion(completion, stage=stage, **revision)
        requests_before = self.budget.request_count
        try:
            run, metadata = fn(**inputs, complete=completion, on_progress=self.save)
        except AgentRunError as exc:
            item["metadata"]["failed_attempts"] = [asdict(a) for a in exc.attempts]
            raise CreativeError("当前阶段两次输出未通过结构或交接校验，已保留此前稿件与评审。", "creative_output_invalid") from exc
        output = run.output.model_dump(mode="json")
        metadata["request_count"] = self.budget.request_count - requests_before
        diagnostic_event("评审完成" if reviewer else "阶段完成", stage=stage,
                         request_count=self.budget.request_count,
                         **({"review_status": output["status"]} if reviewer else {}))
        attempt.update(status="completed", finished_at=utc_now())
        item["metadata"] = {**item["metadata"], **metadata}
        if reviewer:
            item["review_result"] = output
            self.reviews[stage] = output
        else:
            item["revision"] += 1
            item["output"] = output
            self.outputs[stage] = output
        item["history"].append({
            "role": "reviewer" if reviewer else "producer", "revision": item["revision"],
            "output": output, "metadata": deepcopy(item["metadata"]), "traces": [asdict(a) for a in run.attempts],
            "target_ids": revision["target_ids"] if revision else [], "finished_at": utc_now(),
        })
        item["status"] = "completed"
        self.save()
        return output

    def inputs(self, stage, *, reviewer=False):
        config = self.options
        base = dict(keyword=self.search["keyword"], requirements=config.instruction or "围绕样本写自然精简的原创内容。",
                    requested_idea_count=config.idea_count, sample_analysis=self.outputs["sample_analysis"], audience_brief=self.outputs["audience"])
        if stage == "topic" and not reviewer:
            return base
        base.update(topic_brief=self.outputs["topic"], topic_revision=self.state["stages"]["topic"]["revision"])
        if stage == "topic":
            return base
        base.update(topic_review=self.reviews["topic"])
        if stage == "title" and not reviewer:
            return base
        base.update(title_candidates=self.outputs["title"], title_revision=self.state["stages"]["title"]["revision"])
        if stage == "title":
            return base
        base.update(title_review=self.reviews["title"], variant_count=config.variant_count)
        if stage == "copy" and not reviewer:
            return base
        base.update(copy_variants=self.outputs["copy"], copy_revision=self.state["stages"]["copy"]["revision"])
        if stage == "copy":
            return base
        base.update(copy_review=self.reviews["copy"])
        if stage == "image":
            return base
        base.update(image_plans=self.outputs["image"], final_revision=len(self.state["stages"]["final_review"]["history"]) + 1)
        return base

    def failed_ids(self, stage, review):
        if stage == "topic":
            return review["revision_required_idea_ids"]
        if stage == "title":
            return [c["candidate_id"] for i in self.outputs["title"]["ideas"] if i["idea_id"] in review["unresolved_idea_ids"] for c in i["candidates"]]
        return review["revision_required_ids"]

    def quality_stop(self, message):
        raise CreativeError(message + " 当前稿件和评审已保存，没有将未通过内容作为最终成稿。", "creative_quality_exhausted")

    def revision(self, stage, ids, feedback, *, dependency=False):
        item = self.state["stages"][stage]
        if not dependency:
            if item["quality_revisions"] >= MAX_QUALITY_REVISIONS:
                self.quality_stop(f"当前阶段已达到 {MAX_QUALITY_REVISIONS} 次质量返工上限。")
            item["quality_revisions"] += 1
        previous = deepcopy(self.outputs[stage])
        # Dependency changes can rebind an affected group to its newly selected title.
        # Rebase only server-owned parent metadata; unchanged text is still protected.
        if stage == "copy":
            selected = self.reviews["title"]["selected_candidate_ids"]
            for idea in previous["ideas"]:
                if all(v["variant_id"] in ids for v in idea["variants"]):
                    idea["candidate_id"] = selected[idea["idea_id"]]
        elif stage == "image":
            selected = self.reviews["title"]["selected_candidate_ids"]
            for v in previous["variants"]:
                if v["variant_id"] in ids:
                    v["candidate_id"] = selected[v["idea_id"]]
        item["review_result"] = None
        self.reviews.pop(stage, None)
        self.invoke(stage, PRODUCERS[stage], self.inputs(stage), revision={"previous": previous, "target_ids": ids, "feedback": feedback})

    def reviewed_stage(self, stage):
        while True:
            review = self.invoke(stage, JUDGES[stage], self.inputs(stage, reviewer=True), reviewer=True)
            if review["status"] == "passed":
                return
            self.state["stages"][stage]["status"] = "revision_required"
            self.save("评审未通过，准备定向返工")
            self.revision(stage, self.failed_ids(stage, review), review)

    def invalidate_from(self, stage):
        order = [*PRODUCTION_ORDER, "final_review"]
        for name in order[order.index(stage):]:
            if self.state["stages"][name]["output"] or self.state["stages"][name]["review_result"]:
                self.state["stages"][name]["status"] = "invalidated"
                self.state["stages"][name]["review_result"] = None
            self.reviews.pop(name, None)
        self.save("上游稿件需要修改，关联下游评审已失效")

    def revise_final(self, review):
        targets = {stage: set() for stage in PRODUCTION_ORDER}
        copies = {v["variant_id"]: (i["idea_id"], i["candidate_id"]) for i in self.outputs["copy"]["ideas"] for v in i["variants"]}
        for decision in review["reviews"]:
            if decision["quality"]["passed"]:
                continue
            variant_id = decision["judgement"]["item_id"]
            idea_id, title_id = copies[variant_id]
            for issue in decision["judgement"]["issues"]:
                stage = issue["responsible_stage"]
                targets[stage].add(idea_id if stage == "topic" else title_id if stage == "title" else variant_id)
        stages = [s for s in PRODUCTION_ORDER if targets[s]]
        if not stages:
            self.quality_stop("终审未通过，但缺少可执行的责任阶段。")
        if any(self.state["stages"][s]["quality_revisions"] >= MAX_QUALITY_REVISIONS for s in stages):
            self.quality_stop(f"责任阶段已达到 {MAX_QUALITY_REVISIONS} 次质量返工上限。")
        self.invalidate_from(stages[0])
        changed_ideas, changed_copies = set(), set()
        for stage in PRODUCTION_ORDER[PRODUCTION_ORDER.index(stages[0]):]:
            direct = targets[stage]
            if stage == "topic":
                ids = direct
                changed_ideas.update(ids)
            elif stage == "title":
                ids = direct | {c["candidate_id"] for i in self.outputs[stage]["ideas"] if i["idea_id"] in changed_ideas for c in i["candidates"]}
                changed_ideas.update(i["idea_id"] for i in self.outputs[stage]["ideas"] if any(c["candidate_id"] in ids for c in i["candidates"]))
            elif stage == "copy":
                ids = direct | {v["variant_id"] for i in self.outputs[stage]["ideas"] if i["idea_id"] in changed_ideas for v in i["variants"]}
                changed_copies.update(ids)
            else:
                ids = direct | changed_copies
            if ids:
                self.revision(stage, sorted(ids), review, dependency=not bool(direct))
            if stage in JUDGES:
                before = deepcopy(self.outputs[stage])
                self.reviewed_stage(stage)
                old_items, new_items = revision_items(stage, before), revision_items(stage, self.outputs[stage])
                changed = {key for key in old_items if old_items[key] != new_items[key]}
                if stage == "topic":
                    changed_ideas.update(changed)
                elif stage == "title":
                    changed_ideas.update(new_items[key][0][0] for key in changed)
                    # A new review may select another existing title, even without a rewrite.
                    changed_ideas.update(i["idea_id"] for i in self.outputs[stage]["ideas"] if copies and any(
                        old_idea == i["idea_id"] and old_title != self.reviews["title"]["selected_candidate_ids"][i["idea_id"]]
                        for old_idea, old_title in copies.values()))
                elif stage == "copy":
                    changed_copies.update(changed)

    def assemble(self):
        sample, topic = self.outputs["sample_analysis"], self.outputs["topic"]
        titles = {c["candidate_id"]: c for i in self.outputs["title"]["ideas"] for c in i["candidates"]}
        copies = {i["idea_id"]: i for i in self.outputs["copy"]["ideas"]}
        pictures = {v["variant_id"]: v for v in self.outputs["image"]["variants"]}
        title_scores = {d["judgement"]["item_id"]: d["quality"] for d in self.reviews["title"]["review"]["reviews"]}
        copy_scores = {d["judgement"]["item_id"]: d["quality"] for d in self.reviews["copy"]["reviews"]}
        final_scores = {d["judgement"]["item_id"]: d["quality"] for d in self.reviews["final_review"]["reviews"]}
        ideas = []
        for idea in topic["ideas"]:
            candidate_id = self.reviews["title"]["selected_candidate_ids"][idea["idea_id"]]
            title = titles[candidate_id]
            variants = [{**v, **{k: val for k, val in pictures[v["variant_id"]].items() if k not in ("idea_id", "candidate_id")},
                         "quality": {"copy": copy_scores[v["variant_id"]], "final": final_scores[v["variant_id"]]}}
                        for v in copies[idea["idea_id"]]["variants"]]
            ideas.append({"idea_id": idea["idea_id"], "candidate_id": candidate_id, "title": title["title"], "reason": title["rationale"],
                          "source_note_ids": idea["source_note_ids"], "variants": variants, "quality": {"title": title_scores[candidate_id]}})
        result = {"summary": sample["summary"], "style_summary": sample["style_summary"], "patterns": [p["finding"] for p in sample["topic_patterns"]], "ideas": ideas}
        CreativeAnalysisResult.model_validate(result)  # Keep compatibility with existing UI/report shape.
        return result

    def run(self):
        try:
            self.invoke("sample_analysis", run_sample_analysis, dict(search=self.search, sources=self.sources, options=self.options.model_dump()))
            self.invoke("audience", run_reader_needs, dict(keyword=self.search["keyword"], audience=self.options.audience or "内容创作者",
                        requirements=self.options.instruction or "围绕样本写自然精简的原创内容。", sample_analysis=self.outputs["sample_analysis"]))
            for stage in PRODUCTION_ORDER:
                self.invoke(stage, PRODUCERS[stage], self.inputs(stage))
                if stage in JUDGES:
                    self.reviewed_stage(stage)
            while True:
                review = self.invoke("final_review", run_final_review, self.inputs("final_review"), reviewer=True)
                if review["status"] == "passed":
                    break
                self.state["stages"]["final_review"]["status"] = "revision_required"
                self.save("终审未通过，保留稿件并交回责任阶段")
                self.revise_final(review)
            result = self.assemble()
            self.state.update(status="completed", current_stage=None)
            self.save("多 Agent 创作及评审完成")
            metadata = {**self.state["stages"]["sample_analysis"]["metadata"], "workflow": "multi_agent", "request_count": self.budget.request_count,
                        "quality_review_status": "passed", "disclaimer": "基于本批文字样本，未分析原图；模型评分未经人工校准，不保证事实或流量。发布前请核实原创性、事实及素材授权。"}
            return result, metadata
        except Exception as exc:
            cancelled = isinstance(exc, PipelineCancelled)
            quality_failed = isinstance(exc, CreativeError) and exc.code == "creative_quality_exhausted"
            status = "cancelled" if cancelled else "revision_required" if quality_failed else "failed"
            safe = "任务已停止" if cancelled else str(exc) if isinstance(exc, CreativeError) else "多阶段生成中断，已保存此前稿件和评审。"
            self.state.update(status=status, error_code=getattr(exc, "code", "creative_cancelled" if cancelled else "creative_failed"))
            item = self.state["stages"][self.current]
            item.update(status=status, error=safe)
            if item["attempts"] and item["attempts"][-1]["status"] == "running":
                item["attempts"][-1].update(status="failed", finished_at=utc_now(), error=safe)
            try:
                self.save(safe)
            except PipelineCancelled:
                # Cancellation can arrive while a transport exception is being handled.
                self.state.update(status="cancelled", error_code="creative_cancelled")
                item.update(status="cancelled", error="任务已停止")
                self.save("任务已停止")
                raise
            raise


def analyze_multi_agent(db, user_id, search, sources, options, client_factory, on_state):
    config = CreativeAnalysisOptions.model_validate(options)
    model = owned_text_model(db, user_id, config.model_id)
    complete = partial(request_analysis, client_factory(), model_config=model, api_key=decrypt_text(model.encrypted_api_key))
    result, metadata = CreativeWorkflow(search=search, sources=sources, options=options, complete=complete, on_state=on_state).run()
    return result, {**metadata, "model_name": model.model_name}
