"""Render a real report using only synthetic workflow responses; print to stdout."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root), str(root / "tests" / "backend")]

from backend.app.services.creative_report import render_report
from test_creative_sample_agent import SEARCH, sources
from test_creative_workflow import WorkflowCompletion, workflow


def main():
    model = WorkflowCompletion(fail_stage="final_review", final_owner="image")
    w = workflow(model, count=3, variants=2)
    result, metadata = w.run()
    w.state["stages"]["final_review"]["review_result"]["reviews"][0]["quality"]["reviewer_notes"] = '<script>window.__reportXss = true</script>'
    document = {
        "schema_version": 3, "task_id": 12, "created_at": "2026-10-21T12:00:00",
        "search": SEARCH, "analysis": w.options.model_dump(), "result": result,
        "analysis_metadata": metadata, "sources": sources(), "failures": [], "pipeline": w.state,
    }
    print(render_report(document))


if __name__ == "__main__":
    main()
