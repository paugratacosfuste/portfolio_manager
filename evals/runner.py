"""Eval runner — feeds golden cases through the live pipelines and
scores each output with a Haiku judge. Writes results to
`evals/results/<iso-timestamp>.json`.

Usage
-----
  # Run all cases
  PYTHONPATH=. python -m evals.runner

  # Run a subset by pattern
  PYTHONPATH=. python -m evals.runner --pattern triage

  # Run without real Anthropic calls (for smoke-testing the harness)
  PYTHONPATH=. python -m evals.runner --dry-run

Exit code
---------
  0 if all cases verdict == "pass"
  1 if any case is "fail" or hit a hard must_contain/must_reject
  2 if any case errored out before judging
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.cases import EVAL_CASES, EvalCase
from evals.judge import JudgeScore, judge_output


RESULTS_DIR = Path("evals/results")


@dataclass
class CaseResult:
    case_id: str
    pattern: str
    latency_sec: float
    verdict: str  # pass / partial / fail / error
    score: int | None
    must_contain_ok: bool
    must_reject_ok: bool
    output_preview: str
    judge_reasoning: str
    error: str | None


# ── Pipeline dispatch ────────────────────────────────────────────────────────


def _stringify_debate(result: Any) -> str:
    return (
        f"TOPIC: {result.topic}\n\n"
        f"BULL ({result.bull.conviction:.2f}): {result.bull.claim}\n"
        f"  evidence: {'; '.join(result.bull.evidence)}\n"
        f"BEAR ({result.bear.conviction:.2f}): {result.bear.claim}\n"
        f"  evidence: {'; '.join(result.bear.evidence)}\n"
        f"MACRO ({result.macro.conviction:.2f}): {result.macro.claim}\n"
        f"  evidence: {'; '.join(result.macro.evidence)}\n\n"
        f"CIO VERDICT: {result.verdict.verdict} "
        f"(confidence {result.verdict.confidence:.2f})\n"
        f"Reasoning: {result.verdict.reasoning}\n"
        f"Recommended action: {result.verdict.recommended_action}\n"
    )


def _stringify_triage(result: Any) -> str:
    recs = "\n".join(
        f"  - [{r.kind}] {r.label}: {r.reason}" for r in result.recommendations
    )
    return (
        f"SEVERITY: {result.severity} (confidence {result.confidence:.2f})\n"
        f"DIFFERENTIAL: {', '.join(result.differential)}\n"
        f"RED FLAGS: {', '.join(result.red_flags) or '(none)'}\n"
        f"RECOMMENDATIONS:\n{recs}\n"
        f"SUMMARY: {result.plain_language_summary}\n"
    )


def _stringify_contract(result: Any) -> str:
    risks = "\n".join(
        f"  - [{r.severity}] {r.label}: {r.description}" for r in result.risks
    )
    return (
        f"SUMMARY: {result.plain_summary}\n"
        f"PARTIES: {', '.join(result.parties) or '(not extracted)'}\n"
        f"KEY OBLIGATIONS: {', '.join(result.key_obligations)}\n"
        f"RISKS:\n{risks or '  (none)'}\n"
        f"NEGOTIATION: {', '.join(result.negotiation_suggestions) or '(none)'}\n"
    )


def _stringify_travel(plan: Any, transcript: list[str]) -> str:
    days_str = []
    for d in plan.days:
        acts = "; ".join(f"{a.time_of_day}: {a.title} ({a.location})" for a in d.activities)
        days_str.append(f"  Day {d.day_index} [{d.date_label}] "
                        f"{d.weather_summary} — {acts}")
    return (
        f"DESTINATION: {plan.destination}\n"
        f"TRAVELLER: {plan.traveller_profile}\n"
        f"DAYS:\n" + "\n".join(days_str) + "\n"
        f"ESTIMATED COST: ${plan.estimated_total_cost_usd:.0f}\n"
        f"PACKING TIPS: {', '.join(plan.packing_tips) or '(none)'}\n"
        f"TRANSCRIPT ({len(transcript)} tool calls):\n  " + "\n  ".join(transcript)
    )


def run_case(case: EvalCase, client: Any) -> tuple[str, Any]:
    """Dispatch a case to its pipeline, return (string_output, raw_result)."""
    if case.pattern == "debate":
        from utils.debate import run_debate
        result = run_debate(
            topic=case.inputs["topic"],
            context=case.inputs["context"],
            client=client,
        )
        return _stringify_debate(result), result

    if case.pattern == "triage":
        from utils.cross_domain import run_triage
        result = run_triage(
            presentation=case.inputs["presentation"],
            patient_context=case.inputs["patient_context"],
            client=client,
        )
        return _stringify_triage(result), result

    if case.pattern == "contract":
        from utils.cross_domain import run_contract_review
        result = run_contract_review(
            clause_text=case.inputs["clause_text"],
            client=client,
            perspective=case.inputs.get("perspective", "buyer"),
        )
        return _stringify_contract(result), result

    if case.pattern == "travel":
        from utils.cross_domain import run_travel_agent
        plan, transcript = run_travel_agent(
            destination=case.inputs["destination"],
            days=case.inputs["days"],
            traveller_profile=case.inputs["traveller_profile"],
            client=client,
        )
        return _stringify_travel(plan, transcript), (plan, transcript)

    raise ValueError(f"Unknown pattern: {case.pattern}")


# ── Dry-run mock (lets the harness execute without API calls) ───────────────


def _dry_run_output(case: EvalCase) -> str:
    return (
        f"[DRY RUN] Would have executed pattern {case.pattern} "
        f"with inputs {list(case.inputs.keys())}."
    )


# ── Main loop ────────────────────────────────────────────────────────────────


def _check_hard_constraints(
    output: str, case: EvalCase
) -> tuple[bool, bool]:
    must_contain_ok = all(s.lower() in output.lower() for s in case.must_contain)
    must_reject_ok = not any(s.lower() in output.lower() for s in case.must_reject)
    return must_contain_ok, must_reject_ok


def run_all(
    cases: list[EvalCase], client: Any | None, dry_run: bool
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        print(f"[{case.pattern}] {case.id} ...", flush=True)
        t0 = time.time()
        try:
            if dry_run or client is None:
                output = _dry_run_output(case)
                judge_score: JudgeScore | None = None
                verdict = "dry_run"
                score = None
                reasoning = "(dry run — not judged)"
            else:
                output, _raw = run_case(case, client)
                judge_score = judge_output(case.expects, output, client)
                verdict = judge_score.verdict
                score = judge_score.score
                reasoning = judge_score.reasoning

            if dry_run:
                must_contain_ok, must_reject_ok = True, True
            else:
                must_contain_ok, must_reject_ok = _check_hard_constraints(
                    output, case
                )
                if not must_contain_ok or not must_reject_ok:
                    verdict = "fail"

            preview = output if len(output) < 800 else output[:800] + "…"
            results.append(CaseResult(
                case_id=case.id,
                pattern=case.pattern,
                latency_sec=round(time.time() - t0, 2),
                verdict=verdict,
                score=score,
                must_contain_ok=must_contain_ok,
                must_reject_ok=must_reject_ok,
                output_preview=preview,
                judge_reasoning=reasoning,
                error=None,
            ))
            print(f"  -> {verdict} ({score}/5)" if score is not None
                  else f"  -> {verdict}", flush=True)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=3)
            results.append(CaseResult(
                case_id=case.id,
                pattern=case.pattern,
                latency_sec=round(time.time() - t0, 2),
                verdict="error",
                score=None,
                must_contain_ok=False,
                must_reject_ok=False,
                output_preview="",
                judge_reasoning="",
                error=f"{exc}\n{tb}",
            ))
            print(f"  -> ERROR: {exc}", flush=True)
    return results


def _summarize(results: list[CaseResult]) -> dict:
    passes = sum(1 for r in results if r.verdict == "pass")
    partials = sum(1 for r in results if r.verdict == "partial")
    fails = sum(1 for r in results if r.verdict == "fail")
    errors = sum(1 for r in results if r.verdict == "error")
    dry = sum(1 for r in results if r.verdict == "dry_run")
    return {
        "total": len(results),
        "pass": passes,
        "partial": partials,
        "fail": fails,
        "error": errors,
        "dry_run": dry,
    }


def _write_results(
    results: list[CaseResult], summary: dict, out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts}.json"
    payload = {
        "timestamp_utc": ts,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM pattern evals.")
    parser.add_argument(
        "--pattern",
        choices=sorted({c.pattern for c in EVAL_CASES}),
        help="Run only cases matching this pattern.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call Anthropic — just exercise the harness machinery.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to write the result JSON.",
    )
    args = parser.parse_args(argv)

    cases = [c for c in EVAL_CASES if not args.pattern or c.pattern == args.pattern]
    if not cases:
        print("No cases matched.", file=sys.stderr)
        return 2

    client = None
    if not args.dry_run:
        try:
            from utils.ai_advisor import client as anthropic_client
            client = anthropic_client
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load Anthropic client: {exc}", file=sys.stderr)
            return 2
        if client is None:
            print(
                "Anthropic client not initialized. Set ANTHROPIC_API_KEY or "
                "use --dry-run.",
                file=sys.stderr,
            )
            return 2

    results = run_all(cases, client, dry_run=args.dry_run)
    summary = _summarize(results)
    out_path = _write_results(results, summary, args.out_dir)

    print("\nSUMMARY:", json.dumps(summary, indent=2))
    print(f"Results written to: {out_path}")

    if summary["fail"] > 0 or summary["error"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
