"""
Side-by-side test: the stock Metaculus bot against mrpofficebot, on the same questions.

How the score works: real answers take weeks or months, so for each question
the crowd's current forecast on Metaculus stands in as the best available
answer. Each bot is scored on how close it got (the library's "expected
baseline score": higher is better, 0 means no better than always saying 50%).

Nothing is posted to Metaculus.

Run:  uv run python compare_bots.py --questions 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone

# Importing main first also quiets noisy libraries and loads the keys.
from main import FallTemplateBot2026
from my_bot import MrpofficeBot
from bot_settings import make_llms

from forecasting_tools import MetaculusClient, MonetaryCostManager

logger = logging.getLogger(__name__)

OLD = "Stock bot"
NEW = "mrpofficebot"


def make_bot(bot_class, predictions_per_question: int):
    return bot_class(
        research_reports_per_question=1,
        predictions_per_research_report=predictions_per_question,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        llms=make_llms(),
    )


async def run_one_bot(bot, questions) -> dict:
    with MonetaryCostManager() as cost_manager:
        started = time.time()
        reports = await bot.forecast_questions(questions, return_exceptions=True)
        minutes = (time.time() - started) / 60
        cost = cost_manager.current_usage
    good = [r for r in reports if not isinstance(r, BaseException)]
    errors = [str(r) for r in reports if isinstance(r, BaseException)]
    scores = [r.expected_baseline_score for r in good if r.expected_baseline_score is not None]
    return {
        "reports": good,
        "errors": errors,
        "cost": cost,
        "minutes": minutes,
        "score": (sum(scores) / len(scores)) if scores else None,
    }


async def run_both(questions, predictions: int) -> dict:
    results = {}
    for label, bot_class in [(OLD, FallTemplateBot2026), (NEW, MrpofficeBot)]:
        logger.info(f"===== Running {label} on {len(questions)} questions =====")
        results[label] = await run_one_bot(make_bot(bot_class, predictions), questions)
    return results


def pct(value) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def build_rows(results: dict) -> list[dict]:
    rows: dict[int, dict] = {}
    for label, result in results.items():
        for report in result["reports"]:
            question = report.question
            row = rows.setdefault(
                question.id_of_question,
                {
                    "question": question.question_text,
                    "url": question.page_url,
                    "crowd": report.community_prediction,
                },
            )
            row[label] = report.prediction
    return list(rows.values())


def write_summary(results: dict, rows: list[dict], asked: int, predictions: int) -> str:
    both = [r for r in rows if r.get(OLD) is not None and r.get(NEW) is not None and r["crowd"] is not None]
    gap_old = [abs(r[OLD] - r["crowd"]) for r in both]
    gap_new = [abs(r[NEW] - r["crowd"]) for r in both]
    new_closer = sum(1 for a, b in zip(gap_old, gap_new) if b < a - 0.005)
    old_closer = sum(1 for a, b in zip(gap_old, gap_new) if a < b - 0.005)
    ties = len(both) - new_closer - old_closer

    lines = [
        "# Old bot against new bot",
        "",
        f"Run on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC. "
        f"{asked} questions asked, {predictions} answers per question, nothing posted to Metaculus.",
        "",
        "| Bot | Score (higher is better) | Questions answered | Failed | Cost |",
        "|---|---|---|---|---|",
    ]
    for label, result in results.items():
        score, cost = result["score"], result["cost"]
        lines.append(
            f"| {label} | {'-' if score is None else f'{score:.2f}'} | "
            f"{len(result['reports'])} | {len(result['errors'])} | "
            f"{'-' if cost is None else f'${cost:.2f}'} |"
        )
    lines += [""]
    if both:
        lines += [
            f"On the {len(both)} questions both bots answered:",
            "",
            f"- Average distance from the crowd: {OLD} {sum(gap_old) / len(both) * 100:.1f} points, "
            f"{NEW} {sum(gap_new) / len(both) * 100:.1f} points (smaller is better).",
            f"- Closer to the crowd: {NEW} on {new_closer}, {OLD} on {old_closer}, about the same on {ties}.",
            "",
        ]
    lines += [
        f"Caution: {asked} questions is too few to separate two similar bots. "
        "The library's own guide says 100 or more is a decent start.",
        "",
        "| Question | Crowd | Stock bot | mrpofficebot | Closer |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["question"]):
        old, new, crowd = r.get(OLD), r.get(NEW), r["crowd"]
        closer = "-"
        if old is not None and new is not None and crowd is not None:
            a, b = abs(old - crowd), abs(new - crowd)
            closer = NEW if b < a - 0.005 else OLD if a < b - 0.005 else "same"
        title = r["question"].replace("|", "/")
        lines.append(f"| [{title}]({r['url']}) | {pct(crowd)} | {pct(old)} | {pct(new)} | {closer} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the stock bot with mrpofficebot")
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--predictions", type=int, default=3, help="answers per question, combined into one")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    questions = MetaculusClient().get_benchmark_questions(args.questions)
    results = asyncio.run(run_both(questions, args.predictions))

    rows = build_rows(results)
    summary = write_summary(results, rows, args.questions, args.predictions)
    with open("compare_summary.md", "w", encoding="utf-8") as handle:
        handle.write(summary)
    with open("compare_results.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "questions_asked": args.questions,
                "predictions_per_question": args.predictions,
                "bots": [
                    {
                        "name": label,
                        "score": result["score"],
                        "answered": len(result["reports"]),
                        "failed": len(result["errors"]),
                        "errors": result["errors"][:5],
                        "cost": result["cost"],
                        "minutes": result["minutes"],
                    }
                    for label, result in results.items()
                ],
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print("\n" + "=" * 30 + " COMPARISON " + "=" * 30 + "\n")
    print(summary)


if __name__ == "__main__":
    main()
