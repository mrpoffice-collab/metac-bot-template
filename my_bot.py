"""
mrpofficebot's own ideas live in this file, one at a time.

The stock Metaculus bot stays untouched in main.py, so every idea here can be
tested against it with compare_bots.py before it is trusted.

Idea 1 (2026-11-19): before answering, look up how often things like this have
happened before. The stock bot only reads one summary of the latest news.
"""

import logging
from datetime import datetime

from forecasting_tools import MetaculusQuestion

from main import FallTemplateBot2026

logger = logging.getLogger(__name__)


class MrpofficeBot(FallTemplateBot2026):
    """The stock bot plus a history lookup."""

    async def run_research(self, question: MetaculusQuestion) -> str:
        news = await super().run_research(question)
        history = await self._look_up_history(question)
        return (
            "HOW OFTEN THINGS LIKE THIS HAVE HAPPENED BEFORE "
            "(start your thinking from this, then adjust for the latest news):\n"
            f"{history}\n\n"
            "LATEST NEWS:\n"
            f"{news}"
        )

    async def _look_up_history(self, question: MetaculusQuestion) -> str:
        prompt = (
            "You are a research assistant helping a forecaster. Do NOT give a forecast.\n"
            "Your one job: find how often things like this have happened before.\n\n"
            f"Question: {question.question_text}\n\n"
            f"How it will be decided: {question.resolution_criteria}\n\n"
            f"{question.fine_print or ''}\n\n"
            f"Today is {datetime.now().strftime('%Y-%m-%d')}.\n\n"
            "Answer in under 200 words:\n"
            "1. Name one to three groups of similar past situations.\n"
            "2. For each, give the numbers: how many times the outcome happened out of "
            "how many chances, or the typical past value and how fast it changes, "
            "and over which years.\n"
            "3. Say where each number comes from.\n"
            "4. If there is no useful history, say so plainly. Never invent a number."
        )
        try:
            history = await self.get_llm("researcher", "llm").invoke(prompt)
        except Exception as error:  # a failed lookup must not sink the whole forecast
            logger.warning(f"History lookup failed for {question.page_url}: {error}")
            return "No history could be looked up."
        logger.info(f"History for URL {question.page_url}:\n{history}")
        return history
