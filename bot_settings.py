"""
Which AI models the bot uses.

Kept in one place so the live bot and the side-by-side test always use the same
ones. Low-cost choices on purpose while learning: change one thing at a time
and measure before spending more.
"""

from forecasting_tools import GeneralLlm


def make_llms() -> dict:
    return {
        # The forecaster: reads the research and gives the answer.
        "default": GeneralLlm(
            model="openrouter/anthropic/claude-haiku-5.5",
            temperature=0.3,
            timeout=60,
            allowed_tries=2,
        ),
        # Shortens long research when needed.
        "summarizer": "openrouter/openai/gpt-4o-mini",
        # Looks things up on the web.
        "researcher": "openrouter/perplexity/sonar",
        # Turns the forecaster's written answer into a clean number.
        "parser": "openrouter/openai/gpt-4o-mini",
    }
