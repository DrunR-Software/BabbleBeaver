"""Show the FULL context/prompt Kai receives, without GCP or main.py.

Reproduces the /chatbot assembly:
  user_context_block = USER HEALTH PROFILE  (user_service)
                     + KAI HEALTH SCORES    (scoring.py via upsert_daily_scores)
then builds the same prompt template as generate_from_v2(). The restaurant
vector search needs BigQuery, so sample menu results stand in for it.

Usage:
    source venv/bin/activate
    python3 tools/test_full_context.py <core_user_uuid> ["user query"] [bearer_token]
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from user_context import fetch_user_context, format_user_context  # noqa: E402
from daily_scores import upsert_daily_scores  # noqa: E402

# Stand-in for vector_search_restaurants() (BigQuery-backed; GCP not available).
SAMPLE_RESULTS = [
    {
        "dish": "Grilled Salmon Bowl",
        "restaurant": "Cedar & Sea",
        "summary": "Wild salmon over greens with quinoa and lemon-herb dressing.",
        "category": "Bowls",
        "calories": 540,
        "protein": "42g",
        "carbohydrates": "38g",
    },
    {
        "dish": "Chicken Power Plate",
        "restaurant": "Greenline Kitchen",
        "summary": "Roasted chicken breast, roasted veg, brown rice.",
        "category": "Plates",
        "calories": 610,
        "protein": "48g",
        "carbohydrates": "45g",
    },
    {
        "dish": "Lentil Buddha Bowl",
        "restaurant": "Root Cafe",
        "summary": "Spiced lentils, kale, sweet potato, tahini.",
        "category": "Bowls",
        "calories": 480,
        "protein": "22g",
        "carbohydrates": "60g",
    },
]


def build_menu_context(search_results: list) -> str:
    context = ""
    for idx, result in enumerate(search_results[:3], 1):
        context += f"{idx}. {result.get('dish', 'Unknown dish')} - {result.get('restaurant', 'Unknown restaurant')}\n"
        context += f"   {result.get('summary', '')}\n"
        if result.get("category"):
            context += f"   Category: {result['category']}\n"
        parts = []
        if result.get("calories"):
            parts.append(f"{result['calories']} cal")
        if result.get("protein"):
            parts.append(f"{result['protein']} protein")
        if result.get("carbohydrates"):
            parts.append(f"{result['carbohydrates']} carbs")
        if parts:
            context += f"   Nutrition: {', '.join(parts)}\n"
    return context


def build_prompt(user_query: str, menu_context: str, user_context_block: str) -> str:
    user_context_section = f"\n    {user_context_block}\n" if user_context_block else ""
    return f"""
    You are Kai, DrunR's AI nutrition guide. Talk like a warm, encouraging friend
    who happens to know food and health well — natural and conversational, never
    clinical or robotic. Use "you," skip jargon, and don't restate the question.
{user_context_section}
    User request:
    {user_query}

    Available menu results:
    {menu_context}

    [... mode-selection + 3-part recommendation rules omitted for readability ...]
    """


async def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    core_user_uuid = sys.argv[1]
    user_query = sys.argv[2] if len(sys.argv) > 2 else "What should I get for lunch?"
    bearer_token = sys.argv[3] if len(sys.argv) > 3 else None

    ctx = await fetch_user_context(core_user_uuid, bearer_token=bearer_token)

    profile_block = format_user_context(ctx)
    score_block = upsert_daily_scores(ctx, core_user_uuid)

    if profile_block and score_block:
        user_context_block = f"{profile_block}\n\n{score_block}"
    else:
        user_context_block = profile_block or score_block

    print("=" * 70)
    print("USER CONTEXT BLOCK (health profile + scores injected into prompt)")
    print("=" * 70)
    print(user_context_block or "(empty — no context!)")

    menu_context = build_menu_context(SAMPLE_RESULTS)
    full_prompt = build_prompt(user_query, menu_context, user_context_block)

    print("\n" + "=" * 70)
    print("FULL PROMPT SENT TO KAI (sample menu results stand in for BigQuery)")
    print("=" * 70)
    print(full_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
