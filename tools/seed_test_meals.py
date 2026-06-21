"""Seed a few logged meals into user_service for a test user, then they show up
in Kai's context (Recently Logged Meals + nutrition scoring).

Usage:
    source venv/bin/activate
    python3 tools/seed_test_meals.py <core_user_uuid>
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import httpx  # noqa: E402

from user_context import USER_SERVICE_URL, _mint_service_token  # noqa: E402

NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

MEALS = [
    {"name": "Grilled Salmon Bowl", "calories": 540, "protein": 42, "carbs": 38, "fat": 18, "dietary": "pescatarian"},
    {"name": "Greek Yogurt & Berries", "calories": 210, "protein": 18, "carbs": 24, "fat": 5},
    {"name": "Chicken Power Plate", "calories": 610, "protein": 48, "carbs": 45, "fat": 20},
]


async def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    core_user_uuid = sys.argv[1]
    token = _mint_service_token(core_user_uuid)
    if not token:
        print("Could not mint service token (USER_SERVICE_TOKEN_SECRET missing).")
        return 1

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=USER_SERVICE_URL, headers=headers, timeout=10) as client:
        for meal in MEALS:
            payload = {**meal, "core_user_uuid": core_user_uuid, "logged_at": NOW}
            r = await client.post("/meallog/", json=payload)
            status = "OK" if r.status_code in (200, 201) else f"FAIL {r.status_code}"
            print(f"  {status}: {meal['name']}  {r.text[:120] if r.status_code >= 400 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
