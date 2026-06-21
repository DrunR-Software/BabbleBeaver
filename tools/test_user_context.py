"""Manual test for the user_service context path (no GCP / no main.py needed).

Usage:
    source venv/bin/activate
    python3 tools/test_user_context.py <core_user_uuid> [bearer_token]

If a bearer_token is omitted, a short-lived service JWT is minted from
USER_SERVICE_TOKEN_SECRET (same path /chatbot uses for service calls).
Loads .env so USER_SERVICE_URL / USER_SERVICE_TOKEN_SECRET are picked up.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from user_context import fetch_user_context, format_user_context  # noqa: E402


async def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    core_user_uuid = sys.argv[1]
    bearer_token = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Fetching context for core_user_uuid={core_user_uuid} ...\n")
    ctx = await fetch_user_context(core_user_uuid, bearer_token=bearer_token)

    if ctx is None:
        print("fetch_user_context returned None (service down, auth failed, or no profile).")
        return 1

    print("Raw context dict:")
    print(ctx)
    print("\n--- Formatted prompt block ---")
    block = format_user_context(ctx)
    print(block or "(empty block — no usable fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
