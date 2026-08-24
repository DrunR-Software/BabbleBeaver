"""
Prompt performance tracking.

Records one row per chatbot turn so the prompt dashboard can show how the
current prompt template is performing over time: latency, token usage, model
version, and prompt version. Stored in SQLite alongside the other dev logs.
"""

import sqlite3
from datetime import datetime, timedelta


class PromptMetricsLogger:
    def __init__(self, db_path="chatbot.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_metrics (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    session_id TEXT,
                    prompt_version TEXT,
                    model_version TEXT,
                    user_message TEXT,
                    response_preview TEXT,
                    token_count INTEGER,
                    latency_ms INTEGER,
                    num_search_results INTEGER,
                    user_context TEXT,
                    error TEXT
                )
            """)
            # Add user_context to pre-existing tables that predate this column.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(prompt_metrics)")}
            if "user_context" not in cols:
                conn.execute("ALTER TABLE prompt_metrics ADD COLUMN user_context TEXT")

    def log(self, *, session_id, prompt_version, model_version, user_message,
            response_preview, token_count, latency_ms, num_search_results,
            user_context=None, error=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO prompt_metrics
                   (created_at, session_id, prompt_version, model_version,
                    user_message, response_preview, token_count, latency_ms,
                    num_search_results, user_context, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.utcnow().isoformat(), session_id, prompt_version,
                 model_version, user_message, (response_preview or "")[:500],
                 token_count, latency_ms, num_search_results,
                 user_context or "", error),
            )
            conn.commit()

    def summary(self, since_days=7):
        """Aggregate stats grouped by prompt_version for the recent window."""
        cutoff = (datetime.utcnow() - timedelta(days=since_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT
                       COALESCE(prompt_version, 'unknown') AS prompt_version,
                       COUNT(*) AS turns,
                       AVG(latency_ms) AS avg_latency_ms,
                       AVG(token_count) AS avg_tokens,
                       SUM(token_count) AS total_tokens,
                       SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors
                   FROM prompt_metrics
                   WHERE created_at >= ?
                   GROUP BY prompt_version
                   ORDER BY turns DESC""",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    def recent(self, limit=50):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT created_at, session_id, prompt_version, model_version,
                          user_message, response_preview, token_count,
                          latency_ms, num_search_results, user_context, error
                   FROM prompt_metrics
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
