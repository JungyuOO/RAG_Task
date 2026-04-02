from __future__ import annotations

import json

def persist_summary(cursor, session_id: str, summary: str, summary_json: dict, topic_state: dict) -> None:
    cursor.execute(
        """
        INSERT INTO sessions (session_id, summary, summary_json, topic_state_json, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(session_id) DO UPDATE
        SET summary = EXCLUDED.summary,
            summary_json = EXCLUDED.summary_json,
            topic_state_json = EXCLUDED.topic_state_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            session_id,
            summary,
            json.dumps(summary_json, ensure_ascii=False),
            json.dumps(topic_state, ensure_ascii=False),
        ),
    )
