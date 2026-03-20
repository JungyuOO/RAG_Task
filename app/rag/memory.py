from __future__ import annotations

from contextlib import closing
from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path

from app.rag.types import ChatTurn
from app.rag.utils import normalize_text


DEFAULT_SUMMARY = {
    "topic": "",
    "user_goal": "",
    "recent_documents": [],
    "recent_pages": [],
    "unresolved_questions": [],
    "last_user_message": "",
}

DEFAULT_TOPIC_STATE = {
    "active_topic": "",
    "active_entities": [],
    "selected_sources": [],
    "selected_pages": [],
    "last_retrieval_mode": "",
    "last_answer_citations": [],
    "last_user_focus": "",
    "recent_user_topics": [],
}

_GENERIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "may", "might", "shall", "should",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "my", "your", "his", "its", "our", "their",
    "what", "when", "where", "which", "who", "whom", "how", "why",
    "that", "this", "these", "those", "there", "here",
    "and", "or", "but", "if", "then", "so", "because", "as", "than",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "not", "no", "yes", "all", "some", "any", "each", "every",
    "more", "most", "very", "too", "also", "just", "only",
    "again", "above", "below", "up", "down", "out", "off",
    "explain", "show", "tell", "give", "get", "make", "let",
}


class SessionStore:
    """LLM 히스토리에 의존하지 않는 애플리케이션 수준 세션 메모리.

    SQLite에 대화 턴, 구조화된 요약, 토픽 상태를 저장하고,
    매 턴마다 요약과 토픽을 자동 갱신한다. 질의 재작성(query rewrite)은
    대화 맥락과 엔티티를 사용하여 모호한 후속 질문을 보강한다.
    """

    def __init__(self, db_path: Path, memory_window_turns: int = 6) -> None:
        self.db_path = db_path
        self.memory_window_turns = memory_window_turns
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """SQLite 연결을 생성한다. WAL 모드와 busy_timeout으로 동시성을 확보한다."""
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _connection(self):
        with closing(self._connect()) as connection:
            with connection:
                yield connection

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    topic_state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS session_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_session_turns_session_id
                ON session_turns(session_id, turn_id);
                """
            )
            session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "summary_json" not in session_columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN summary_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "topic_state_json" not in session_columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN topic_state_json TEXT NOT NULL DEFAULT '{}'"
                )

            turn_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(session_turns)").fetchall()
            }
            if "metadata" not in turn_columns:
                connection.execute(
                    "ALTER TABLE session_turns ADD COLUMN metadata TEXT NOT NULL DEFAULT ''"
                )

    def add_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, summary, summary_json, topic_state_json, updated_at)
                VALUES (?, '', '{}', '{}', CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                """,
                (session_id,),
            )
            connection.execute(
                "INSERT INTO session_turns (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (session_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
            )
        self._refresh_summary(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._connection() as connection:
            turn_result = connection.execute(
                "DELETE FROM session_turns WHERE session_id = ?",
                (session_id,),
            )
            session_result = connection.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
        return bool(turn_result.rowcount or session_result.rowcount)

    def recent_turns(self, session_id: str) -> list[ChatTurn]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, metadata
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_id DESC
                LIMIT ?
                """,
                (session_id, self.memory_window_turns),
            ).fetchall()
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in reversed(rows)
        ]

    def summary(self, session_id: str) -> str:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["summary"] if row else ""

    def structured_summary(self, session_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT summary_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return DEFAULT_SUMMARY.copy()
        return self._merge_defaults(DEFAULT_SUMMARY, json.loads(row["summary_json"] or "{}"))

    def topic_state(self, session_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT topic_state_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return DEFAULT_TOPIC_STATE.copy()
        return self._merge_defaults(DEFAULT_TOPIC_STATE, json.loads(row["topic_state_json"] or "{}"))

    def memory_snapshot(self, session_id: str) -> dict:
        return {
            "recent_turns": [turn.to_dict() for turn in self.recent_turns(session_id)],
            "session_summary": self.structured_summary(session_id),
            "topic_state": self.topic_state(session_id),
        }

    def build_rewrite_context(self, session_id: str, user_message: str) -> dict | None:
        """LLM 질의 재작성에 필요한 대화 맥락을 구성한다.

        대화 이력이 없으면 None을 반환하여 재작성이 불필요함을 알린다.
        반환된 dict는 LLM 프롬프트 구성에 사용되며, 하드코딩된 규칙 대신
        LLM이 맥락을 판단하여 질의를 재작성한다.
        """
        recent = self.recent_turns(session_id)
        if not recent:
            return None

        summary = self.structured_summary(session_id)
        topic_state = self.topic_state(session_id)

        # 최근 대화 턴을 간결하게 정리
        conversation_history: list[dict] = []
        for turn in recent[-4:]:
            entry: dict = {"role": turn.role, "content": turn.content[:200]}
            if turn.role == "assistant" and turn.metadata:
                sources = [
                    s.get("file_name", "")
                    for s in turn.metadata.get("source_grounding", [])[:2]
                    if s.get("file_name")
                ]
                if sources:
                    entry["sources"] = sources
            conversation_history.append(entry)

        return {
            "conversation_history": conversation_history,
            "active_topic": str(topic_state.get("active_topic") or summary.get("topic") or ""),
            "active_entities": [str(e) for e in topic_state.get("active_entities", []) if e][:6],
            "selected_sources": [str(s) for s in topic_state.get("selected_sources", []) if s][:3],
            "selected_pages": topic_state.get("selected_pages", [])[:5],
            "last_retrieval_mode": str(topic_state.get("last_retrieval_mode") or ""),
        }

    def export_session(self, session_id: str) -> dict:
        turns = self.all_turns(session_id)
        return {
            "session_id": session_id,
            "summary": self.summary(session_id),
            "memory": self.memory_snapshot(session_id),
            "turns": [turn.to_dict() for turn in turns],
        }

    def last_turn(self, session_id: str) -> ChatTurn | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT role, content, metadata
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return ChatTurn(
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def pending_user_message(self, session_id: str) -> str | None:
        last_turn = self.last_turn(session_id)
        if last_turn and last_turn.role == "user":
            return last_turn.content
        return None

    def all_turns(self, session_id: str) -> list[ChatTurn]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, metadata
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT s.session_id, s.summary, s.summary_json, s.updated_at,
                       (
                           SELECT content
                           FROM session_turns
                           WHERE session_id = s.session_id AND role = 'user'
                           ORDER BY turn_id ASC
                           LIMIT 1
                       ) AS first_user_message,
                       (
                           SELECT content
                           FROM session_turns
                           WHERE session_id = s.session_id AND role = 'user'
                           ORDER BY turn_id DESC
                           LIMIT 1
                       ) AS last_user_message,
                       (
                           SELECT created_at
                           FROM session_turns
                           WHERE session_id = s.session_id AND role = 'user'
                           ORDER BY turn_id DESC
                           LIMIT 1
                       ) AS last_user_at,
                       (
                           SELECT COUNT(*)
                           FROM session_turns
                           WHERE session_id = s.session_id
                       ) AS turn_count
                FROM sessions s
                ORDER BY datetime(s.updated_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        sessions: list[dict] = []
        for row in rows:
            summary_json = self._merge_defaults(DEFAULT_SUMMARY, json.loads(row["summary_json"] or "{}"))
            title_source = row["last_user_message"] or row["first_user_message"] or summary_json["topic"] or "New chat"
            title = normalize_text(str(title_source))[:42] or "New chat"
            summary_text = normalize_text(str(row["summary"] or ""))[:96]
            sessions.append(
                {
                    "session_id": row["session_id"],
                    "title": title,
                    "summary": summary_text,
                    "updated_at": row["updated_at"],
                    "turn_count": int(row["turn_count"] or 0),
                    "last_user_message": normalize_text(str(row["last_user_message"] or ""))[:80],
                    "last_user_at": str(row["last_user_at"] or ""),
                }
            )
        return sessions

    def _refresh_summary(self, session_id: str) -> None:
        turns = self.all_turns(session_id)
        summary_json = self._build_structured_summary(turns)
        topic_state = self._build_topic_state(turns)
        summary = self._stringify_summary(summary_json, topic_state)

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, summary, summary_json, topic_state_json, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE
                SET summary = excluded.summary,
                    summary_json = excluded.summary_json,
                    topic_state_json = excluded.topic_state_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    summary,
                    json.dumps(summary_json, ensure_ascii=False),
                    json.dumps(topic_state, ensure_ascii=False),
                ),
            )

    def _build_structured_summary(self, turns: list[ChatTurn]) -> dict:
        recent_turns = turns[-10:]
        user_turns = [turn for turn in recent_turns if turn.role == "user"]
        assistant_turns = [turn for turn in recent_turns if turn.role == "assistant"]

        recent_documents: list[str] = []
        recent_pages: list[int] = []
        unresolved_questions: list[str] = []
        for turn in assistant_turns[-3:]:
            metadata = turn.metadata or {}
            for source in metadata.get("source_grounding", []):
                file_name = normalize_text(str(source.get("file_name") or ""))
                if file_name and file_name not in recent_documents:
                    recent_documents.append(file_name)
            for page in metadata.get("grounded_pages", []):
                page_number = int(page.get("page_number") or 0)
                if page_number and page_number not in recent_pages:
                    recent_pages.append(page_number)
            if metadata.get("mode") == "general" and turn.content:
                unresolved_questions.append(normalize_text(turn.content[:120]))

        last_user_message = normalize_text(user_turns[-1].content) if user_turns else ""
        user_goal = normalize_text(user_turns[-1].content) if user_turns else ""
        topic = recent_documents[0] if recent_documents else self._extract_topic_from_turns(user_turns)

        return {
            "topic": topic,
            "user_goal": user_goal,
            "recent_documents": recent_documents[:3],
            "recent_pages": recent_pages[:5],
            "unresolved_questions": unresolved_questions[:2],
            "last_user_message": last_user_message,
        }

    def _build_topic_state(self, turns: list[ChatTurn]) -> dict:
        recent_turns = turns[-8:]
        selected_sources: list[str] = []
        selected_pages: list[int] = []
        last_answer_citations: list[dict] = []
        last_retrieval_mode = ""
        active_entities: list[str] = []
        recent_user_topics: list[str] = []
        last_user_focus = ""

        for turn in recent_turns:
            metadata = turn.metadata or {}
            if turn.role == "assistant":
                last_retrieval_mode = str(metadata.get("mode") or last_retrieval_mode)
                for item in metadata.get("source_grounding", []):
                    file_name = normalize_text(str(item.get("file_name") or ""))
                    if file_name and file_name not in selected_sources:
                        selected_sources.append(file_name)
                for item in metadata.get("preview_pages", []):
                    page_number = int(item.get("page_number") or 0)
                    if page_number and page_number not in selected_pages:
                        selected_pages.append(page_number)
                if metadata.get("answer_citations"):
                    last_answer_citations = metadata["answer_citations"][:4]
            else:
                active_entities.extend(self._extract_entities(turn.content))
                focus = self._extract_focus_phrase(turn.content)
                if focus:
                    last_user_focus = focus
                    if focus not in recent_user_topics:
                        recent_user_topics.append(focus)

        active_topic = selected_sources[0] if selected_sources else self._extract_topic_from_turns(
            [turn for turn in recent_turns if turn.role == "user"]
        )
        deduped_entities: list[str] = []
        for entity in active_entities:
            if entity and entity not in deduped_entities:
                deduped_entities.append(entity)

        return {
            "active_topic": active_topic,
            "active_entities": deduped_entities[:6],
            "selected_sources": selected_sources[:3],
            "selected_pages": selected_pages[:5],
            "last_retrieval_mode": last_retrieval_mode,
            "last_answer_citations": last_answer_citations,
            "last_user_focus": last_user_focus,
            "recent_user_topics": recent_user_topics[-4:],
        }

    def _stringify_summary(self, summary_json: dict, topic_state: dict) -> str:
        parts: list[str] = []
        if summary_json.get("topic"):
            parts.append(f"Topic: {summary_json['topic']}")
        if summary_json.get("user_goal"):
            parts.append(f"Goal: {normalize_text(str(summary_json['user_goal']))[:160]}")
        if summary_json.get("recent_documents"):
            parts.append("Docs: " + ", ".join(summary_json["recent_documents"][:3]))
        if summary_json.get("recent_pages"):
            parts.append("Pages: " + ", ".join(str(page) for page in summary_json["recent_pages"][:5]))
        if topic_state.get("active_entities"):
            parts.append("Entities: " + ", ".join(topic_state["active_entities"][:4]))
        if summary_json.get("unresolved_questions"):
            parts.append("Open: " + " | ".join(summary_json["unresolved_questions"][:2]))
        return normalize_text(" ; ".join(parts)[:700])

    def _extract_topic_from_turns(self, user_turns: list[ChatTurn]) -> str:
        for turn in reversed(user_turns):
            entities = self._extract_entities(turn.content)
            if entities:
                return entities[0]
            words = normalize_text(turn.content).split(" ")
            if words:
                return " ".join(words[:5])
        return ""

    def _extract_entities(self, text: str) -> list[str]:
        """텍스트에서 의미 있는 엔티티를 범용적으로 추출한다.

        하드코딩된 도메인 키워드 없이, 토큰의 형태적 특성만으로 판별한다:
        - 대문자로 시작하는 단어 (고유명사, 약어: PV, StorageClass)
        - 대소문자 혼합 단어 (camelCase: hostPath, configMap)
        - 한글이 포함된 토큰 (한국어 명사)
        - 숫자가 포함된 토큰 (버전, 식별자)
        - .pdf로 끝나는 파일명
        일반적인 영어 불용어(관사, 전치사, 대명사 등)는 제외한다.
        """
        normalized = normalize_text(text)
        if not normalized:
            return []
        tokens = [token.strip(".,:;()[]{}!?") for token in normalized.split(" ")]
        entities: list[str] = []

        for token in tokens:
            if len(token) < 2:
                continue
            lowered = token.lower()
            if lowered in _GENERIC_STOPWORDS:
                continue
            # 파일명
            if lowered.endswith(".pdf"):
                entities.append(token)
                continue
            # 숫자 포함 (버전, 식별자)
            if any(char.isdigit() for char in token):
                entities.append(token)
                continue
            # 대문자 시작 (고유명사, 약어: PV, Pod, StorageClass)
            if token[0].isupper():
                entities.append(token)
                continue
            # 대소문자 혼합 (camelCase: hostPath, configMap)
            if any(char.isalpha() for char in token) and any(char.isupper() for char in token[1:]):
                entities.append(token)
                continue
            # 한글 포함 토큰 (한국어 명사)
            if any("\uac00" <= char <= "\ud7a3" for char in token):
                entities.append(token)
                continue
        return entities[:8]

    def _extract_focus_phrase(self, text: str) -> str:
        """사용자 메시지에서 핵심 구문을 추출한다."""
        normalized = normalize_text(text)
        if not normalized:
            return ""
        entities = self._extract_entities(normalized)
        if entities:
            return normalize_text(" ".join(entities[:3]))
        words = [
            word
            for word in normalized.split(" ")
            if len(word) >= 2 and word.lower() not in _GENERIC_STOPWORDS
        ]
        return normalize_text(" ".join(words[:4]))

    def _merge_defaults(self, default: dict, loaded: dict) -> dict:
        merged = default.copy()
        merged.update(loaded or {})
        return merged
