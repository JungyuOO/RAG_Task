from __future__ import annotations

from dataclasses import asdict, dataclass

from app.rag.types import ChatTurn
from app.rag.utils import normalize_text


@dataclass(slots=True)
class TopicCandidate:
    topic_id: str
    score: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class TurnContextResolution:
    resolution_type: str
    topic_id: str | None
    confidence: float
    candidate_topics: list[TopicCandidate]
    needs_clarification: bool = False
    clarification_prompt: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_topics"] = [candidate.to_dict() for candidate in self.candidate_topics]
        return payload


class TurnContextResolver:
    EXPLICIT_SWITCH_MARKERS = ("instead", "back to", "switch to", "다시", "이번엔", "말고", "아까", "이전")
    REFERENT_MARKERS = (
        "that", "this", "those", "it", "그거", "그건", "그 코드", "그 예시", "그 차이", "그중", "바꿔", "로도", "아까 그거",
    )
    CODE_REQUEST_MARKERS = ("yaml", "manifest", "code", "example", "sample", "예시", "코드", "보여")

    def resolve(self, user_message: str, session_topics: list[dict], recent_turns: list[ChatTurn], current_topic_id: str | None = None) -> TurnContextResolution:
        normalized = self._normalize(user_message)
        if not session_topics:
            return TurnContextResolution("new_topic", None, 1.0, [])
        candidates = self._score_topics(normalized, session_topics, current_topic_id)
        if not candidates:
            return TurnContextResolution("new_topic", None, 1.0, [])
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        looks_like_referent = any(marker in normalized for marker in self.REFERENT_MARKERS)
        looks_like_code_request = any(marker in normalized for marker in self.CODE_REQUEST_MARKERS)
        ambiguity_gap = best.score - (second.score if second else 0.0)
        if looks_like_referent and looks_like_code_request and second and ambiguity_gap < 0.15:
            return TurnContextResolution(
                resolution_type="ambiguous",
                topic_id=None,
                confidence=max(best.score, 0.0),
                candidate_topics=candidates[:3],
                needs_clarification=True,
                clarification_prompt=self._build_clarification_prompt(session_topics, candidates[:2]),
            )
        if best.score < 0.2 and not self._looks_like_topic_continuation(normalized, recent_turns):
            return TurnContextResolution("new_topic", None, max(best.score, 0.0), candidates[:3])
        resolution_type = "continue" if best.topic_id == current_topic_id else "switch_existing"
        if current_topic_id is None and best.score < 0.35:
            resolution_type = "new_topic"
        return TurnContextResolution(
            resolution_type=resolution_type,
            topic_id=best.topic_id if resolution_type != "new_topic" else None,
            confidence=max(best.score, 0.0),
            candidate_topics=candidates[:3],
        )

    def _score_topics(self, normalized_message: str, session_topics: list[dict], current_topic_id: str | None) -> list[TopicCandidate]:
        scored: list[TopicCandidate] = []
        for topic in session_topics:
            topic_id = str(topic.get("topic_id") or "")
            label = self._normalize(str(topic.get("topic_label") or ""))
            summary = self._normalize(str(topic.get("summary", {}).get("summary", "")))
            sources = [self._normalize(str(value)) for value in topic.get("sources", []) if value]
            entities = [self._normalize(str(value)) for value in topic.get("entities", []) if value]
            last_user_focus = self._normalize(str(topic.get("last_user_focus") or ""))
            score = 0.0
            reasons: list[str] = []
            if label and label in normalized_message:
                score += 0.55
                reasons.append("label")
            for source in sources[:3]:
                if source and source in normalized_message:
                    score += 0.45
                    reasons.append("source")
            entity_hits = 0
            for entity in entities[:6]:
                if entity and entity in normalized_message:
                    score += 0.2
                    entity_hits += 1
            if entity_hits:
                reasons.append(f"entities:{entity_hits}")
            if last_user_focus and last_user_focus in normalized_message:
                score += 0.25
                reasons.append("focus")
            if summary and any(token in summary for token in normalized_message.split() if len(token) >= 3):
                score += 0.1
                reasons.append("summary")
            if current_topic_id and topic_id == current_topic_id:
                score += 0.12
                reasons.append("current")
            if any(marker in normalized_message for marker in self.EXPLICIT_SWITCH_MARKERS) and topic_id != current_topic_id:
                if label and label in normalized_message:
                    score += 0.2
                    reasons.append("switch")
            scored.append(TopicCandidate(topic_id, round(score, 4), ",".join(reasons) if reasons else "weak"))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    def _looks_like_topic_continuation(self, normalized_message: str, recent_turns: list[ChatTurn]) -> bool:
        if any(marker in normalized_message for marker in self.REFERENT_MARKERS):
            return True
        if len(normalized_message) <= 24 and recent_turns:
            return True
        return False

    def _build_clarification_prompt(self, session_topics: list[dict], candidates: list[TopicCandidate]) -> str:
        labels: list[str] = []
        by_id = {str(topic.get("topic_id")): topic for topic in session_topics}
        for candidate in candidates:
            topic = by_id.get(candidate.topic_id)
            if not topic:
                continue
            label = normalize_text(str(topic.get("topic_label") or "")) or normalize_text(str(topic.get("summary", {}).get("topic_label", "") or ""))
            if label and label not in labels:
                labels.append(label)
        scope_hint = ", ".join(labels[:3]) or "방금 이야기한 항목"
        return f"어느 주제를 말하는지 조금만 더 구체적으로 적어주세요. 예를 들어 {scope_hint} 중 하나를 지정해주시면 바로 이어서 답변하겠습니다."

    def _normalize(self, value: str) -> str:
        return " ".join(normalize_text(value).lower().split())

