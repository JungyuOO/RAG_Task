from __future__ import annotations

import logging
from pathlib import Path

from app.rag.llm import LlmClient

logger = logging.getLogger("rag.agent")


class QueryAgent:
    """사용자 질문을 검색 친화적인 짧은 질의로 정리한다."""

    INTENT_MARKERS = {
        "yaml": ("yaml",),
        "manifest": ("manifest",),
        "code": ("code", "코드"),
        "example": ("example", "sample", "예시"),
        "create": ("create", "생성", "작성"),
        "compare": ("compare", "difference", "diff", "차이", "비교"),
        "explain": ("explain", "설명", "정리"),
        "config": ("설정", "config", "configuration"),
    }
    FILLER_TOKENS = {
        "그럼",
        "거기서",
        "기준으로",
        "알려줘",
        "알려주세요",
        "보여줘",
        "보여주세요",
        "다시",
        "좀",
        "조금",
        "파일",
        "파일로",
    }

    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    async def refine_query(
        self,
        user_message: str,
        context: dict | None = None,
        available_sources: list[str] | None = None,
    ) -> dict:
        source_names = [Path(source).stem for source in (available_sources or [])[:10]]
        source_hint = ", ".join(source_names) if source_names else "없음"

        context_parts: list[str] = []
        if context:
            if context.get("active_topic"):
                context_parts.append(f"현재 토픽: {context['active_topic']}")
            if context.get("selected_sources"):
                context_parts.append(f"참조 문서: {', '.join(context['selected_sources'][:3])}")
        context_text = "\n".join(context_parts) if context_parts else "없음"

        prompt = (
            "너는 RAG 검색 질의 정리기다.\n"
            "목표는 질문을 짧게 만드는 것이 아니라 검색 성공률을 높이면서 사용자 의도를 유지하는 것이다.\n\n"
            "규칙:\n"
            "- 질문에 있는 핵심 주제와 요청 타입을 유지하라.\n"
            "- 요청 타입 예: 설명, 비교, 차이, 예시, 코드, yaml, manifest, 생성, 설정.\n"
            "- 질문에 없는 요청 타입을 새로 추가하지 마라.\n"
            "- 질문에 있는 요청 타입은 제거하지 마라.\n"
            "- 외부 지식이나 새로운 개념을 추가하지 마라.\n"
            "- 후속 질문의 대명사는 맥락으로만 복원하라.\n"
            "- 문서명은 질문과 직접 관련 있을 때만 보강하라.\n"
            "- 답변하지 말고 검색용 질의만 출력하라.\n"
            "- 분석 문장, 사고과정, 설명 문장은 출력하지 마라.\n\n"
            f"문서목록: {source_hint}\n"
            f"맥락: {context_text}\n"
            f"질문: {user_message}\n\n"
            "예시 1)\n"
            "질문: ConfigMap 생성 예시 yaml로 보여줘\n"
            "검색쿼리: ConfigMap 생성 예시 yaml\n"
            "대안1: ConfigMap yaml manifest example\n"
            "대안2: ConfigMap 생성 방법\n"
            "키워드: ConfigMap, 생성, 예시, yaml\n\n"
            "예시 2)\n"
            "질문: ConfigMap과 Secret 차이\n"
            "검색쿼리: ConfigMap Secret 차이 비교\n"
            "대안1: ConfigMap Secret difference\n"
            "대안2: ConfigMap Secret compare\n"
            "키워드: ConfigMap, Secret, 차이, 비교\n\n"
            "예시 3)\n"
            "질문: 아까 그거 다시 설명해줘\n"
            "맥락: 현재 토픽 SCC, 참조 문서: SCC.pdf\n"
            "검색쿼리: SCC 다시 설명\n"
            "대안1: SCC 개념 설명\n"
            "대안2: SCC 요약 설명\n"
            "키워드: SCC, 설명\n\n"
            "출력 형식:\n"
            "검색쿼리: ...\n"
            "대안1: ...\n"
            "대안2: ...\n"
            "키워드: ..."
        )

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            logger.info("[QueryAgent] LLM 응답: %r", response[:200])
            parsed = self._parse_query_response(response, user_message)
            return self._validate_and_normalize(user_message, parsed)
        except Exception as exc:
            logger.warning("[QueryAgent] LLM 호출 실패: %s", exc)
            return {
                "refined_query": self._compact_query_from_user_message(user_message) or user_message.strip(),
                "alternative_queries": [],
                "search_keywords": self._keywords_from_user_message(user_message),
            }

    def _parse_query_response(self, response: str, fallback: str) -> dict:
        refined = fallback.strip()
        alternatives: list[str] = []
        keywords: list[str] = []

        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if line.startswith("검색쿼리:"):
                value = line[len("검색쿼리:"):].strip()
                if value:
                    refined = value
            elif line.startswith("대안1:"):
                value = line[len("대안1:"):].strip()
                if value:
                    alternatives.append(value)
            elif line.startswith("대안2:"):
                value = line[len("대안2:"):].strip()
                if value:
                    alternatives.append(value)
            elif line.startswith("키워드:"):
                value = line[len("키워드:"):].strip()
                if value:
                    keywords = [keyword.strip() for keyword in value.split(",") if keyword.strip()]

        return {
            "refined_query": refined,
            "alternative_queries": alternatives[:2],
            "search_keywords": keywords[:8],
        }

    def _validate_and_normalize(self, user_message: str, parsed: dict) -> dict:
        refined = str(parsed.get("refined_query") or user_message).strip()
        alternatives = [str(item).strip() for item in parsed.get("alternative_queries", []) if str(item).strip()]
        keywords = [str(item).strip() for item in parsed.get("search_keywords", []) if str(item).strip()]

        if not refined:
            refined = user_message.strip()

        if len(refined) > max(len(user_message) * 2, 80):
            refined = user_message.strip()

        user_intents = self._detect_intents(user_message)
        refined_intents = self._detect_intents(refined)

        missing_intents = user_intents - refined_intents
        invented_intents = refined_intents - user_intents

        if missing_intents:
            refined = self._merge_missing_intents(user_message, refined, missing_intents)

        if invented_intents:
            refined = user_message.strip()
            alternatives = []
            keywords = []

        compact = self._compact_query_from_user_message(refined)
        if compact:
            refined = compact

        if not keywords:
            keywords = self._keywords_from_user_message(user_message)

        return {
            "refined_query": refined,
            "alternative_queries": alternatives[:2],
            "search_keywords": keywords[:8],
        }

    def _detect_intents(self, text: str) -> set[str]:
        normalized = text.casefold()
        found: set[str] = set()
        for intent, markers in self.INTENT_MARKERS.items():
            if any(marker.casefold() in normalized for marker in markers):
                found.add(intent)
        return found

    def _merge_missing_intents(self, user_message: str, refined: str, missing_intents: set[str]) -> str:
        merged = refined
        normalized_user = user_message.casefold()
        for intent in missing_intents:
            markers = self.INTENT_MARKERS.get(intent, ())
            for marker in markers:
                if marker.casefold() in normalized_user and marker.casefold() not in merged.casefold():
                    merged = f"{merged} {marker}".strip()
                    break
        return merged

    def _keywords_from_user_message(self, user_message: str) -> list[str]:
        raw_tokens = [token.strip(" ,.?/\\()[]{}:;'\"") for token in user_message.split()]
        keywords: list[str] = []
        for token in raw_tokens:
            if len(token) < 2:
                continue
            if token in self.FILLER_TOKENS:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords[:8]

    def _compact_query_from_user_message(self, user_message: str) -> str:
        raw_tokens = [token.strip(" ,.?/\\()[]{}:;'\"") for token in user_message.split()]
        compact_tokens: list[str] = []
        for token in raw_tokens:
            if len(token) < 2:
                continue
            if token in self.FILLER_TOKENS:
                continue
            normalized = token.casefold()
            if normalized in {"해주세요", "해줘", "주세요"}:
                continue
            if token not in compact_tokens:
                compact_tokens.append(token)
        return " ".join(compact_tokens[:8]).strip()


class JudgeAgent:
    """검색 결과가 질문에 답할 수 있는지 판단한다."""

    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    async def evaluate(
        self,
        user_message: str,
        context_texts: list[str],
        top_score: float,
    ) -> dict:
        if not context_texts:
            return {
                "relevant": False,
                "confidence": "high",
                "clarification_message": "업로드된 문서에서 관련 내용을 찾을 수 없습니다. 다른 질문을 해주시거나, 관련 문서를 업로드해 주세요.",
            }

        context_preview = "\n---\n".join(text[:300] for text in context_texts[:3])
        prompt = (
            "너는 검색 품질 판단기다. 아래 형식만 출력하라.\n\n"
            f"질문: {user_message}\n"
            f"검색점수: {top_score:.4f}\n"
            f"검색내용:\n{context_preview}\n\n"
            "규칙:\n"
            "- 검색 내용이 질문에 답할 수 있으면 적합\n"
            "- 아니면 부적합과 함께 짧은 추가 질문을 제안\n\n"
            "출력 형식:\n"
            "판정: 적합 또는 부적합\n"
            "확신도: high 또는 medium 또는 low\n"
            "추가질문: ..."
        )

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=220,
            )
            logger.info("[JudgeAgent] LLM 응답: %r", response[:200])
            return self._parse_judge_response(response, top_score=top_score)
        except Exception as exc:
            logger.warning("[JudgeAgent] LLM 호출 실패: %s", exc)
            if top_score >= 0.2:
                return {"relevant": True, "confidence": "low", "clarification_message": ""}
            return {
                "relevant": False,
                "confidence": "low",
                "clarification_message": "질문을 조금 더 구체적으로 적어주시면 어떤 내용을 찾아야 하는지 더 정확히 판단할 수 있습니다.",
            }

    def _parse_judge_response(self, response: str, top_score: float = 0.0) -> dict:
        relevant = top_score >= 0.2
        confidence = "medium"
        clarification = ""
        found_verdict = False

        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if line.startswith("판정:"):
                value = line[len("판정:"):].strip().casefold()
                found_verdict = True
                if "부적합" in value or "irrelevant" in value or value == "no":
                    relevant = False
                elif "적합" in value or "relevant" in value or value == "yes":
                    relevant = True
            elif line.startswith("확신도:"):
                value = line[len("확신도:"):].strip().lower()
                if value in {"high", "medium", "low"}:
                    confidence = value
            elif line.startswith("추가질문:"):
                value = line[len("추가질문:"):].strip()
                if value and value not in {"없음", "none"}:
                    clarification = value

        if not found_verdict:
            relevant = top_score >= 0.2

        if not relevant and not clarification:
            clarification = "질문을 조금 더 구체적으로 적어주시면 어떤 내용을 찾아야 하는지 더 정확히 판단할 수 있습니다."

        return {
            "relevant": relevant,
            "confidence": confidence,
            "clarification_message": clarification,
        }
