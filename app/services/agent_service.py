from __future__ import annotations

import logging
from pathlib import Path

from app.rag.llm import LlmClient
from app.rag.utils import normalize_query_keywords

logger = logging.getLogger("rag.agent")


class QueryAgent:
    """Search-query refiner for Korean technical RAG queries."""

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
            "- 질문에 들어 있는 핵심 주제와 요청 타입을 유지해라.\n"
            "- 요청 타입 예시: 설명, 비교, 차이, 예시, 코드, yaml, manifest, 생성, 설정.\n"
            "- 질문에 없는 요청 타입을 새로 추가하지 마라.\n"
            "- 질문에 있는 요청 타입을 제거하지 마라.\n"
            "- 없는 기술 개념을 임의로 추가하지 마라.\n"
            "- 후속 질문의 대명사는 문맥이 있을 때만 복원해라.\n"
            "- 문서명은 질문과 직접 관련 있을 때만 보강해라.\n"
            "- 답변하지 말고 검색용 질의만 출력해라.\n"
            "- 분석 문장, 사고과정, 설명 문장은 출력하지 마라.\n\n"
            f"문서목록: {source_hint}\n"
            f"문맥: {context_text}\n"
            f"질문: {user_message}\n\n"
            "예시 1)\n"
            "질문: ConfigMap 생성 예시 yaml로 보여줘\n"
            "검색질의: ConfigMap 생성 예시 yaml\n"
            "대안: ConfigMap yaml manifest example\n"
            "대안: ConfigMap 생성 방법\n"
            "키워드: ConfigMap, 생성, 예시, yaml\n\n"
            "예시 2)\n"
            "질문: ConfigMap과 Secret 차이\n"
            "검색질의: ConfigMap Secret 차이 비교\n"
            "대안: ConfigMap Secret difference\n"
            "대안: ConfigMap Secret compare\n"
            "키워드: ConfigMap, Secret, 차이, 비교\n\n"
            "예시 3)\n"
            "질문: 아까 그거 다시 설명해줘\n"
            "문맥: 현재 토픽 SCC, 참조 문서: SCC.pdf\n"
            "검색질의: SCC 다시 설명\n"
            "대안: SCC 개념 설명\n"
            "대안: SCC 요약 설명\n"
            "키워드: SCC, 설명\n\n"
            "출력 형식:\n"
            "검색질의: ...\n"
            "대안: ...\n"
            "대안: ...\n"
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
            if line.startswith("검색질의:"):
                value = line[len("검색질의:"):].strip()
                if value:
                    refined = value
            elif line.startswith("대안:"):
                value = line[len("대안:"):].strip()
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
        raw_keywords = [str(item).strip() for item in parsed.get("search_keywords", []) if str(item).strip()]

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
            raw_keywords = []

        compact = self._compact_query_from_user_message(refined)
        if compact:
            refined = compact

        keywords = normalize_query_keywords(refined, raw_keywords)
        if not keywords:
            keywords = self._keywords_from_user_message(user_message)

        return {
            "refined_query": refined,
            "alternative_queries": self._normalize_alternative_queries(alternatives)[:2],
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
        filtered_tokens: list[str] = []
        for token in raw_tokens:
            if len(token) < 2:
                continue
            if token in self.FILLER_TOKENS:
                continue
            filtered_tokens.append(token)
        return normalize_query_keywords(user_message, filtered_tokens)[:8]

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

    def _normalize_alternative_queries(self, alternatives: list[str]) -> list[str]:
        normalized_alternatives: list[str] = []
        seen: set[str] = set()
        for alternative in alternatives:
            compact = self._compact_query_from_user_message(alternative)
            value = compact or alternative.strip()
            if not value:
                continue
            dedupe_key = value.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized_alternatives.append(value)
        return normalized_alternatives


class JudgeAgent:
    """Judge whether retrieved context is relevant enough to answer."""

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
                "clarification_message": "업로드한 문서에서 관련 내용을 찾을 수 없습니다. 다른 질문을 하거나 관련 문서를 업로드해 주세요.",
            }

        prompt = (
            "너는 검색 품질 판단기다.\n"
            "사용자 질문과 검색된 문맥이 실제로 맞는지 판단해라.\n"
            "문맥이 질문과 맞으면 relevant=yes, 아니면 relevant=no로 답해라.\n"
            "문맥이 애매하면 사용자가 더 구체적으로 물어볼 수 있는 짧은 안내 문장을 작성해라.\n\n"
            f"질문: {user_message}\n"
            f"검색 점수: {top_score:.4f}\n\n"
            "문맥:\n"
            + "\n\n".join(context_texts[:3])
            + "\n\n출력 형식:\n"
            "판정: yes|no\n"
            "확신도: low|medium|high\n"
            "추가질문: ..."
        )

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            logger.info("[JudgeAgent] LLM 응답: %r", response[:200])
            return self._parse_judge_response(response)
        except Exception as exc:
            logger.warning("[JudgeAgent] LLM 호출 실패: %s", exc)
            return {
                "relevant": top_score >= 0.25,
                "confidence": "medium",
                "clarification_message": "질문을 조금 더 구체적으로 적어 주시면 관련 문서를 다시 확인하겠습니다.",
            }

    def _parse_judge_response(self, response: str) -> dict:
        relevant = False
        confidence = "medium"
        clarification_message = ""

        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            lowered = line.casefold()
            if lowered.startswith("판정:"):
                value = line.split(":", 1)[1].strip().casefold()
                relevant = value in {"yes", "true", "relevant"}
            elif lowered.startswith("확신도:"):
                value = line.split(":", 1)[1].strip().casefold()
                if value in {"low", "medium", "high"}:
                    confidence = value
            elif lowered.startswith("추가질문:"):
                clarification_message = line.split(":", 1)[1].strip()

        if not relevant and not clarification_message:
            clarification_message = "질문 범위를 조금 더 좁혀 주시면 문서 안에서 다시 찾아보겠습니다."

        return {
            "relevant": relevant,
            "confidence": confidence,
            "clarification_message": clarification_message,
        }
