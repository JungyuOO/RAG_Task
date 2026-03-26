from __future__ import annotations

import logging
from pathlib import Path

from app.rag.llm import LlmClient

logger = logging.getLogger("rag.agent")


class QueryAgent:
    """사용자 질문을 분석하여 검색에 최적화된 쿼리를 생성하는 에이전트.

    기존 query rewrite(대명사 해소)보다 넓은 범위를 담당한다:
    - 모호한 표현을 구체화
    - 문서 제목/키워드와 매칭될 수 있도록 용어 확장
    - 검색 의도가 불명확하면 여러 후보 쿼리 생성
    """

    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    async def refine_query(self, user_message: str, context: dict | None = None, available_sources: list[str] | None = None) -> dict:
        """사용자 메시지를 분석하여 검색용 쿼리와 메타데이터를 반환한다.

        Returns:
            {
                "refined_query": str,       # 검색에 최적화된 쿼리
                "alternative_queries": list, # 대안 쿼리 (최대 2개)
                "search_keywords": list,    # 핵심 검색 키워드
            }
        """
        source_names = [Path(s).stem for s in (available_sources or [])[:10]]
        source_hint = ", ".join(source_names) if source_names else "없음"

        context_parts: list[str] = []
        if context:
            if context.get("active_topic"):
                context_parts.append(f"현재 토픽: {context['active_topic']}")
            if context.get("selected_sources"):
                context_parts.append(f"참조 문서: {', '.join(context['selected_sources'][:3])}")
        context_text = "\n".join(context_parts) if context_parts else "없음"

        prompt = (
            "쿼리 최적화 에이전트. 설명·분석·사고과정 없이 아래 형식만 출력하라.\n\n"
            "규칙:\n"
            "- 원본 질문의 의도를 반드시 유지하라\n"
            "- 외부 지식(L2/L3, OSI 모델 등)을 절대 추가하지 마라\n"
            "- 문서목록에 있는 문서명과 매칭될 수 있는 변형어만 추가하라\n"
            "- 키워드는 질문에 등장하는 단어 + 문서명 매칭 단어만 사용\n\n"
            f"문서목록: {source_hint}\n"
            f"맥락: {context_text}\n"
            f"질문: {user_message}\n\n"
            "예시)\n"
            "문서목록: 네트워킹, 스토리지, 변수\n"
            "질문: 스토리지 종류 알려줘\n"
            "검색쿼리: 스토리지 종류 유형\n"
            "대안1: 스토리지 분류 개념\n"
            "대안2: 저장소 종류 타입\n"
            "키워드: 스토리지, 저장소, 종류\n\n"
            "출력:\n"
            "검색쿼리: \n"
            "대안1: \n"
            "대안2: \n"
            "키워드: "
        )

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            logger.info("[QueryAgent] LLM 응답: %r", response[:200])
            return self._parse_query_response(response, user_message)
        except Exception as exc:
            logger.warning("[QueryAgent] LLM 호출 실패: %s", exc)
            return {
                "refined_query": user_message.strip(),
                "alternative_queries": [],
                "search_keywords": [],
            }

    def _parse_query_response(self, response: str, fallback: str) -> dict:
        refined = fallback.strip()
        alternatives: list[str] = []
        keywords: list[str] = []

        for line in response.strip().splitlines():
            line = line.strip()
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
                    keywords = [kw.strip() for kw in value.split(",") if kw.strip()]

        return {
            "refined_query": refined,
            "alternative_queries": alternatives,
            "search_keywords": keywords,
        }


class JudgeAgent:
    """검색 결과가 사용자 질문에 적합한지 판단하는 에이전트.

    적합하면 답변 생성을 허용하고, 부적합하면 사용자에게 재질문을 생성한다.
    """

    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    async def evaluate(
        self,
        user_message: str,
        context_texts: list[str],
        top_score: float,
    ) -> dict:
        """검색 결과의 적합성을 판단한다.

        Returns:
            {
                "relevant": bool,            # 적합 여부
                "confidence": str,           # "high", "medium", "low"
                "clarification_message": str # 부적합 시 재질문 메시지 (적합 시 빈 문자열)
            }
        """
        if not context_texts:
            return {
                "relevant": False,
                "confidence": "high",
                "clarification_message": "업로드된 문서에서 관련 내용을 찾을 수 없습니다. 다른 질문을 해주시거나, 관련 문서를 업로드해 주세요.",
            }

        context_preview = "\n---\n".join(text[:300] for text in context_texts[:3])

        prompt = (
            "검색 품질 판단 에이전트. 설명·분석·사고과정 없이 아래 형식만 출력하라.\n\n"
            f"질문: {user_message}\n"
            f"검색점수: {top_score:.4f}\n"
            f"검색내용:\n{context_preview}\n\n"
            "규칙: 검색내용이 질문에 답할 수 있으면 적합, 아니면 부적합+재질문 생성.\n\n"
            "예시)\n"
            "판정: 적합\n"
            "확신도: high\n"
            "재질문: 없음\n\n"
            "출력:\n"
            "판정: \n"
            "확신도: \n"
            "재질문: "
        )

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            logger.info("[JudgeAgent] LLM 응답: %r", response[:200])
            return self._parse_judge_response(response, top_score=top_score)
        except Exception as exc:
            logger.warning("[JudgeAgent] LLM 호출 실패: %s", exc)
            # LLM 실패 시 점수 기반 fallback
            if top_score >= 0.2:
                return {"relevant": True, "confidence": "low", "clarification_message": ""}
            return {
                "relevant": False,
                "confidence": "low",
                "clarification_message": "질문을 좀 더 구체적으로 해주시겠어요? 어떤 문서의 어떤 내용이 궁금하신지 알려주시면 더 정확한 답변을 드릴 수 있습니다.",
            }

    _RELEVANT_KEYWORDS = {"적합", "suitable", "relevant", "yes"}
    _IRRELEVANT_KEYWORDS = {"부적합", "unsuitable", "irrelevant", "no"}

    def _parse_judge_response(self, response: str, top_score: float = 0.0) -> dict:
        relevant = True
        confidence = "medium"
        clarification = ""
        found_verdict = False

        for line in response.strip().splitlines():
            line = line.strip()
            if line.startswith("판정:"):
                value = line[len("판정:"):].strip().lower()
                # bracket placeholder("[suitable/unsuitable]" 등)는 실제 판정이 아니므로 무시
                if value.startswith("[") and value.endswith("]"):
                    continue
                found_verdict = True
                # 부적합 키워드 체크 시 "적합"만 단독 포함된 경우를 잘못 잡지 않도록
                # "부적합"을 먼저 제거한 뒤 "적합"을 확인
                if any(kw in value for kw in self._IRRELEVANT_KEYWORDS):
                    relevant = False
                elif any(kw in value for kw in self._RELEVANT_KEYWORDS):
                    relevant = True
                else:
                    # 파싱 불가 시 점수 기반 fallback
                    relevant = top_score >= 0.2
            elif line.startswith("확신도:"):
                value = line[len("확신도:"):].strip().lower()
                if value in ("high", "medium", "low"):
                    confidence = value
            elif line.startswith("재질문:"):
                value = line[len("재질문:"):].strip()
                if value and value != "없음" and value.lower() != "none":
                    clarification = value

        # 판정 라인 자체를 찾지 못한 경우 점수 기반 fallback
        if not found_verdict:
            relevant = top_score >= 0.2

        if not relevant and not clarification:
            clarification = "질문을 좀 더 구체적으로 해주시겠어요? 어떤 내용이 궁금하신지 알려주시면 더 정확한 답변을 드릴 수 있습니다."

        return {
            "relevant": relevant,
            "confidence": confidence,
            "clarification_message": clarification,
        }
