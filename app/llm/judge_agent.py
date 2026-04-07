from __future__ import annotations

import logging

from app.llm.base_agent import BaseAgent

logger = logging.getLogger("rag.agent")


class JudgeAgent(BaseAgent):
    """Judge whether retrieved context is relevant enough to answer."""

    async def evaluate(self, user_message: str, context_texts: list[str], top_score: float) -> dict:
        if not context_texts:
            return {
                "relevant": False,
                "confidence": "medium", 
                "clarification_message": (
                    "관련 페이지 후보는 있었지만 현재 기준으로는 답변 근거가 충분하지 않았습니다."
                    "질문에 제품명, 버전, 리소스명(Pod/Service/Deployment 등)을 함께 적어 주시면 다시 확인하겠습니다."
                ),
            }

        prompt = (
            "너는 검색 품질 판단기다.\n"
            "사용자 질문과 검색된 문맥이 실제로 맞는지 판단해라.\n\n"
            "판정 규칙:\n"
            "1) 질문이 기본 개념 설명(예: Pod, Service, Deployment, Route, Node)이면,\n"
            "   문맥이 정확히 동일 문장을 포함하지 않아도 정의, 설명, overview, 구성 요소 설명이 있으면 relevant=yes로 본다.\n"
            "2) 질문과 직접 무관한 절차, 설정값, 다른 리소스만 설명하면 relevant=no로 본다.\n"
            "3) 문맥이 일부만 맞더라도 사용자의 질문에 핵심 답변을 시작할 수 있으면 relevant=yes로 본다.\n"
            "4) 너무 애매하면 relevant=no로 하되, 사용자가 더 잘 물을 수 있는 짧은 안내를 작성한다.\n\n"
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
            response = await self._generate(prompt, 200)
            logger.info("[JudgeAgent] LLM 응답: %r", response[:200])
            return self._parse_judge_response(response)
        except Exception as exc:
            logger.warning("[JudgeAgent] LLM 호출 실패: %s", exc)

            fallback_threshold = 0.08
            return {
                "relevant": top_score >= fallback_threshold,
                "confidence": "low",
                "clarification_message": (
                    "질문을 리소스 명(Pod/Service/Deployment 등)과 함께 조금 더 구체적으로 작성해 주시면 "
                    "관련 문서를 정확히 찾겠습니다."
                ),
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
        return {"relevant": relevant, "confidence": confidence, "clarification_message": clarification_message}
