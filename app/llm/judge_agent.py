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
            response = await self._generate(prompt, 200)
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
        return {"relevant": relevant, "confidence": confidence, "clarification_message": clarification_message}
