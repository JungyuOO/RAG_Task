from __future__ import annotations

import json

from app.llm.base_agent import BaseAgent


ANSWER_SYSTEM_PROMPT = """당신은 RAG 시스템의 답변 생성 에이전트입니다.
검색된 문서 컨텍스트를 기반으로 정확하고 자연스러운 한국어 답변을 생성합니다.

규칙:
1. 반드시 제공된 컨텍스트 기반으로만 답변
2. 문장마다 해당 정보의 출처를 인라인 인용 태그로 표시
3. 여러 문서에서 정보를 종합할 경우 각각의 출처를 명시
4. 컨텍스트에 없는 정보는 문서에서 찾을 수 없다고 명시

대화 컨텍스트:
{conversation_context}

검색된 문서 컨텍스트:
{context_items}
"""


PROCEDURE_CHECK_PROMPT = """다음 검색 결과에 단계별 절차가 포함되어 있는지 분석하세요.
절차가 있다면 총 단계 수와 순차/일괄 선택 제안 메시지를 생성하세요.

검색 결과:
{context_items}

응답은 항상 JSON 객체로만 반환하세요.
"""


class AnswerAgent(BaseAgent):
    def __init__(self, llm_client) -> None:
        super().__init__(llm_client, system_prompt=ANSWER_SYSTEM_PROMPT)
        self._procedure_prompt = PROCEDURE_CHECK_PROMPT

    async def generate_stream(self, user_message: str, context_items: list, conversation_context: str):
        context_str = json.dumps(context_items, ensure_ascii=False, default=str)
        async for token in self.stream(
            user_message,
            context_items=context_str,
            conversation_context=conversation_context,
        ):
            yield token

    async def check_procedure(self, user_message: str, context_items: list) -> dict:
        context_str = json.dumps(context_items, ensure_ascii=False, default=str)
        original_prompt = self.system_prompt
        self.system_prompt = self._procedure_prompt
        try:
            result = await self.call(user_message, context_items=context_str)
        finally:
            self.system_prompt = original_prompt

        if "has_procedure" not in result:
            return {"has_procedure": False, "total_steps": 0, "offer_message": ""}
        return result
