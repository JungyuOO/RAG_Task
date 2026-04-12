from __future__ import annotations

from app.llm.base_agent import BaseAgent


ANSWER_REWRITE_SYSTEM_PROMPT = """You rewrite extractive, document-grounded draft answers into concise Korean.

Rules:
1. Use only the information already present in the draft answer.
2. Do not add facts, assumptions, editing commentary, or explanations about rewriting.
3. Keep the answer in Korean.
4. Preserve technical terms, CLI commands, YAML fields, object names, and identifiers as-is.
5. Prefer short paragraphs or short bullet points.
6. Do not add citations, source tags, markdown separators, or editorial preambles.
7. Never say things like:
   - "초안"
   - "다시 작성했습니다"
   - "자연스러운 한국어로"
   - "문맥이 중복되고 어색한 부분"
   - "핵심 의미와 내용을 유지하면서"
   - any similar rewrite/editor meta commentary
8. Return only the final answer body. Start immediately with the content itself.
"""


class AnswerRewriteAgent(BaseAgent):
    def __init__(self, llm_client) -> None:
        super().__init__(llm_client, system_prompt=ANSWER_REWRITE_SYSTEM_PROMPT)

    async def rewrite(self, user_message: str, draft_answer: str) -> str:
        prompt = (
            f"사용자 질문:\n{user_message.strip()}\n\n"
            f"초안 답변:\n{draft_answer.strip()}\n\n"
            "초안의 사실과 구조를 벗어나지 말고, 메타 설명 없이 바로 최종 답변만 자연스럽게 정리해 주세요."
        )
        return (await self._generate(prompt, 320)).strip()
