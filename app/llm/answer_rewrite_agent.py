from __future__ import annotations

from app.llm.base_agent import BaseAgent


ANSWER_REWRITE_SYSTEM_PROMPT = """You rewrite extractive document-grounded notes into concise Korean.

Rules:
1. Use only the provided draft answer.
2. Do not add facts that are not already present.
3. Keep the answer in Korean.
4. Preserve technical terms, CLI, YAML fields, and identifiers as-is.
5. Prefer short paragraphs or short bullet points.
6. Do not add source tags or citations yourself.
7. Do not say things like "초안", "정리하면", "제공해주신 초안", "자연스러운 한국어로", or any meta commentary.
8. Return only the final answer body. Start immediately with the content itself.
9. Do not output markdown separators such as "---".
"""


class AnswerRewriteAgent(BaseAgent):
    def __init__(self, llm_client) -> None:
        super().__init__(llm_client, system_prompt=ANSWER_REWRITE_SYSTEM_PROMPT)

    async def rewrite(self, user_message: str, draft_answer: str) -> str:
        prompt = (
            f"사용자 질문:\n{user_message.strip()}\n\n"
            f"초안 답변:\n{draft_answer.strip()}\n\n"
            "위 초안의 의미를 바꾸지 말고, 자연스러운 한국어 답변으로만 다시 정리해 주세요."
        )
        return (await self._generate(prompt, 320)).strip()
