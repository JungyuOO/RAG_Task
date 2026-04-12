from __future__ import annotations

import unittest

from app.rag.answer import AnswerGenerator


class GenericCommandSelectionTests(unittest.TestCase):
    def test_generic_namespace_query_prefers_project_command_over_specific_deployment_command(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "final_retrieval_score": 0.45,
                "chunk": {
                    "source_path": "/docs/security_and_compliance.md",
                    "page_number": 1,
                    "metadata": {"page_start": 1, "page_end": 1},
                    "text": "```text $ oc get deployment -n <istio_csr_project_name> ```",
                },
            },
            {
                "final_retrieval_score": 0.40,
                "chunk": {
                    "source_path": "/docs/cli_tools.md",
                    "page_number": 1,
                    "metadata": {"page_start": 1, "page_end": 1},
                    "text": "```text $ oc project ```",
                },
            },
        ]

        answer = generator.build_extractive_code_answer(
            context_items,
            user_message="namespace 확인 명령어 뭐야?",
            query_interpretation={
                "format_constraints": ["cli"],
                "response_shape": "code",
                "generic_command_query": True,
                "normalized_keywords": ["namespace", "확인", "명령어"],
            },
        )

        self.assertIsNotNone(answer)
        self.assertIn("oc project", answer)
        self.assertNotIn("oc get deployment -n <istio_csr_project_name>", answer.split("```")[1])

    def test_generic_pod_query_prefers_simple_pod_list_over_specific_selector(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "final_retrieval_score": 0.52,
                "chunk": {
                    "source_path": "/docs/security_and_compliance.md",
                    "page_number": 1,
                    "metadata": {"page_start": 1, "page_end": 1},
                    "text": "```text $ oc get pods -l app.kubernetes.io/name=cert-manager -n cert-manager ```",
                },
            },
            {
                "final_retrieval_score": 0.45,
                "chunk": {
                    "source_path": "/docs/cli_tools.md",
                    "page_number": 1,
                    "metadata": {"page_start": 1, "page_end": 1},
                    "text": "```text $ oc get pods ```",
                },
            },
        ]

        answer = generator.build_extractive_code_answer(
            context_items,
            user_message="pod 확인하는 명령어 뭐야?",
            query_interpretation={
                "resources": ["pod"],
                "format_constraints": ["cli"],
                "response_shape": "code",
                "generic_command_query": True,
                "normalized_keywords": ["pod", "확인", "명령어"],
            },
        )

        self.assertIsNotNone(answer)
        self.assertIn("oc get pods", answer)
        self.assertNotIn("app.kubernetes.io/name=cert-manager", answer)


if __name__ == "__main__":
    unittest.main()
