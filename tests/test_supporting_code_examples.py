from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.rag.answer import AnswerGenerator
from app.rag.pipeline_streaming import ChatTurnDeps, ChatTurnOrchestrator


class SupportingCodeExampleTests(unittest.TestCase):
    def test_build_supporting_code_example_returns_structured_payload(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "chunk": {
                    "source_path": "/docs/sample.pdf",
                    "page_number": 7,
                    "metadata": {"page_start": 7, "page_end": 7, "code_language": "bash", "html_anchor": "page-7", "primary_block_anchor": "page-7-block-1"},
                    "text": "```bash\noc get pod\n```",
                }
            }
        ]
        appendix = generator.build_supporting_code_example(context_items)
        self.assertIsNotNone(appendix)
        self.assertEqual(appendix["type"], "code")
        self.assertEqual(appendix["title"], "예시 코드")
        self.assertEqual(appendix["language"], "bash")
        self.assertEqual(appendix["page_start"], "7")
        self.assertEqual(appendix["source_path"], "/docs/sample.pdf")
        self.assertEqual(appendix["html_anchor"], "page-7")
        self.assertEqual(appendix["block_anchor"], "page-7-block-1")
        self.assertIn("oc get pod", appendix["content"])

    def test_build_supporting_examples_skips_when_answer_already_has_code(self) -> None:
        deps = ChatTurnDeps(
            detect_non_korean_query=None,
            session_repository=None,
            should_skip_procedure_shortcut=None,
            detect_procedure_followup=None,
            build_procedure_followup_answer=None,
            looks_like_step_navigation_without_state=None,
            resolve_turn_context=None,
            domain_guard_state=None,
            prepare_retrieval_state=None,
            resolve_answer_route=None,
            interleave_context_items_by_source=None,
            build_context_blocks=None,
            ensure_topic_for_resolution=None,
            build_answer_cache_key=None,
            canonical_cache_query=None,
            build_policy_answer=None,
            build_missing_extractive_answer=None,
            select_code_example_context_items=lambda *_args, **_kwargs: [
                {
                    "chunk": {
                        "source_path": "/docs/sample.pdf",
                        "page_number": 1,
                        "metadata": {"page_start": 1, "page_end": 1, "code_language": "bash"},
                        "text": "```bash\noc get pod\n```",
                    }
                }
            ],
            resolve_requested_resource_kinds=lambda _qi: set(),
            prefer_block_type_items=lambda *_args, **_kwargs: [],
            finalize_answer=None,
            store_assistant_turn=None,
            build_llm_failure_fallback=None,
            get_prompt_composer=None,
            answer_service=AnswerGenerator(retrieval_service=None),
            answer_cache_repository=None,
            answer_rewrite_agent=None,
            llm=None,
        )
        orchestrator = ChatTurnOrchestrator(deps)
        examples = orchestrator._build_supporting_examples(
            answer="이미 코드가 있습니다.\n```bash\noc get pod\n```",
            deps=deps,
            user_message="pod 생성 방식 설명해줘",
            query_interpretation={"response_shape": "text"},
            ordered_context_items=[],
            selected_context_items=[],
        )
        self.assertEqual(examples, [])

    def test_build_extractive_code_answer_extracts_inline_oc_command(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "chunk": {
                    "source_path": "/docs/cli_tools.md",
                    "page_number": 12,
                    "metadata": {"page_start": 12, "page_end": 12},
                    "text": "Procedure To list pods run the following command: ```text $ oc get pods -o wide ``` Example output ...",
                }
            }
        ]

        answer = generator.build_extractive_code_answer(context_items)

        self.assertIsNotNone(answer)
        self.assertIn("oc get pods -o wide", answer)

    def test_build_extractive_code_answer_uses_generic_yaml_templates_for_generic_query(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "final_retrieval_score": 0.4,
                "chunk": {
                    "source_path": "/docs/sample.md",
                    "page_number": 3,
                    "metadata": {"page_start": 3, "page_end": 3},
                    "text": "```text $ oc get pod test -o yaml ```",
                },
            }
        ]

        answer = generator.build_extractive_code_answer(
            context_items,
            requested_resource_kinds={"pod"},
            user_message="yaml 보려면 무슨 명령어 써?",
            query_interpretation={
                "generic_command_query": True,
                "format_constraints": ["yaml", "cli"],
                "resources": ["pod"],
            },
        )

        self.assertIsNotNone(answer)
        self.assertIn("oc get pod <pod_name> -o yaml", answer)
        self.assertIn("oc describe pod <pod_name>", answer)

    def test_build_extractive_code_answer_prefers_pod_templates_over_inherited_namespace_resource(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "final_retrieval_score": 0.4,
                "chunk": {
                    "source_path": "/docs/sample.md",
                    "page_number": 3,
                    "metadata": {"page_start": 3, "page_end": 3},
                    "text": "```text $ oc get pod test -o yaml ```",
                },
            }
        ]

        answer = generator.build_extractive_code_answer(
            context_items,
            requested_resource_kinds={"pod", "namespace"},
            user_message="yaml 보려면 무슨 명령어 써?",
            query_interpretation={
                "generic_command_query": True,
                "format_constraints": ["yaml", "cli"],
                "resources": ["pod", "namespace"],
            },
        )

        self.assertIsNotNone(answer)
        self.assertIn("oc get pod <pod_name> -o yaml", answer)
        self.assertNotIn("oc get namespace <name> -o yaml", answer)

    def test_build_extractive_code_answer_preserves_procedure_label_command_pairs_for_specific_query(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "final_retrieval_score": 0.7,
                "chunk": {
                    "source_path": "/docs/etcd.md",
                    "page_number": 12,
                    "metadata": {"page_start": 12, "page_end": 12},
                    "text": (
                        "Procedure\n\n"
                        "Check the status of etcd pods.\n\n"
                        "openshift-etcd\n"
                        "$ oc get pods -n openshift-etcd\n"
                        "openshift-etcd-operator\n"
                        "$ oc get pods -n openshift-etcd-operator\n"
                    ),
                },
            }
        ]

        answer = generator.build_extractive_code_answer(
            context_items,
            requested_resource_kinds={"pod"},
            user_message="etcd pod 상태 확인 명령어 알려줘",
            query_interpretation={
                "generic_command_query": False,
                "format_constraints": ["cli"],
                "resources": ["pod"],
                "response_shape": "code",
            },
        )

        self.assertIsNotNone(answer)
        self.assertIn("openshift-etcd", answer)
        self.assertIn("oc get pods -n openshift-etcd", answer)
        self.assertIn("openshift-etcd-operator", answer)

    def test_build_extractive_code_answer_can_merge_adjacent_procedure_chunks(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "final_retrieval_score": 0.7,
                "chunk": {
                    "source_path": "/docs/support.md",
                    "page_number": 116,
                    "metadata": {"page_start": 116, "page_end": 116},
                    "text": "Procedure\nCheck the status of etcd pods.\n```text\nopenshift-etcd\n```",
                },
            },
            {
                "final_retrieval_score": 0.68,
                "chunk": {
                    "source_path": "/docs/support.md",
                    "page_number": 116,
                    "metadata": {"page_start": 116, "page_end": 116},
                    "text": "```text\n$ oc get pods -n openshift-etcd\n```\n```text\nopenshift-etcd-operator\n```",
                },
            },
            {
                "final_retrieval_score": 0.66,
                "chunk": {
                    "source_path": "/docs/support.md",
                    "page_number": 116,
                    "metadata": {"page_start": 116, "page_end": 116},
                    "text": "```text\n$ oc get pods -n openshift-etcd-operator\n```",
                },
            },
        ]

        answer = generator.build_extractive_code_answer(
            context_items,
            requested_resource_kinds={"pod"},
            user_message="etcd pod 상태 확인 명령어 알려줘",
            query_interpretation={
                "generic_command_query": False,
                "format_constraints": ["cli"],
                "resources": ["pod"],
                "response_shape": "code",
            },
        )

        self.assertIsNotNone(answer)
        self.assertIn("openshift-etcd", answer)
        self.assertIn("oc get pods -n openshift-etcd", answer)
        self.assertIn("openshift-etcd-operator", answer)
        self.assertIn("oc get pods -n openshift-etcd-operator", answer)

    def test_build_extractive_code_answer_can_pull_pairs_from_neighbor_markdown_pages(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "support.md"
            source_path.write_text(
                "## Page 113\n"
                "If you experience etcd issues during installation, you can check etcd pod status.\n"
                "## Page 114\n"
                "Procedure\n"
                "Check the status of etcd pods.\n"
                "```text\nopenshift-etcd\n```\n"
                "```text\n$ oc get pods -n openshift-etcd\n```\n"
                "```text\nopenshift-etcd-operator\n```\n"
                "```text\n$ oc get pods -n openshift-etcd-operator\n```\n",
                encoding="utf-8",
            )
            context_items = [
                {
                    "final_retrieval_score": 0.7,
                    "chunk": {
                        "source_path": str(source_path),
                        "page_number": 113,
                        "metadata": {"page_start": 113, "page_end": 113},
                        "text": "If you experience etcd issues during installation, you can check etcd pod status.",
                    },
                }
            ]

            answer = generator.build_extractive_code_answer(
                context_items,
                requested_resource_kinds={"pod"},
                user_message="etcd pod 상태 확인 명령어 알려줘",
                query_interpretation={
                    "generic_command_query": False,
                    "format_constraints": ["cli"],
                    "resources": ["pod"],
                    "response_shape": "code",
                    "normalized_keywords": ["etcd", "pod", "상태", "확인", "명령어"],
                },
            )

        self.assertIsNotNone(answer)
        self.assertIn("openshift-etcd", answer)
        self.assertIn("oc get pods -n openshift-etcd", answer)
        self.assertIn("openshift-etcd-operator", answer)

    def test_build_generic_yaml_compare_answer_uses_resource_specific_placeholders(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)

        answer = generator.build_generic_yaml_compare_answer(resource_kind="pod")

        self.assertIn("oc get pod <pod_name> -o yaml", answer)
        self.assertIn("oc describe pod <pod_name>", answer)

    def test_build_extractive_code_answer_prefers_oc_status_for_generic_status_query(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "final_retrieval_score": 0.3,
                "chunk": {
                    "source_path": "/docs/cli_tools.md",
                    "page_number": 1,
                    "metadata": {"page_start": 1, "page_end": 1},
                    "text": "```text $ oc status ```",
                },
            }
        ]

        answer = generator.build_extractive_code_answer(
            context_items,
            user_message="현재 상태 확인 명령어랑 실제 결과 같이 알려줘",
            query_interpretation={
                "generic_command_query": True,
                "format_constraints": ["cli"],
                "normalized_keywords": ["현재", "상태", "확인", "명령어"],
            },
        )

        self.assertIsNotNone(answer)
        self.assertIn("oc status", answer)
        self.assertIn("oc get pods", answer)


if __name__ == "__main__":
    unittest.main()
