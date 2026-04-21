from __future__ import annotations

import unittest

from apps.api.schemas.chat import CopilotChatSourceItem
from apps.api.rag.generation.answer_planner import AnswerPlanner
from apps.api.rag.generation.citation_grounding import CitationGroundingValidator


class AnswerPlannerTests(unittest.TestCase):
    def test_prefers_diverse_sources_when_relevance_is_similar(self) -> None:
        planner = AnswerPlanner()
        sources = [
            CopilotChatSourceItem(
                source_type="doc",
                label="nodes.md · Pod spec",
                source_path="official/en/nodes.md",
                chunk_id="chunk-1",
                score=0.92,
                metadata={
                    "section_title": "About pods",
                    "preview_text": "Pod specs define containers and metadata.",
                    "synthesis_text": "Pod specs define containers, metadata, and restart policy.",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="pipelines.md · CI pipeline",
                source_path="official/en/pipelines.md",
                chunk_id="chunk-2",
                score=0.88,
                metadata={
                    "section_title": "Creating pipelines",
                    "preview_text": "Pipelines define CI/CD tasks and resources.",
                    "synthesis_text": "Pipelines define CI/CD tasks, resources, and execution flow.",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="nodes.md · Read operations",
                source_path="official/en/nodes.md",
                chunk_id="chunk-3",
                score=0.87,
                metadata={
                    "section_title": "Read operations",
                    "preview_text": "Use oc get node to inspect nodes.",
                    "synthesis_text": "Use oc get node and oc describe node to inspect node state.",
                },
            ),
        ]

        plan = planner.plan(
            message="pod configuration and ci cd pipeline explain",
            sources=sources,
        )

        self.assertEqual(len(plan.sources), 2)
        self.assertEqual(plan.sources[0].source_path, "official/en/nodes.md")
        self.assertEqual(plan.sources[1].source_path, "official/en/pipelines.md")
        planned_indexes = {index for group in plan.paragraph_source_indexes for index in group}
        self.assertIn(0, planned_indexes)
        self.assertIn(1, planned_indexes)

    def test_citation_validator_removes_mismatched_citation(self) -> None:
        validator = CitationGroundingValidator()
        sources = [
            CopilotChatSourceItem(
                source_type="doc",
                label="nodes.md · About pods",
                source_path="official/en/nodes.md",
                chunk_id="chunk-1",
                metadata={
                    "section_title": "About pods",
                    "preview_text": "Pod specs define containers and metadata.",
                    "synthesis_text": "Pod specs define containers, metadata, and restart policy.",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="pipelines.md · Creating pipelines",
                source_path="official/en/pipelines.md",
                chunk_id="chunk-2",
                metadata={
                    "section_title": "Creating pipelines",
                    "preview_text": "Pipelines define CI/CD tasks and resources.",
                    "synthesis_text": "Pipelines define CI/CD tasks, resources, and execution flow.",
                },
            ),
        ]

        validated = validator.validate(
            "Pod specs define containers and metadata.[1]\n\nPod specs define containers and metadata.[2]",
            sources,
        )

        self.assertIn("[1]", validated)
        self.assertNotIn("[2]", validated)

    def test_prefers_title_overlap_for_specific_section(self) -> None:
        planner = AnswerPlanner()
        sources = [
            CopilotChatSourceItem(
                source_type="doc",
                label="auth.md · OIDC",
                source_path="official/en/authentication_and_authorization.md",
                chunk_id="chunk-1",
                score=0.96,
                metadata={
                    "section_title": "Chapter 8. Enabling direct authentication with an external OIDC identity provider",
                    "preview_text": "OIDC providers integrate with OpenShift.",
                    "synthesis_text": "OIDC providers integrate with OpenShift.",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="auth.md · OAuth token list",
                source_path="official/en/authentication_and_authorization.md",
                chunk_id="chunk-2",
                score=0.92,
                metadata={
                    "section_title": "5.1. Listing user-owned OAuth access tokens",
                    "preview_text": "Use oc get useroauthaccesstokens to list tokens.",
                    "synthesis_text": "Use oc get useroauthaccesstokens to list user-owned OAuth access tokens.",
                },
            ),
        ]

        plan = planner.plan(
            message="유저가 가진 oauth 토큰 리스트 뽑는 방법",
            sources=sources,
        )

        self.assertEqual(plan.sources[0].metadata["section_title"], "5.1. Listing user-owned OAuth access tokens")

    def test_multi_clause_question_keeps_multiple_relevant_sources(self) -> None:
        planner = AnswerPlanner()
        sources = [
            CopilotChatSourceItem(
                source_type="doc",
                label="jenkins.md · Cross project",
                source_path="official/en/jenkins.md",
                chunk_id="chunk-1",
                score=0.94,
                metadata={
                    "section_title": "1.3. Providing Jenkins cross project access",
                    "preview_text": "Grant Jenkins access across projects.",
                    "synthesis_text": "Grant Jenkins access across projects.",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="jenkins.md · Cross volume",
                source_path="official/en/jenkins.md",
                chunk_id="chunk-2",
                score=0.91,
                metadata={
                    "section_title": "1.4. Jenkins cross volume mount points",
                    "preview_text": "Configure Jenkins cross volume mount points.",
                    "synthesis_text": "Configure Jenkins cross volume mount points.",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="storage.md · Distractor",
                source_path="official/en/storage.md",
                chunk_id="chunk-3",
                score=0.89,
                metadata={
                    "section_title": "1.1. Storage overview",
                    "preview_text": "Storage glossary terms.",
                    "synthesis_text": "Storage glossary terms.",
                },
            ),
        ]

        plan = planner.plan(
            message="jenkins cross project access 랑 cross volume mount 차이도 설명해줘",
            sources=sources,
        )

        planned_sections = [source.metadata["section_title"] for source in plan.sources]
        self.assertIn("1.3. Providing Jenkins cross project access", planned_sections)
        self.assertIn("1.4. Jenkins cross volume mount points", planned_sections)


if __name__ == "__main__":
    unittest.main()

