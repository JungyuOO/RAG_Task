from __future__ import annotations

from dataclasses import dataclass

from apps.api.schemas.chat import CopilotChatSourceItem
from apps.api.rag.query.query_features import answer_source_budget, tokenize_query


@dataclass(slots=True)
class PlannedEvidence:
    source_index: int
    source: CopilotChatSourceItem
    context: str
    relevance: float
    token_overlap: int
    title_overlap: int


@dataclass(slots=True)
class AnswerPlan:
    evidences: list[PlannedEvidence]
    evidence_groups: list[list[int]]

    def _ordered_evidence_indexes(self) -> tuple[list[int], dict[int, int]]:
        ordered: list[int] = []
        seen: set[int] = set()
        for group in self.evidence_groups:
            for index in group:
                if index in seen:
                    continue
                seen.add(index)
                ordered.append(index)
        remap = {original: new_index for new_index, original in enumerate(ordered)}
        return ordered, remap

    @property
    def sources(self) -> list[CopilotChatSourceItem]:
        ordered, _remap = self._ordered_evidence_indexes()
        return [self.evidences[index].source for index in ordered]

    @property
    def ordered_evidences(self) -> list[PlannedEvidence]:
        ordered, _remap = self._ordered_evidence_indexes()
        return [self.evidences[index] for index in ordered]

    @property
    def paragraph_source_indexes(self) -> list[list[int]]:
        _ordered, remap = self._ordered_evidence_indexes()
        remapped: list[list[int]] = []
        for group in self.evidence_groups:
            mapped_group = [remap[index] for index in group if index in remap]
            if mapped_group:
                remapped.append(mapped_group)
        return remapped

    @property
    def paragraph_count(self) -> int:
        return max(1, len(self.evidence_groups))


class AnswerPlanner:
    def plan(
        self,
        *,
        message: str,
        sources: list[CopilotChatSourceItem],
        max_evidences: int = 3,
    ) -> AnswerPlan:
        budget = min(max(answer_source_budget(message), 1), max_evidences)
        doc_candidates: list[PlannedEvidence] = []
        query_tokens = set(tokenize_query(message))

        for index, source in enumerate(sources):
            if source.source_type != "doc":
                continue
            context = self._source_context(source)
            if not context:
                continue
            token_overlap, title_overlap, relevance = self._relevance_score(source, context, query_tokens)
            if relevance <= 0:
                continue
            doc_candidates.append(
                PlannedEvidence(
                    source_index=index,
                    source=source,
                    context=context,
                    relevance=relevance,
                    token_overlap=token_overlap,
                    title_overlap=title_overlap,
                )
            )

        if not doc_candidates:
            return AnswerPlan(evidences=[], evidence_groups=[])

        ranked = sorted(
            doc_candidates,
            key=lambda candidate: (candidate.title_overlap, candidate.token_overlap, candidate.relevance),
            reverse=True,
        )
        top_candidate = ranked[0]
        if top_candidate.title_overlap > 0:
            ranked = [
                candidate
                for candidate in ranked
                if candidate.title_overlap > 0 or candidate.relevance >= (top_candidate.relevance * 0.88)
            ]
        selected = [ranked[0]]

        while len(selected) < min(budget, len(ranked)):
            best_candidate: PlannedEvidence | None = None
            best_score = float("-inf")
            for candidate in ranked:
                if candidate in selected:
                    continue
                diversity_bonus = 0.0
                if all(candidate.source.source_path != chosen.source.source_path for chosen in selected):
                    diversity_bonus += 0.18
                if all(
                    str(candidate.source.metadata.get("section_title") or "")
                    != str(chosen.source.metadata.get("section_title") or "")
                    for chosen in selected
                ):
                    diversity_bonus += 0.06
                marginal = 0.72 * candidate.relevance - 0.28 * max(
                    self._similarity(candidate.context, chosen.context)
                    for chosen in selected
                ) + diversity_bonus
                if marginal > best_score:
                    best_score = marginal
                    best_candidate = candidate
            if best_candidate is None or best_score < 0.08:
                break
            selected.append(best_candidate)

        groups = self._build_evidence_groups(message, selected, budget=budget)
        if not groups:
            groups = [[0]]

        used_indexes = sorted({index for group in groups for index in group})
        remapped_selected = [selected[index] for index in used_indexes]
        remap = {old_index: new_index for new_index, old_index in enumerate(used_indexes)}
        remapped_groups = [
            [remap[index] for index in group if index in remap]
            for group in groups
        ]
        remapped_groups = [group for group in remapped_groups if group]
        return AnswerPlan(evidences=remapped_selected, evidence_groups=remapped_groups)

    @staticmethod
    def _source_context(source: CopilotChatSourceItem) -> str:
        section = str(source.metadata.get("section_title") or source.label or "").strip()
        synthesis_text = str(
            source.metadata.get("synthesis_text")
            or source.metadata.get("preview_text")
            or ""
        ).strip()
        if not synthesis_text:
            return ""
        normalized = " ".join(synthesis_text.split())
        return f"{section}: {normalized}".strip(": ")

    @staticmethod
    def _relevance_score(
        source: CopilotChatSourceItem,
        context: str,
        query_tokens: set[str],
    ) -> tuple[int, int, float]:
        source_tokens = set(tokenize_query(context))
        overlap = len(query_tokens.intersection(source_tokens))
        title = str(source.metadata.get("section_title") or source.label or "")
        title_tokens = set(tokenize_query(title))
        title_overlap = len(query_tokens.intersection(title_tokens))
        coverage = overlap / max(len(query_tokens), 1)
        score = float(source.score or 0.0)
        return overlap, title_overlap, score + coverage + (title_overlap * 0.18)

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        left_tokens = set(tokenize_query(left))
        right_tokens = set(tokenize_query(right))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens.intersection(right_tokens))
        union = len(left_tokens.union(right_tokens))
        return overlap / max(union, 1)

    def _build_evidence_groups(
        self,
        message: str,
        selected: list[PlannedEvidence],
        *,
        budget: int,
    ) -> list[list[int]]:
        if not selected:
            return []

        clauses = self._query_clauses(message)
        clause_groups: list[list[int]] = []
        used: set[int] = set()

        if len(clauses) > 1:
            for clause in clauses[:budget]:
                group = self._best_group_for_clause(clause, selected, used)
                if group:
                    clause_groups.append(group)
                    used.update(group)

        if not clause_groups:
            clause_groups.append(self._primary_group(selected))
            used.update(clause_groups[0])
        else:
            clause_groups = clause_groups[: max(1, budget)]

        return clause_groups[: max(1, budget)]

    def _best_group_for_clause(
        self,
        clause: str,
        selected: list[PlannedEvidence],
        used: set[int],
    ) -> list[int]:
        clause_tokens = set(tokenize_query(clause))
        if not clause_tokens:
            return []

        clause_scores: list[tuple[float, int, int, int]] = []
        for index, evidence in enumerate(selected):
            token_overlap, title_overlap, relevance = self._relevance_score(evidence.source, evidence.context, clause_tokens)
            if token_overlap <= 0 and title_overlap <= 0:
                continue
            clause_scores.append((relevance, title_overlap, token_overlap, index))
        if not clause_scores:
            return []

        clause_scores.sort(reverse=True)
        primary_index = next((index for _score, _title, _overlap, index in clause_scores if index not in used), clause_scores[0][3])
        group = [primary_index]
        primary = selected[primary_index]
        primary_score = next(score for score, _title, _overlap, index in clause_scores if index == primary_index)

        for score, title_overlap, token_overlap, index in clause_scores:
            if index == primary_index or len(group) >= 2:
                continue
            candidate = selected[index]
            if score < primary_score * 0.72:
                continue
            if self._similarity(primary.context, candidate.context) >= 0.58:
                continue
            if title_overlap <= 0 and token_overlap <= 1:
                continue
            group.append(index)
            break
        return sorted(group)

    def _primary_group(self, selected: list[PlannedEvidence]) -> list[int]:
        group = [0]
        primary = selected[0]
        for index, candidate in enumerate(selected[1:], start=1):
            if len(group) >= 2:
                break
            if candidate.relevance < primary.relevance * 0.93:
                continue
            if candidate.title_overlap <= 0 and candidate.token_overlap <= 1:
                continue
            if self._similarity(primary.context, candidate.context) >= 0.46:
                continue
            group.append(index)
        return sorted(group)

    @staticmethod
    def _query_clauses(message: str) -> list[str]:
        text = str(message or "").strip()
        if not text:
            return []
        normalized = text
        for marker in (" 그리고 ", " 그리고", " 및 ", " 랑 ", "이랑 ", " 하고 ", " 와 ", " 과 ", " also ", " and "):
            normalized = normalized.replace(marker, "|")
        parts = [
            part.strip()
            for part in normalized.replace("?", "|").replace("/", "|").split("|")
            if part.strip()
        ]
        return parts[:3] or [text]

