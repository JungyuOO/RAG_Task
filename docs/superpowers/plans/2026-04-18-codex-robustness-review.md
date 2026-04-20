# Codex Robustness Review

- Date: 2026-04-18
- Scope: retrieval hint generalization and paraphrase robustness review

## What Was Audited

The current RAG path was checked for over-specific keyword dependence in these areas:

- `apps/api/rag/query/synonym_expansion.py`
- `apps/api/rag/query/query_features.py`
- `apps/api/rag/query/query_router.py`

The main risk was not answer hardcoding. The main risk was retrieval bias caused by narrow keyword triggers.

## Risks Found

### Source hint dependence

`preferred_source_paths_for_query()` was strongly dependent on exact surface forms such as:

- `oauth`
- `route`
- `gitops`

This means semantically similar variants such as:

- `인증`
- `인가`
- `권한`
- `라우트`
- `git ops`
- `깃옵스`

could miss the intended document prior.

### Query expansion dependence

`expand_acronyms()` previously favored exact acronym tokens only. That was too narrow for:

- Korean paraphrases
- transliterated forms
- spaced variants such as `git ops`

### Router topic-anchor dependence

`QueryRouter` topic anchoring favored a narrow set of English anchors. That could make generic follow-up handling weaker when the earlier user turn used Korean or transliterated domain terms.

## Changes Made

### Query expansion

Added phrase-level expansion coverage for:

- `인증`
- `인가`
- `권한`
- `라우트`
- `깃옵스`
- `git ops`

These expansions are retrieval hints only. They do not inject answer text.

### Source hint generalization

Broadened source-hint mapping for:

- auth domain:
  - `oauth`
  - `rbac`
  - `authentication`
  - `authorization`
  - `인증`
  - `인가`
  - `권한`
- route domain:
  - `route`
  - `ingress`
  - `라우트`
- GitOps domain:
  - `gitops`
  - `git ops`
  - `argocd`
  - `깃옵스`

### Router topic anchors

Broadened topic-anchor detection to include the same validated variant set so follow-up handling is less brittle across paraphrases.

## Tests Added

- `tests/test_query_hint_generalization.py`

Coverage added for:

- `oauth` style questions phrased as `인증`
- `route` phrased as `라우트`
- `gitops` phrased as `git ops`
- authorization questions phrased as `권한`

## Remaining Risks

The current system is more robust than before, but it is still heuristic-driven in these ways:

- It still relies on domain anchors appearing somewhere in the query.
- It does not yet normalize all Korean transliterations or slang variants.
- It does not learn semantic equivalence from examples; it uses fixed retrieval-side hints.
- New domains will still need new hint coverage if the document corpus is highly specialized.

## Bottom Line

The current implementation is not answer-hardcoded.

It is heuristic-tuned at the retrieval layer.

After this pass, the main remaining risk is not exact keyword brittleness for the audited domains, but coverage gaps for unseen paraphrase families outside the currently validated set.
