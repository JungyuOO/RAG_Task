"""멀티턴 대화 I/O 기록 테스트.

실제 대화 시나리오를 시뮬레이션하고, 각 턴의 input/output/context를
JSON 파일로 tests/results/ 에 기록한다.

검색·세션 메모리·쿼리 재작성 등 RAG 파이프라인의 실제 동작을 확인하기 위한
통합 테스트이며, LLM 응답만 mock으로 대체한다.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import unittest
from pathlib import Path

from tests.conftest import clean_pg_tables, get_test_dsn, requires_pg

from app.config import Settings
from app.rag.chunking import StructuredMarkdownChunker, TextChunker
from app.rag.index import VectorIndex
from app.rag.memory import SessionStore
from app.rag.retrieval import HybridRetriever
from app.rag.utils import stable_hash, tokenize
from app.services.retrieval_service import RetrievalService


def _make_test_vector(text: str, dim: int = 1024) -> list[float]:
    """Deterministic fake vector for testing purposes."""
    digest = hashlib.md5(text.encode()).digest()
    vals = [((digest[i % 16] ^ (i * 7)) - 128) / 128.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


RESULTS_DIR = Path("tests/results")


def _make_index_item(chunk_id: str, source_path: str, text: str, page: int) -> dict:
    """테스트용 인덱스 아이템을 생성한다."""
    tokens = tokenize(text)
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "doc_id": stable_hash(source_path),
            "source_path": source_path,
            "text": text,
            "tokens": tokens,
            "page_number": page,
            "metadata": {"page_start": page, "page_end": page},
        },
        "vector": _make_test_vector(text),
    }


@requires_pg
class MultiTurnIOTest(unittest.TestCase):
    """멀티턴 대화 시나리오를 시뮬레이션하고 I/O를 JSON으로 기록한다."""

    def setUp(self) -> None:
        clean_pg_tables()
        self.tmp = Path("tests/.tmp/multiturn-io")
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self.dsn = get_test_dsn()
        self.retriever = HybridRetriever(
            top_k=3, candidate_pool_size=6,
            dense_weight=0.45, sparse_weight=0.25, title_weight=0.15,
            bm25_k1=1.2, bm25_b=0.75,
            rerank_base_weight=0.68, rerank_overlap_weight=0.17,
            rerank_title_weight=0.08, rerank_title_bonus_weight=0.07,
            rerank_compact_bonus_weight=0.12, title_match_bonus=0.35,
        )
        self.session_store = SessionStore(
            dsn=self.dsn,
            memory_window_turns=6,
        )
        self.settings = Settings(
            rag_data_dir=self.tmp,
            rag_source_dir=self.tmp / "pdfs",
            rag_index_dir=self.tmp / "index",
            rag_cache_dir=self.tmp / "cache",
            rag_extract_dir=self.tmp / "md",
        )
        self.retrieval_service = RetrievalService(self.settings)

    def tearDown(self) -> None:
        clean_pg_tables()
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _expand_query(self, query: str, topic_state: dict) -> str:
        """파이프라인의 _expand_query_with_context와 동일한 로직."""
        if not topic_state:
            return query
        entities = topic_state.get("active_entities", [])
        sources = topic_state.get("selected_sources", [])
        if not entities and not sources:
            return query

        import re as _re
        has_explicit_keyword = bool(_re.compile(r"[A-Z]{2,}").search(query))

        query_lower = query.lower()
        expansion_tokens: list[str] = []

        if not has_explicit_keyword:
            for source in sources[:2]:
                stem = Path(source).stem
                if stem.lower() not in query_lower:
                    expansion_tokens.append(stem)
            added = 0
            for entity in entities:
                if added >= 3:
                    break
                if entity.lower() not in query_lower and len(entity) >= 2:
                    expansion_tokens.append(entity)
                    added += 1

        if not expansion_tokens:
            return query
        return " ".join(expansion_tokens) + " " + query

    def _build_corpus(self) -> list[dict]:
        """SCC, RBAC, 스토리지 관련 테스트 코퍼스를 생성한다."""
        corpus_data = [
            # SCC 문서
            ("SCC.pdf", 1, "SCC(Security Context Constraints)는 OpenShift에서 Pod의 보안 컨텍스트를 제어하는 정책이다. "
             "컨테이너가 호스트 OS의 커널을 공유하므로, 루트 권한 남용이나 호스트 파티션 접근을 방지하여 클러스터를 보호한다."),
            ("SCC.pdf", 2, "SCC의 주요 제어 항목: UID/GID 범위 제한, 호스트 네트워크 사용 여부, "
             "특권 컨테이너 실행 여부, SELinux 컨텍스트, 볼륨 유형 제한. "
             "기본 SCC 유형으로는 restricted, anyuid, privileged 등이 있다."),
            ("SCC.pdf", 3, "SCC 적용 방법: oc adm policy add-scc-to-user 명령으로 서비스 계정에 SCC를 부여한다. "
             "예시: oc adm policy add-scc-to-user anyuid -z my-service-account -n my-namespace"),
            # RBAC 문서
            ("RBAC.pdf", 1, "RBAC(Role-Based Access Control)은 역할 기반 접근 제어로, "
             "사용자나 서비스 계정이 클러스터 리소스에 대해 어떤 작업을 할 수 있는지 정의한다."),
            ("RBAC.pdf", 2, "RBAC 구성 요소: Role(네임스페이스 범위), ClusterRole(클러스터 범위), "
             "RoleBinding, ClusterRoleBinding. 각 Role은 리소스(Pod, Service 등)와 동사(get, create, delete)를 정의한다."),
            ("RBAC.pdf", 3, "RBAC과 SCC의 차이: RBAC은 '누가 무엇을 할 수 있는가'(API 접근 제어)를 다루고, "
             "SCC는 'Pod가 어떤 보안 환경에서 실행되는가'(런타임 보안)를 다룬다. 두 개념은 보완 관계이다."),
            # 스토리지 문서
            ("스토리지.pdf", 1, "Persistent Volume(PV)은 클러스터 관리자가 프로비저닝한 스토리지 리소스이다. "
             "PV는 NFS, iSCSI, 클라우드 스토리지 등 다양한 백엔드를 지원한다."),
            ("스토리지.pdf", 2, "PVC(Persistent Volume Claim)는 사용자가 스토리지를 요청하는 방법이다. "
             "PVC를 생성하면 적절한 PV에 자동 바인딩되며, Pod에서 volumeMounts로 마운트하여 사용한다."),
        ]

        items = []
        for source, page, text in corpus_data:
            chunk_id = stable_hash(f"{source}:{page}")
            items.append(_make_index_item(chunk_id, source, text, page))
        return items

    def _simulate_turn(
        self,
        session_id: str,
        user_message: str,
        index_items: list[dict],
        mock_llm_response: str,
    ) -> dict:
        """한 턴의 대화를 시뮬레이션하고 I/O를 dict로 반환한다."""

        # 1) 쿼리 재작성 컨텍스트 확인
        rewrite_context = self.session_store.build_rewrite_context(session_id, user_message)

        # 2) 메모리 스냅샷
        memory_snapshot = self.session_store.memory_snapshot(session_id)

        # 3) 쿼리 확장 (세션 토픽 상태 기반)
        topic_state = self.session_store.topic_state(session_id)
        expanded_query = self._expand_query(user_message, topic_state)

        # 4) 검색 수행
        query_vector = _make_test_vector(expanded_query)
        retrieved = self.retriever.search(expanded_query, query_vector, index_items)

        # 5) 검색 메트릭
        metrics = self.retriever.compute_retrieval_metrics(retrieved, min_score=0.12)

        # 7) 페이지 그라운딩
        grounded_pages = self.retrieval_service.aggregate_page_grounding(retrieved)
        source_grounding = self.retrieval_service.aggregate_source_grounding(grounded_pages)

        # 8) 사용자 턴 저장
        self.session_store.add_turn(session_id, "user", user_message)

        # 9) 어시스턴트 턴 저장 (LLM 응답 mock)
        turn_metadata = {
            "mode": "rag" if metrics["top_score"] >= 0.12 else "general",
            "source_grounding": source_grounding,
            "preview_pages": [{"page_number": p["page_number"], "score": round(p["score"], 4)} for p in grounded_pages[:3]],
            "answer_citations": [
                {"file_name": Path(p["source_path"]).name, "page_number": p["page_number"], "score": round(p["score"], 4)}
                for p in grounded_pages[:3]
            ],
        }
        self.session_store.add_turn(session_id, "assistant", mock_llm_response, metadata=turn_metadata)

        # 10) 턴 후 토픽 상태
        topic_state = self.session_store.topic_state(session_id)

        # I/O 기록 구성
        return {
            "input": {
                "user_message": user_message,
                "expanded_query": expanded_query if expanded_query != user_message else None,
                "rewrite_context": rewrite_context,
            },
            "retrieval": {
                "query_used": expanded_query,
                "metrics": {
                    "top_score": round(metrics["top_score"], 4),
                    "hit_count": metrics["hit_count"],
                    "mean_score": round(metrics["mean_score"], 4),
                },
                "top_results": [
                    {
                        "source": r["chunk"]["source_path"],
                        "page": r["chunk"]["page_number"],
                        "score": round(r.get("rerank_score", r.get("score", 0)), 4),
                        "text_preview": r["chunk"]["text"][:120] + "...",
                    }
                    for r in retrieved[:3]
                ],
                "grounded_pages": [
                    {"file": Path(p["source_path"]).name, "page": p["page_number"], "score": round(p["score"], 4)}
                    for p in grounded_pages[:5]
                ],
            },
            "output": {
                "assistant_response": mock_llm_response,
                "mode": turn_metadata["mode"],
                "citations": turn_metadata["answer_citations"],
            },
            "session_state": {
                "active_topic": topic_state.get("active_topic", ""),
                "active_entities": topic_state.get("active_entities", []),
                "selected_sources": topic_state.get("selected_sources", []),
                "memory_snapshot_summary": {
                    "topic": memory_snapshot.get("session_summary", {}).get("topic", ""),
                    "recent_documents": memory_snapshot.get("session_summary", {}).get("recent_documents", []),
                },
            },
        }

    def test_scc_multiturn_conversation(self) -> None:
        """SCC 관련 5턴 멀티턴 대화 시나리오를 시뮬레이션하고 I/O를 JSON으로 기록한다."""
        index_items = self._build_corpus()
        session_id = "test-scc-multiturn"
        turns = []

        # --- Turn 1: SCC 기본 질문 ---
        turn1 = self._simulate_turn(
            session_id, "SCC가 뭐야?", index_items,
            "SCC(Security Context Constraints)는 OpenShift에서 Pod의 보안 컨텍스트를 제어하는 정책입니다. "
            "컨테이너가 호스트 OS의 커널을 공유하기 때문에, 루트 권한 남용이나 호스트 파티션 접근을 방지하여 "
            "클러스터 전체를 보호하는 역할을 합니다. [SCC.pdf] p.1",
        )
        turns.append({"turn": 1, "description": "SCC 기본 개념 질문", **turn1})

        # --- Turn 2: 후속 질문 (대명사 사용) ---
        turn2 = self._simulate_turn(
            session_id, "그거 더 자세하게 어떤 항목들을 제어해?", index_items,
            "SCC의 주요 제어 항목은 다음과 같습니다: UID/GID 범위 제한, 호스트 네트워크 사용 여부, "
            "특권 컨테이너 실행 여부, SELinux 컨텍스트, 볼륨 유형 제한 등이 있습니다. "
            "기본 SCC 유형으로는 restricted, anyuid, privileged 등이 있습니다. [SCC.pdf] p.2",
        )
        turns.append({"turn": 2, "description": "SCC 제어 항목 후속 질문 (대명사 '그거')", **turn2})

        # --- Turn 3: 구현 방법 질문 ---
        turn3 = self._simulate_turn(
            session_id, "그럼 실제로 어떻게 적용해?", index_items,
            "SCC는 oc adm policy add-scc-to-user 명령으로 서비스 계정에 부여합니다. "
            "예시: oc adm policy add-scc-to-user anyuid -z my-service-account -n my-namespace "
            "[SCC.pdf] p.3",
        )
        turns.append({"turn": 3, "description": "SCC 적용 방법 질문 (대명사 '어떻게')", **turn3})

        # --- Turn 4: 토픽 전환 (RBAC과 비교) ---
        turn4 = self._simulate_turn(
            session_id, "RBAC이랑은 뭐가 달라?", index_items,
            "RBAC은 '누가 무엇을 할 수 있는가'(API 접근 제어)를 다루고, "
            "SCC는 'Pod가 어떤 보안 환경에서 실행되는가'(런타임 보안)를 다룹니다. "
            "두 개념은 보완 관계입니다. [RBAC.pdf] p.3",
        )
        turns.append({"turn": 4, "description": "RBAC과 SCC 비교 (토픽 확장)", **turn4})

        # --- Turn 5: 완전히 다른 토픽 ---
        turn5 = self._simulate_turn(
            session_id, "PV랑 PVC는 뭐야?", index_items,
            "PV(Persistent Volume)는 클러스터 관리자가 프로비저닝한 스토리지 리소스이고, "
            "PVC(Persistent Volume Claim)는 사용자가 스토리지를 요청하는 방법입니다. "
            "PVC를 생성하면 적절한 PV에 자동 바인딩됩니다. [스토리지.pdf] p.1-2",
        )
        turns.append({"turn": 5, "description": "토픽 전환 (스토리지)", **turn5})

        # --- Turn 6: 이전 토픽 복귀 ---
        turn6 = self._simulate_turn(
            session_id, "아까 SCC에서 restricted랑 anyuid 차이가 뭐였지?", index_items,
            "restricted SCC는 가장 제한적인 기본 정책으로 루트 실행을 금지하고, "
            "anyuid SCC는 임의의 UID로 컨테이너를 실행할 수 있게 허용합니다. [SCC.pdf] p.2",
        )
        turns.append({"turn": 6, "description": "이전 토픽(SCC) 복귀", **turn6})

        # JSON 저장
        output = {
            "test_name": "SCC 멀티턴 대화 시나리오",
            "session_id": session_id,
            "total_turns": len(turns),
            "corpus_summary": {
                "documents": ["SCC.pdf (3 pages)", "RBAC.pdf (3 pages)", "스토리지.pdf (2 pages)"],
                "total_chunks": len(index_items),
            },
            "turns": turns,
        }

        output_path = RESULTS_DIR / "test_multiturn_io_scc.json"
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        # 기본 검증
        self.assertEqual(len(turns), 6)
        # Turn 1: SCC 문서가 검색되어야 함
        self.assertTrue(any("SCC" in r["source"] for r in turns[0]["retrieval"]["top_results"]))
        # Turn 4: RBAC 문서가 검색되어야 함
        self.assertTrue(any("RBAC" in r["source"] for r in turns[3]["retrieval"]["top_results"]))
        # Turn 5: 스토리지 문서가 검색되어야 함
        self.assertTrue(any("스토리지" in r["source"] for r in turns[4]["retrieval"]["top_results"]))
        # Turn 6: SCC 토픽으로 복귀 시 SCC 문서 검색
        self.assertTrue(any("SCC" in r["source"] for r in turns[5]["retrieval"]["top_results"]))

    def test_rbac_deep_dive_conversation(self) -> None:
        """RBAC 심층 탐구 3턴 대화를 시뮬레이션하고 I/O를 JSON으로 기록한다."""
        index_items = self._build_corpus()
        session_id = "test-rbac-deep"
        turns = []

        turn1 = self._simulate_turn(
            session_id, "RBAC이 뭔지 설명해줘", index_items,
            "RBAC(Role-Based Access Control)은 역할 기반 접근 제어로, "
            "사용자나 서비스 계정이 클러스터 리소스에 대해 어떤 작업을 할 수 있는지 정의합니다. [RBAC.pdf] p.1",
        )
        turns.append({"turn": 1, "description": "RBAC 기본 질문", **turn1})

        turn2 = self._simulate_turn(
            session_id, "구성 요소가 뭐가 있어?", index_items,
            "RBAC의 구성 요소는 Role, ClusterRole, RoleBinding, ClusterRoleBinding입니다. "
            "각 Role은 리소스(Pod, Service 등)와 동사(get, create, delete)를 정의합니다. [RBAC.pdf] p.2",
        )
        turns.append({"turn": 2, "description": "RBAC 구성 요소 후속 질문", **turn2})

        turn3 = self._simulate_turn(
            session_id, "그거랑 SCC 같이 쓰는 경우는?", index_items,
            "RBAC과 SCC는 보완 관계입니다. RBAC으로 사용자에게 SCC 사용 권한을 부여하고, "
            "SCC로 Pod의 런타임 보안을 제어합니다. [RBAC.pdf] p.3",
        )
        turns.append({"turn": 3, "description": "RBAC+SCC 연계 질문", **turn3})

        output = {
            "test_name": "RBAC 심층 탐구 시나리오",
            "session_id": session_id,
            "total_turns": len(turns),
            "turns": turns,
        }

        output_path = RESULTS_DIR / "test_multiturn_io_rbac.json"
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        self.assertEqual(len(turns), 3)
        # Turn 2: 대명사("구성 요소") 질문에서도 RBAC 관련 검색
        self.assertTrue(any("RBAC" in r["source"] for r in turns[1]["retrieval"]["top_results"]))


if __name__ == "__main__":
    unittest.main()
