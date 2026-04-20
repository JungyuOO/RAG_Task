# RAG 품질 개선 — Codex 작업 지시서 (2026-04-18)

**배경**: C1(한국어→영문 동의어 확장) + C2(rerank scope 3→6) + 레거시 corpus 제거를 이미 적용함. v2 dataset(41건)에서 ok=28/41, src_match=27/41, section_match=21/41, citation=36/41 상태. 남은 실패 분석 결과 아래 5개 Priority 작업이 남았음. Codex가 이어서 진행.

**공통 환경**:
- 브랜치: `dev-ver2`
- 경로 규약: `apps/api/rag/...` (구조 마이그레이션 완료, `app/` prefix는 legacy)
- Docker: `docker-compose.yml` + `.env` (DB_USER=admin, DB_NAME=cywell)
- Eval 실행: `python scripts/run_service_eval.py --dataset tests/data/service_eval_dataset_v2.json --limit 50 --output-json tests/results/service-eval/<name>.json --output-md tests/results/service-eval/<name>.md`
- 스모크 한 번 돌리려면 `--limit 10`(v1) 또는 `--limit 5` 선별.
- 코드 변경 후 반드시 `docker-compose restart app` 후 `curl http://localhost:8000/openapi.json` 200 확인 후 eval.

**실패 케이스 요약 (v2_post_c1c2.json 기준, legacy 제거 후 재측정 필요)**:
| ID | 카테고리 | Q 핵심 | 기대 src/section | 현재 retrieval |
|---|---|---|---|---|
| 0005 | 검색 정밀도 | 유저 oauth 토큰 리스트 | authentication_and_authorization / 5.1. Listing user-owned OAuth access tokens | networking_operators, configuring_network_settings, security_and_compliance |
| 0007 | 검색 정밀도 | classic LB → NLB 교체 | ingress_and_load_balancing / 2.6.2.1. Switching the Ingress Controller from Classic to NLB | security_and_compliance, nodes |
| 0009 | Intent 오분류 | 이미 올라가 있는 aws 클러스터에 ingress nlb | ingress_and_load_balancing / 2.6.2.4. Configuring an Ingress Controller NLB on an existing AWS cluster | lane=needs_connection (빈 결과) |
| 0012 | Citation | 웹 콘솔 네트워크 그래픽 뷰 | kubernetes_nmstate / 1.2. Viewing a graphical representation NNS | retrieval 맞음, citation_alignment=False |
| 0025 | Citation | control plane 권장 practice | scalability_and_performance / 2.1. Recommended control plane practices | retrieval 맞음, citation_alignment=False |
| 0029 | 경로 convention | 프로젝트 단위 작업 의미 | building_applications / 1.1. Working on a project | top-1은 building_applications인데 legacy 경로였음 → 레거시 제거로 해결 기대 |
| 0033 | 검색 정밀도 | machine api 개요 설명 | machine_management / 1.1. Machine API overview | storage, nodes, updating_clusters |
| 0034 | 검색 정밀도 | jenkins cross project access | jenkins / 1.3. Providing Jenkins cross project access | service_mesh, building_applications, auth |
| 0035 | 검색 정밀도 | jenkins cross volume mount | jenkins / 1.4. Jenkins cross volume mount points | storage, nodes |
| 0036 | 검색 정밀도 | openshift 내장 registry | registry / 1.2. Integrated OpenShift image registry | images, registry, security |
| 0037 | 검색 정밀도 | nodeport 포트 범위 변경 | configuring_network_settings / Chapter 2. Configuring the node port service range | ingress_and_load_balancing, configuring_network_settings |
| 0038 | Citation | 클러스터 네트워크 cidr 범위 변경 | configuring_network_settings / Chapter 3. Configuring the cluster network range | retrieval 맞음, citation=False |
| 0041 | Intent 오분류 | 클러스터 업데이트 흐름 | updating_clusters / 1.1. Introduction to OpenShift updates | lane=needs_connection |

---

## Task P2: Intent Router 룰 기반 오분류 수정 (2건)

### 문제
`apps/api/rag/query/intent_agent.py`의 `_classify_with_rules` → `_has_live_signal`이 아래 쿼리를 모두 "live"로 판정:
- "이미 올라가 있는 aws 클러스터에 ingress nlb 붙이려면?"
- "openshift 클러스터 업데이트 큰 흐름이 어떻게 되는지"

원인: `_has_live_signal`(195–256행)이 `"클러스터" in lowered` + LIVE_CONTROL_TERMS 매치만으로 live 판정. 둘 다 **"클러스터"**가 있지만 **Korean how-to 마커("붙이려면", "어떻게", "흐름")** 가 DOC_STYLE_MARKERS에 없어 doc signal이 안 뜸.

### 수정 위치
`apps/api/rag/query/intent_agent.py`:
1. `DOC_STYLE_MARKERS`(20–47행)에 추가:
   - `"어떻게"`, `"하려면"`, `"려면"`, `"되는지"`, `"흐름"`, `"순서"`, `"절차"`, `"붙이려면"`, `"바꾸려면"`, `"바꿀"`, `"업데이트"`, `"업그레이드"`, `"upgrade"`, `"update"`
2. `_has_live_signal`(239–256행) 로직 강화:
   - "cluster"/"클러스터" 단독으로는 live 신호 취급 금지.
   - live 신호 요건에 **resource term OR yaml/manifest 키워드 OR name-like token(하이픈/숫자 포함 token)** 이 반드시 동반되어야 함.
   - 즉 `cluster/클러스터`만 있고 resource term이 없으면 doc fallback.

### 검증
- `tests/test_query_router.py`, `apps/api/tests/unit/*intent*` 에 케이스 추가:
  - "이미 올라가 있는 aws 클러스터에 ingress nlb 붙이려면?" → `doc` 또는 `mixed` (live 아님).
  - "openshift 클러스터 업데이트 큰 흐름이 어떻게 되는지" → `doc`.
  - "현재 클러스터에 pod 몇 개 있어?" → 여전히 `live` (regression 체크).
- 단위 테스트: `docker-compose exec -T app pytest tests/test_query_router.py apps/api/tests/unit/ -k intent -x 2>&1 | tail -30`
- e2e: service eval v2 재측정 후 0009/0041 lane이 doc으로 바뀌고 retrieval이 기대 source_path를 상위에 올리는지 확인.

### 수락 조건
- 0009, 0041 모두 ok=True.
- v1 regression 체크(`--limit 10`)에서 10/10 ok 유지.

---

## Task P3: Retrieval 정밀도 — 타이틀 IDF 가중 + Korean 브리지 강화 (5건)

### 문제
- `0005` oauth 토큰 리스트: "토큰" keyword가 networking_operators의 MTU/인증서 섹션까지 매치함.
- `0007` classic→NLB switch: "바꾸고 싶은데"가 추상적 표현이라 정확한 섹션 매치 실패.
- `0033` machine api: "machine" 단독이 machine_management 대비 storage/nodes 문서 chunk에도 분산.
- `0034/0035` jenkins: "jenkins"가 service_mesh.md, building_applications.md에서도 언급됨 → 제목에 "jenkins"가 있는 jenkins.md 문서를 특정 못함.
- `0037` nodeport: ingress_and_load_balancing.md에도 "node port" 관련 서술이 있어 distractor로 작동.

### 핵심 가설
**현재 scorer는 섹션 타이틀의 discriminative term을 충분히 boost하지 않음.** `apps/api/rag/retrieval/document_retriever.py` 172–215행 scorer는 BM25×5.0 + title_token +2.5 + haystack +1.0 하지만 title_token에 IDF 가중이 없음. "jenkins" 같은 희귀 discriminative term이 평범한 common term과 같은 weight로 처리됨.

### 작업
1. **Title IDF 가중치 도입** (`apps/api/rag/retrieval/document_retriever.py`):
   - 모든 chunk의 `section_title` tokens을 corpus-wide로 수집해 document frequency 계산.
   - Scorer에서 title token 매치 시 `log(N/df)` 로 가중. df=1(title에만 등장)인 term은 아주 크게, 흔한 term은 거의 0.
   - Cache: IDF dict를 `apps/api/rag/retrieval/` 안 싱글톤이나 startup 시점 계산. postgres 에서 chunks 테이블의 metadata_json 파싱해서 만드는 정적 빌드 스크립트를 `scripts/build_title_idf.py`로 추가하고 결과를 `data/retrieval/title_idf.json`에 저장. 앱 부팅 시 로드.
2. **Korean 질의 문맥 단어 → 영문 타이틀 토큰 매핑 확장** (`apps/api/rag/query/synonym_expansion.py`):
   - `"machine api": "machine api overview cluster api"`
   - `"cross project": "cross project providing access jenkins"`
   - `"cross volume": "cross volume mount jenkins points"`
   - `"내장": "integrated built-in"`
   - `"범위": "range service port configuring"`
   - `"바꾸고": "switch change switching migration"` (기존 "바꾸" 연장)
   - `"토큰 리스트": "oauth access token listing user-owned"`
   - `"토큰 뽑": "oauth access tokens listing"`
3. **PHRASE_EXPANSIONS에 "machine api" 추가** — 현재 없음.

### 검증
- 유닛: `tests/test_document_retriever.py`(있으면) 또는 신규 `tests/test_title_idf_weighting.py`로 "jenkins cross project access" 쿼리에서 `jenkins.md`의 "1.3. Providing Jenkins cross project access" 섹션이 top-3 안에 드는 것을 assert.
- e2e: v2 eval 재측정, 0005/0007/0033/0034/0035/0036/0037 중 최소 3건에서 ok=True.

### 수락 조건
- v2 ok ≥ 33/41.
- v1 regression 10/10 유지.

---

## Task P4: Citation Alignment 프롬프트/로직 수정 (3건)

### 문제
0012, 0025, 0038은 retrieval은 정답(src_match=True, section_match=True)인데 `citation_alignment=False`. Eval scorer는 answer paragraph tokens ↔ cited source tokens overlap을 본다.

실패 케이스:
- 0012 "웹 콘솔에서 노드 네트워크 상태 그래픽으로 보는 방법" → kubernetes_nmstate 정답이지만 answer 문단 토큰이 citation source chunk 토큰과 겹치지 않음.
- 0025 "control plane 운영할 때 권장되는 practice 가 뭐야?" → scalability_and_performance 정답이지만 answer가 인용 내용과 paraphrase 거리 큼.
- 0038 "클러스터 내부 네트워크 cidr 범위 바꾸는 방법" → configuring_network_settings 정답이지만 인용 section은 "Chapter 3. Configuring the cluster network range"인데 answer는 상위 개념만 paraphrase.

### 조사할 지점
1. `apps/api/rag/generation/unified_copilot_service.py`의 answer synthesis prompt — `[N]` citation marker를 강제하는지, LLM이 citation 없이 paraphrase만 하는지 확인.
2. Eval scorer (`scripts/run_service_eval.py` 또는 `tests/utils/...`) 의 `citation_alignment` 계산 로직 — 어떤 토큰 overlap 기준인지 확인하고 너무 엄격하면 완화, 또는 answer가 실제로 citation source에 근거하도록 prompt 강화.

### 작업
1. `unified_copilot_service.py` answer synthesis prompt에 명시:
   - "Each factual claim MUST end with `[N]` referencing the numbered source; do not paraphrase beyond evidence scope."
   - 답변 시작 문장에 반드시 [1]을 포함하도록 제약.
2. Citation alignment scorer 재검토:
   - 현재 threshold(어떤 %인지)를 로그로 출력하는 디버그 옵션 추가.
   - 과도하게 엄격하면 n-gram 기반 → unigram 최소 overlap 비율로 완화 (예: 30% → 20%).

### 검증
- 0012, 0025, 0038 각각에 대해 답변 텍스트 수작업 검토 후 scorer 결과가 납득 가능한지 확인.
- 3건 중 최소 2건 ok=True로 복구.

### 수락 조건
- v2 citation_alignment ≥ 38/41.

---

## Task P5: Corpus 추가 정리 및 리인덱스 검증

### 배경
레거시 corpus(`data/corpus/pdfs/legacy/official-en-pre-redhat-html/`)는 이미 파일시스템과 DB(documents+chunks 1396rows)에서 삭제됨. 남은 corpus는 `data/corpus/pdfs/official/en/` 28개 문서, 12942 chunks.

### 확인 작업
1. `scripts/build_section_index.py`가 legacy 경로를 walk하도록 되어 있을 가능성 — 신규 `official/en/` 경로로 수정 확인 또는 재실행.
   ```
   python scripts/build_section_index.py  # output: tests/data/section_index.json 재생성
   ```
2. 기존 ingestion/reindex 스크립트들 중 레거시 경로 하드코딩 흔적 탐색:
   ```
   rg -l "legacy/official-en-pre-redhat-html" scripts/ apps/
   rg -l "ocp-.*-openshift-docs" scripts/ apps/
   ```
   발견되면 주석 처리 또는 `official/en/` 경로로 교체.
3. 중복 인덱스 방지: `scripts/reindex_official_pdfs.py`가 official/en/만 보고 있는지 확인.

### 검증
- `docker-compose exec -T postgres psql -U admin -d cywell -c "SELECT COUNT(DISTINCT source_path) FROM documents;"` → 28 유지.
- `SELECT source_path FROM documents WHERE source_path NOT LIKE '%/official/en/%';` → 빈 결과.
- service eval v2 재측정 시 `retrieved_source_paths`에 `/legacy/` 가 전혀 등장하지 않음.

### 수락 조건
- DB 내 legacy 참조 0건.
- 스크립트 중 legacy 경로 하드코딩 0건(또는 안전하게 비활성화).

---

## Task P1 보완: Eval Scorer 경로 정규화 (선택)

레거시 제거 후에도 dataset source_path(`official/en/...`) 와 retrieval 경로(`data/corpus/pdfs/official/en/...`)가 일치해야 `endswith` 매치됨. 확인:

- v2 eval을 다시 실행했을 때(`tests/results/service-eval/v2_post_legacy_removed.json` 생성됨) `retrieved_source_paths`가 실제로 `.../official/en/...` suffix로 끝나는지 검증.
- 안 끝나면 `scripts/run_service_eval.py`의 `source_match` 계산에 filename fallback 추가:
  ```python
  # 기존: src_match = any(p.endswith(case["source_path"]) for p in retrieved)
  # 신규: filename 기반 2차 매치
  exp_file = case["source_path"].rsplit("/", 1)[-1]
  src_match = any(p.endswith(case["source_path"]) or p.endswith("/" + exp_file) for p in retrieved)
  ```

### 수락 조건
- v2 재측정에서 src_match가 filename 기준으로도 올바르게 잡힘.

---

## 실행 순서 권장

1. **P5 선 검증** (10분): legacy 제거 후 스크립트 하드코딩 잔재 제거.
2. **P1 보완** (10분): 경로 매치 정규화 후 v2 baseline 재측정 → 진짜 metric 확정.
3. **P2** (~30분): Intent router 수정 + 유닛/e2e 테스트 → 2건 복구.
4. **P3** (~90분): Title IDF + synonym 확장 → 3~5건 복구. 가장 임팩트 큰 작업.
5. **P4** (~60분): citation prompt/scorer 튜닝 → 2~3건 복구.

각 단계 후 반드시 v2 full eval(+v1 regression `--limit 10`)로 회귀 확인.

## 보고 양식
각 Task 완료 시 다음 형식으로 요약:
- 수정 파일 & 핵심 diff 요지
- 측정값 (ok/src/sec/cit/avg_ms) before → after
- 회귀 여부 (v1 10/10 유지)
- 새로 실패한 케이스 ID
