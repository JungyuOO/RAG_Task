<!--
  README_PPT.md — 발표용 슬라이드
  ────────────────────────────────────────────────
  · GitHub 에서 열면 `---` 가로줄로 슬라이드가 자연스럽게 구분됩니다.
  · Marp / Marp for VS Code 로 열면 실제 PPT 슬라이드로 변환됩니다.
    (VS Code 에서 `Marp: Export slide deck` 선택)
-->

---
marp: true
theme: default
paginate: true
size: 16:9
header: "RAG Task — OCP 문서 기반 질의응답 시스템"
footer: "2026-04-13 · 정유"
---

<!-- _class: lead -->

# RAG Task
## OCP 문서 기반 멀티턴 질의응답 시스템

프레임워크 없이 직접 구현한 Retrieval-Augmented Generation

`2026-04-13` · 발표자: **정유**

---

## 1. 배경 — 우리가 풀고 싶은 문제

- 고객사는 **Openshift Container Platform(OCP)** 를 운영하지만, 공식 문서와 실제 운영 매뉴얼은 매번 따로 찾아봐야 함.
- "명령어 뭐였지?", "yaml 어떻게 생겼지?" 같은 질문에 답하려면 **여러 PDF 를 뒤적여야 하는 상황**.
- 실제 운영 상태(`oc get pod`, 이벤트, YAML)는 **"지금 이 순간"** 의 정보가 필요 — 문서만으로는 부족.
- 기존 RAG 프레임워크(LangChain/LlamaIndex) 는 내부 동작이 블랙박스 → **현업 요구사항 맞춤 제어가 어려움**.

---

## 2. 목표

1. **문서 기반**으로 질문에 답하되, 근거 페이지·인용을 항상 노출.
2. **멀티턴**으로 "그거 yaml 로 보여줘" 같은 이어지는 질문도 이해.
3. **Live OCP API** 와 결합해서 문서 예시 + 실제 cluster 결과를 **한 답변에 결합**.
4. RAG 프레임워크 없이 **색인·검색·임베딩·세션을 직접 구현** — 모든 로직이 코드로 드러남.

---

## 3. 아키텍처 한눈에 보기

<!-- 아래 mermaid 를 이미지로 export 후 경로 교체 -->
<p align="center">
  <img src="docs/images/pipeline.png" alt="RAG Pipeline" width="820"/>
</p>

- **Backend**: FastAPI · PostgreSQL · BGE-M3
- **LLM**: 사내 vLLM (Qwen 계열)
- **실행**: `docker compose up --build -d` 한 줄

---

## 4. 파이프라인 — Mermaid 소스

```mermaid
flowchart TB
    subgraph INDEX["① 문서 준비 (미리 해두는 작업)"]
        direction TB
        D1["PDF 문서 업로드"] --> D2["문서에서 텍스트 꺼내기"]
        D2 --> D3["목차·헤딩 기준 의미 단위 쪼개기"]
        D3 --> D4["검색용 숫자로 변환"]
        D4 --> D5[("문서 저장소에 보관")]
    end

    subgraph QUERY["② 질문 응답 (사용자 질문할 때)"]
        direction TB
        Q1["사용자 질문 입력"] --> Q2{"질문 의도 확인"}
        Q2 -->|문서 질문| Q3["이전 대화 문맥 복원"]
        Q2 -->|인사·일반| Q7
        Q3 --> Q4["관련 내용 찾기<br/>(의미 + 키워드 동시 검색)"]
        Q4 --> Q5["관련 내용 순위화"]
        Q5 --> Q6["답변에 쓸 근거 선택"]
        Q6 --> Q7["LLM 답변 작성"]
        Q7 --> Q8["대화 기억 갱신"]
        Q8 --> Q9["근거와 함께 답변 전달"]
    end

    D5 -. "검색 대상으로 사용" .-> Q4
```

---

## 5. 핵심 기술 스택

| 영역 | 기술 |
| --- | --- |
| **색인** | PyMuPDF · 헤딩 기반 청킹 · BGE-M3 임베딩 |
| **검색** | BGE-M3 Dense + BM25 Sparse + **RRF 융합** + BGE Cross-Encoder Reranker |
| **의도** | LLM Agent 3 종 (`IntentAgent` · `RetrievalAgent` · `AnswerRewriteAgent`) |
| **세션** | PostgreSQL 기반 topic state · step cursor · example anchor |
| **실시간** | OCP REST API (`/api/v1/namespaces/...`) 직접 호출 |
| **Stream** | FastAPI SSE + `status` 이벤트 (검색 중 / OCP 조회 중) |

---

## 6. 시연 ① — 문서 + Follow-up

### 🎯 상황
기술 엔지니어가 OCP 매뉴얼을 처음 보며 명령어를 순차적으로 확인하는 흐름.

### 🎯 목표
- 대명사 · 이어지는 질문을 이전 turn 의 주제 상태로 **자동 복원**.
- 모든 답변에 **source 태그 + preview 페이지** 가 인라인으로 부착되는지 확인.

### 💬 질문 흐름 (5 turn)
1. `namespace 확인 명령어 뭐야?`
2. `pod 확인하는 명령어 뭐야?`
3. `yaml 보려면 무슨 명령어 써?`
4. `pod 확인하는 명령어 뭐야?`
5. `그거 yaml로 보려면?`

---

## 6-a. 시연 ① — 확인 포인트

- ✅ `oc project`, `oc projects`
- ✅ `oc get pods`, `oc get pods -o wide`
- ✅ `oc get pod <pod_name> -o yaml`, `oc describe pod <pod_name>`
- ✅ **인라인 source 태그** + **preview 페이지** 연결
- ✅ 5 턴에서 대명사 `"그거"` 가 직전 resource (pod) 로 자동 resolve
- ✅ 각 답변의 근거 chunk 가 preview 우측 패널에서 하이라이트

---

## 7. 시연 ② — OCP API 기반 5턴

### 🎯 상황
운영 담당자가 실제 cluster 상태를 **live 로 조회**.

### 🎯 목표
- 문서 RAG 대신 **OCP REST API 라우팅**이 트리거되는지 확인.
- ambiguous 질문은 clarification 요구, explicit pod name 은 실제 YAML 반환.

### 💬 질문 흐름 (5 turn)
1. `지금 pandas 관련 pod 보여줘`
2. `그쪽 yaml 파일 알려줘`
3. `build-and-push-crxvmo-build-image-pod 이거 yaml 알려줘`
4. `warning 이벤트 보여줘`
5. `현재 demo namespace pod 몇개야?`

---

## 7-a. 시연 ② — 확인 포인트

- ✅ `pandas` pod live 조회 성공 (실제 pod 목록 반환)
- ✅ `그쪽 yaml` → ambiguous → **clarification 요구**
- ✅ explicit pod name 지정 시 **실제 YAML 전체** 반환
- ✅ `warning 이벤트` → `type=Warning` 필터링된 이벤트만 표시
- ✅ `몇개야?` → **pod count 집계** (숫자로 응답)
- ✅ 각 답변 route 가 `ocp_live` / `ocp_yaml` / `ocp_event` 로 분류

---

## 8. 시연 ③ — Mixed / Compare 5턴

### 🎯 상황
문서 기반 설명과 실제 cluster 결과를 **한 답변에 동시에** 요구하는 고난도 시나리오.

### 🎯 목표
- 단일 질문에서 **문서 RAG + live OCP API** 를 자동으로 결합하는 `mixed` 라우팅 검증.
- 비교 요청(`뭐가 달라?`)일 때 `공식 문서 기준` / `현재 OCP 기준` 두 섹션으로 분리.

### 💬 질문 흐름 (5 turn)
1. `pod 확인하는 명령어 뭐야?`
2. `지금 pandas 관련 pod 보여줘`
3. `그 yaml 보여줘`
4. `현재 상태 확인 명령어랑 실제 결과 같이 알려줘`
5. `그 pod yaml이랑 공식 문서의 pod yaml은 뭐가 달라?`

---

## 8-a. 시연 ③ — 확인 포인트

- ✅ 1턴: 순수 문서 RAG → `oc get pods` 답변
- ✅ 2턴: live OCP → pandas pod 실제 목록
- ✅ 3턴: follow-up resource 추적 (`pandas pod` → `그 yaml`)
- ✅ 4턴: `mixed` 라우팅 → `문서 기준 명령어` + `현재 OCP 결과` 결합
- ✅ 5턴: **compare 모드** → `공식 문서 기준` / `현재 OCP 기준` / `비교 가이드` 3 섹션으로 구조화
- ✅ `answer_route = "mixed_doc_ocp"` 플래그가 payload 에 부착

---

## 9. 로컬 설치

```bash
# 1) 환경 변수 준비
cp .env.example .env
# → CLLM_* / DB_* / OCP_API_* 값 채우기

# 2) 실행
docker compose up --build -d
```

| URL | 설명 |
| --- | --- |
| http://localhost:8000 | 채팅 UI |
| http://localhost:8000/api/library | 색인 상태 |
| http://localhost:8000/docs | FastAPI Swagger |

- PDF 는 `data/corpus/pdfs/` 에 넣거나 UI 에서 업로드 시 자동 색인.

---

## 10. 기술적 하이라이트

- **프레임워크 zero** — 색인 · 검색 · 리랭킹 · 세션 · 멀티턴 로직 전부 직접 구현.
- **3-signal hybrid retrieval** — dense cosine + BM25 + RRF 융합 후 Cross-Encoder 재정렬.
- **LLM Agent 기반 의도 분류** — 하드코딩 키워드 매칭 최소화, 사내 LLM 만으로 동작.
- **주제 상태 기반 멀티턴** — 단순 history 전달이 아닌 `topic state` 를 매 턴 갱신.
- **Mixed 모드** — 문서 RAG 와 live OCP API 를 한 응답에 자동 결합.
- **버전 태그** — OCP 버전별 색인 분리 · 질문 내 버전 언급 자동 감지.

---

## 11. 로드맵

- 테스트 데이터 기반 **검색 가중치 자동 튜닝**
- **Tokenizer-aware chunk sizing** (모델 입력 길이 직접 반영)
- **버전 간 차이 비교 UI**
- 페이지 경계(표 · 코드 · YAML) **복원 품질 강화**
- **Selection policy 분리** (도메인 서비스화)
- retrieval acceptance · chunking 전략 · follow-up 전후 **실험 로그 정리**

---

<!-- _class: lead -->

# 감사합니다

**질문 환영합니다.**

- 📖 상세 문서: [`README.md`](./README.md)
- 🏗️ 아키텍처: [`DESIGN.md`](./DESIGN.md)
- 💻 Repo: `RAG_Task`
