# RAG Task

## 과제 내용

- User → RAG → LLM 이어지는 UI 및 전체 파이프라인 직접 구현
- (오픈소스 프레임워크 사용 금지, 모든 구성 직접 개발)
- RAG 시스템 구축(Tech Stack Free) - 자료는 자사 교육자료 및 타 팀 제공 정리 문서 자율 수집
- 멀티턴 대화 (minimum 5 turn) 지원
- LLM Model - Qwen/Qwen3.5-9B | Endpoint: http://cllm.cywell.co.kr/v1

## 추가 요구사항

- Streaming 응답 처리 구현 (토큰 단위 or chunk 단위 출력)
- Vector Index 구조 직접 설계 및 구현 (단순 라이브러리 호출 지양)
- 대화 이력 기반 Context 관리 (세션 단위 memory 구조 설계)
- RAG 성능 개선 전략 1가지 이상 적용 (ex: reranking, chunking 전략 등)
- 캐싱 전략 적용 (ex: 동일 질의 응답 캐싱, embedding 캐싱 등)

## 평가 기준(총 100점)

1. RAG 정확성( 30점 )
- 문서 검색 정확도
- 적절한 context 구성 여부
- 단순 LLM 질의 기반 응답이 아닌 RAG 파이프라인을 통한 결과 생성이 필수이며 관련 로직에 대한 상세 질의가 진행될 예정

1. 멀티턴 처리 정확성( 20점 )
- 이전 대화 맥락 반영 여부
- context 유지 및 활용 능력
- 단순 LLM의 히스토리 기능에 의존하지 않고, 대화 이력 관리 및 context 구성 로직을 직접 구현해야 하며 관련 로직에 대한 상세 질의가 진행될 예정

1. 시스템 아키텍처( 20점 )
- 구조 설계의 명확성
- 확장성 및 안정성 고려 여부

1. 코드 품질 및 설명 ( 15점 )
- 코드 가독성 및 구조화 수준
- 주요 로직 설명 능력
- 바이브 코딩을 활용한 구현은 가능하나, **구현 코드 전반에 대한 상세 질의가 진행될 예정이며, 이에 대한 설명이 가능해함**

1. 추가 요구사항 구현 ( 15점 )
- Streaming / 캐싱 / 성능 개선 요소 반영 여부
- 구현 완성도

## 제출 방법

- 개인별로 Git Repository 구성
- 실행 방법 및 아키텍처 설명을 포함한 README 작성
- Repository URL을 본 메일에 전체 회신 → Repo URL은 바로 회신
- 개발 기간 동안 일일 커밋 필수 (진행 이력 확인 예정)

## 과제 정리
- 오픈소스 RAG 프레임워크 없이 직접 파이프라인을 구성
- PDF 중심 문서 저장소를 기반으로 검색 가능한 지식베이스를 생성
- 멀티턴 대화를 지원하되, 단순 LLM history 기능에 의존하지 않고 세션 단위 memory를 직접 설계
- 스트리밍 응답, 캐싱, 검색 성능 개선 요소를 포함한 구조를 고려
- 추후 OCP 환경에서 API 서버, 인덱싱 워커, 저장소 계층을 분리 확장할 수 있도록 모듈 경계를 명확하게 구성

## 목표 아키텍처

흐름 초기 구상

```text
User
  -> Web UI
  -> API Layer
  -> Session Memory / Query Rewrite
  -> Retrieval
      -> Vector Index
      -> Sparse Score
      -> Rerank
  -> Context Builder
  -> LLM
  -> Streaming Response
```

문서 업로드 설계 계획

```text
PDF Upload / Library Ingestion
  -> PDF Text Extraction
  -> OCR Fallback
  -> Text Normalization
  -> Chunking
  -> Embedding
  -> Vector Store / Metadata Store
```

## 주요 설계 내용

### 1. 자료실 중심 구조

- 사용자가 직접 문서를 업로드하거나 자료실 폴더에 PDF를 넣을 수 있는 구조로 설계
- 업로드된 문서는 원본 저장소와 검색용 인덱스를 분리하여 관리할 수 있도록 함
- 자료실에서는 문서 목록, 상태, 메타데이터를 확인할 수 있도록 함

### 2. PDF + OCR 처리

- 첫번째로 PDF의 텍스트 레이어를 추출
- 텍스트가 부족한 경우 OCR fallback 실행
- OCR 결과는 페이지 번호, 원본 파일명등과 같이 메타데이터로 저장하는 방식으로 함
- 자사 문서 대부분이 한국어로 이루어져있기에 한국어 기능 OCR을 우선시함, 대신 영문 혼합 문서도 처리 가능하도록 설계

### 3. Chunking 전략

- 문서를 통째로 넣지 않고 일정 길이로 청킹
- 문맥 간 간격에서 손실을 줄이기 위해 overlap 사용
- 페이지/문서/섹션 단위 메타데이터를 함께 저장해 검색 근거 제시에 활용

대충 청킹 예상:

- chunk size: 500~800 chars
- overlap: 80~150 chars

### 4. Vector Index 구조

- 단순 라이브러리 호출은 지양이기에 애플리케이션 수준에서 벡터 저장 및 검색 로직을 직접 설계해야함
- 초기 구현은 SQLite 또는 파일 기반 저장으로 함
- 저장할 것:
  - 원문 청크
  - 메타데이터
  - 벡터 값
  - 인덱싱 시간

### 5. Retrieval 전략

- dense retrieval만 쓰지 않고 sparse signal도 생각
- hybrid retrieval 또는 간단한 reranking 전략을 넣어 검색 정확도 향상
- retrieval score threshold를 사용해, 문서 근거가 부족할 때는 일반 답변 fallback 기반으로 답변나오게 설계

### 6. 세션 메모리 구조

- 세션별로 최근 대화와 요약 메모리를 분리해서 저장
- 후속 질문이 짧거나 모호한 경우 이전 대화를 기반으로 검색 질의를 재작성해야함
- 목표는 `LLM history 전달`이 아니라 `애플리케이션 레벨 context 구성`

예상 메모리 구성:

- 최근 대화 내역
- 세션 요약
- 사용자 질문 이전 대화 기반 재생성

### 7. Streaming 응답

- 사용자에게 시각적으로 빠른 답변을 위해 스트리밍 응답 사용
- LLM 응답은 chunk 단위나 token 단위
- 캐시 응답도 동일하게 스트리밍 형태로 사용해서 통일성

### 8. 캐싱 전략

- embedding cache: 동일 문서/청크 재계산 방지
- answer cache: 동일 질의 + 동일 retrieval context 조합 재사용
- 캐시 키는 질의, 세션 맥락, retrieval 결과를 반영하는 형태로

## 기술 선택 초안

- Backend: `Python + FastAPI`
- Frontend: `HTML/CSS/Vanilla JS`
- PDF Parsing: `PyMuPDF` or `PDFPlumber`
- OCR: `PaddleOCR` , `Tessaract OCR`, `EasyOCR`중 선택
- Storage: `SQLite`
- LLM Client: `httpx`
- Streaming: `SSE`
