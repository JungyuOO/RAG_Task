# Docker 컨테이너화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 멀티 스테이지 Dockerfile + docker-compose.yml로 RAG 앱을 컨테이너로 실행 가능하게 만든다.

**Architecture:** Python 3.13-slim 멀티 스테이지 빌드(builder → runtime)로 이미지를 경량화하고, 역할별 named volume 4개(pdf/index/cache/extract)로 OCP PVC 마이그레이션 경로를 확보한다. 경로 설정은 전부 `.env`에서 주입한다.

**Tech Stack:** Docker, docker-compose v2, Python 3.13-slim, uvicorn, FastAPI

**Spec:** `docs/superpowers/specs/2026-03-23-docker-containerization-design.md`

---

## 파일 목록

| 동작 | 파일 | 역할 |
|---|---|---|
| 수정 | `.env.example` | 경로 기본값 정정 및 Docker 경로 주석 추가 (**기존 버그 수정 — 최우선**) |
| 생성 | `.dockerignore` | 빌드 컨텍스트 제외 목록 |
| 생성 | `Dockerfile` | 멀티 스테이지 빌드 정의 |
| 생성 | `docker-compose.yml` | 서비스·볼륨·healthcheck 정의 |

---

## Task 1: .env.example 경로 정정 (**기존 버그 수정 — 최우선 실행**)

**Files:**
- Modify: `.env.example`

> 현재 `RAG_EXTRACT_DIR=/app/data/extracted_markdown`(Docker 절대 경로)가 로컬 개발 기본값으로 잘못 설정되어 있어 로컬 실행이 깨진 상태다. 다른 태스크보다 먼저 실행한다.

- [ ] **Step 1: .env.example 수정**

```
CLLM_BASE_URL=
CLLM_MODEL=

# 로컬 개발 기본 경로
RAG_DATA_DIR=./data
RAG_SOURCE_DIR=./data/corpus/pdfs
RAG_INDEX_DIR=./data/index
RAG_CACHE_DIR=./data/cache
RAG_EXTRACT_DIR=./data/extracted_markdown

# Docker 컨테이너 실행 시 아래 경로로 변경:
# RAG_DATA_DIR=/app/data
# RAG_SOURCE_DIR=/app/data/corpus/pdfs
# RAG_INDEX_DIR=/app/data/index
# RAG_CACHE_DIR=/app/data/cache
# RAG_EXTRACT_DIR=/app/data/extracted_markdown
```

- [ ] **Step 2: 커밋**

```bash
git add .env.example
git commit -m "fix: correct RAG_EXTRACT_DIR default path and add Docker path comments"
```

---

## Task 2: .dockerignore 작성

**Files:**
- Create: `.dockerignore`

- [ ] **Step 1: .dockerignore 파일 생성**

```
.git
.venv
__pycache__
**/__pycache__
*.pyc
*.pyo
*.egg-info
.pytest_cache
data/
.env
tests/
docs/
*.md
```

- [ ] **Step 2: 커밋**

```bash
git add .dockerignore
git commit -m "chore: add .dockerignore for Docker build context"
```

---

## Task 3: Dockerfile 멀티 스테이지 빌드 작성

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Dockerfile 생성**

```dockerfile
# ---- builder: 의존성 설치 전용 스테이지 ----
# TODO: 빌드 안정화 시 패치 버전으로 고정 권장 (예: python:3.13.3-slim)
FROM python:3.13-slim AS builder

WORKDIR /build

# pymupdf 등 네이티브 wheel 빌드가 필요할 경우를 대비한 빌드 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --prefix=/install 로 격리된 경로에 설치 → runtime 스테이지로 복사
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- runtime: 실행 전용 스테이지 ----
# TODO: 빌드 안정화 시 패치 버전으로 고정 권장 (예: python:3.13.3-slim)
FROM python:3.13-slim AS runtime

# builder에서 설치한 패키지만 복사 (빌드 도구 미포함)
COPY --from=builder /install /usr/local

# 비루트 유저 생성 (보안)
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# 프로젝트 루트 전체 복사
# - app/web, app/resources 정적 파일 포함 (StaticFiles 상대 경로 정상 동작)
# - scripts/build_index.py 포함 (컨테이너 내 인덱스 수동 빌드용)
COPY . /app/

# 볼륨 마운트 경로 사전 생성 및 소유권 설정
# appuser 권한으로 SQLite 파일 생성·쓰기 가능하도록 함
RUN mkdir -p \
    /app/data/corpus/pdfs \
    /app/data/index \
    /app/data/cache \
    /app/data/extracted_markdown \
    && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: 이미지 빌드 확인**

```bash
docker build -t rag-task:local .
```

Expected: `Successfully built ...` 또는 `=> => naming to docker.io/library/rag-task:local` 출력

- [ ] **Step 3: 런타임 패키지 임포트 확인**

`--prefix=/install` 후 `COPY /install → /usr/local` 패턴에서 C 확장(pymupdf)이 정상 로드되는지 확인한다.

```bash
docker run --rm rag-task:local python -c "import fitz; import fastapi; import uvicorn; print('OK')"
```

Expected: `OK` 출력. 실패 시(`ImportError`, `ModuleNotFoundError`) Dockerfile의 COPY 경로를 점검한다.

- [ ] **Step 4: 이미지 크기 확인**

```bash
docker images rag-task:local
```

Expected: SIZE가 300~600MB 수준 (단일 스테이지 대비 빌드 도구 제외로 경량화)

- [ ] **Step 5: 커밋**

```bash
git add Dockerfile
git commit -m "feat: add multi-stage Dockerfile (Python 3.13-slim)"
```

---

## Task 4: docker-compose.yml 작성

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: docker-compose.yml 생성**

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    # .env 파일에서 환경 변수 주입 (CLLM_BASE_URL, RAG_SOURCE_DIR 등)
    # Docker 실행 시 RAG_*_DIR 값을 /app/data/... 경로로 변경할 것 (.env.example 주석 참고)
    env_file:
      - .env
    volumes:
      - pdf_data:/app/data/corpus/pdfs          # 원본 PDF — 영속 필수
      - index_data:/app/data/index               # SQLite (벡터·세션) — 영속 필수
      - cache_data:/app/data/cache               # JSON 캐시 — 재생성 가능
      - extract_data:/app/data/extracted_markdown # PDF 추출 마크다운 — 재생성 가능
    healthcheck:
      # python:3.13-slim에는 curl 미포함 → Python 내장 urllib 사용
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/library')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s  # 첫 배포 후 실측하여 조정 권장 (Task 5 Step 5 참고)

volumes:
  pdf_data:
  index_data:
  cache_data:
  extract_data:
```

- [ ] **Step 2: compose 구문 유효성 확인**

```bash
docker-compose config
```

Expected: 에러 없이 파싱된 YAML 출력

- [ ] **Step 3: 커밋**

```bash
git add docker-compose.yml
git commit -m "feat: add docker-compose.yml with role-separated volumes"
```

---

## Task 5: 엔드투엔드 실행 검증

- [ ] **Step 1: .env Docker 경로로 설정**

`.env` 파일에서 RAG 경로를 컨테이너 내부 경로로 변경한다 (`.env.example` 주석 참고):

```
RAG_DATA_DIR=/app/data
RAG_SOURCE_DIR=/app/data/corpus/pdfs
RAG_INDEX_DIR=/app/data/index
RAG_CACHE_DIR=/app/data/cache
RAG_EXTRACT_DIR=/app/data/extracted_markdown
```

- [ ] **Step 2: 컨테이너 빌드 및 실행**

```bash
docker-compose up --build
```

Expected: `Uvicorn running on http://0.0.0.0:8000` 출력

- [ ] **Step 3: 앱 기동 시간 실측**

별도 터미널에서:

```bash
docker-compose logs app | grep "Uvicorn running"
```

출력된 타임스탬프로 실제 기동 소요 시간을 확인한다. 10초를 초과하면 `docker-compose.yml`의 `start_period` 값을 실측값 + 여유 5초로 조정한다.

- [ ] **Step 4: 앱 응답 확인**

```bash
curl http://localhost:8000/
```

Expected: HTML 응답 반환 (index.html)

- [ ] **Step 5: API 엔드포인트 확인**

```bash
curl http://localhost:8000/api/library
```

Expected: `{"documents": [], "count": 0}` 또는 유사한 JSON 응답

- [ ] **Step 6: healthcheck 상태 확인**

```bash
docker-compose ps
```

Expected: `STATUS` 컬럼에 `healthy` 표시 (start_period 이후)

- [ ] **Step 7: 볼륨 생성 확인**

```bash
docker volume ls | grep -E "pdf_data|index_data|cache_data|extract_data"
```

Expected: 4개 볼륨 출력 (프로젝트명 prefix가 붙어 `<project>_pdf_data` 형태로 나타남)

- [ ] **Step 8: 인덱스 수동 빌드 확인 (PDF가 있는 경우)**

> 주의: 인덱스 재빌드 중 API 서버와 SQLite 파일에 동시 쓰기가 발생할 수 있다. 운영 환경에서는 서비스 중단 후 실행 권장.

```bash
docker-compose exec app python scripts/build_index.py
```

Expected: 인덱싱 결과 JSON 출력

- [ ] **Step 9: 최종 커밋**

```bash
git add docs/superpowers/plans/2026-03-23-docker-containerization.md
git commit -m "docs: finalize Docker containerization implementation plan"
```
