# Docker 컨테이너화 설계 문서

**날짜:** 2026-03-23
**대상 프로젝트:** RAG Task — FastAPI 기반 커스텀 RAG 시스템

---

## 목표

- 단일 Docker 이미지로 앱 전체를 실행 가능하게 컨테이너화
- 역할별 볼륨 분리로 OCP(OpenShift Container Platform) 마이그레이션 경로 확보
- 단일 `docker-compose.yml` 구성 (개발/운영 분리 없음)

---

## Dockerfile — 멀티 스테이지 빌드

### 스테이지 구성

| 스테이지 | 베이스 이미지 | 역할 |
|---|---|---|
| `builder` | `python:3.13-slim` | 의존성 설치만 담당 |
| `runtime` | `python:3.13-slim` | 패키지 복사 + 앱 실행 |

### 세부 설계

**builder 스테이지**
- `apt-get install build-essential` 등 네이티브 빌드 도구 설치 (pymupdf wheel 빌드 대비)
  - pymupdf 최신 버전은 pre-built wheel을 제공하는 경우가 많지만, 3.13 지원 wheel이 없을 경우 빌드 도구가 필요
- `requirements.txt`만 먼저 복사해 pip install 레이어 캐시 활용
- 설치 경로: `/install` (런타임 이미지로 복사할 대상)
- `--no-cache-dir` 옵션으로 pip 캐시 제거

**runtime 스테이지**
- `/install`에서 설치된 패키지만 복사 → 빌드 도구 미포함
- 프로젝트 루트 전체를 `/app`에 복사 (`COPY . /app/`)
  - 작업 디렉토리 `/app` 기준으로 `"app/web"`, `"app/resources"` 상대 경로가 `/app/app/web`, `/app/app/resources`로 정상 해석됨
  - `scripts/build_index.py`도 포함되어 컨테이너 내에서 인덱스 수동 빌드 가능
- 비루트 유저 `appuser` 생성 및 전환 (보안)
- 볼륨 마운트 대상 경로(`/app/data/*`) 소유권을 `appuser`로 사전 설정
  ```
  RUN mkdir -p /app/data/corpus/pdfs /app/data/index /app/data/cache /app/data/extracted_markdown \
      && chown -R appuser:appuser /app/data
  ```
- 작업 디렉토리: `/app`
- 노출 포트: `8000`
- 실행 명령: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### 이미지 버전 고정 정책
- 재현 가능한 빌드를 위해 `python:3.13-slim` 대신 패치 버전 고정 권장
  - 예: `python:3.13.3-slim`

### 이미지에 포함하지 않는 것
- `.env` 파일 (compose에서 `env_file`로 주입)
- `data/` 디렉토리 (볼륨으로 관리)

---

## docker-compose.yml

### 서비스 구성

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - pdf_data:/app/data/corpus/pdfs
      - index_data:/app/data/index
      - cache_data:/app/data/cache
      - extract_data:/app/data/extracted_markdown
    healthcheck:
      # python:3.13-slim에는 curl이 포함되지 않으므로 Python 내장 urllib 사용
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/library')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s  # 첫 배포 후 기동 시간을 실측하여 조정 권장

volumes:
  pdf_data:
  index_data:
  cache_data:
  extract_data:
```

### 볼륨 역할 및 OCP 대응

| 볼륨명 | 컨테이너 경로 | 영속 필요 | OCP 이전 시 |
|---|---|---|---|
| `pdf_data` | `/app/data/corpus/pdfs` | 필수 (원본 PDF) | RWX PVC (API서버 + 인덱싱 워커 공유) |
| `index_data` | `/app/data/index` | 필수 (SQLite, 세션 이력) | RWO PVC → 장기적으로 DB 교체 지점 |
| `cache_data` | `/app/data/cache` | 선택 (재생성 가능) | ephemeral 또는 Redis 교체 지점 |
| `extract_data` | `/app/data/extracted_markdown` | 선택 (재생성 가능) | ephemeral 또는 pdf_data 볼륨에 병합 |

> **주의:** 인덱스 재빌드(`build_index.py`) 실행 중에는 API 서버와 동일한 SQLite 파일에 동시 쓰기가 발생할 수 있습니다. 재빌드 시에는 서비스 중단 또는 쓰기 잠금 상태에서 진행하는 것을 권장합니다 (SQLite WAL 모드로 어느 정도 완화되지만 완전하지 않음).

### 환경 변수 (.env 경로 설정)

컨테이너 내부 경로와 맞추기 위해 `.env`에 아래 경로 설정 필요:

```
RAG_DATA_DIR=/app/data
RAG_SOURCE_DIR=/app/data/corpus/pdfs
RAG_INDEX_DIR=/app/data/index
RAG_CACHE_DIR=/app/data/cache
RAG_EXTRACT_DIR=/app/data/extracted_markdown
```

> **구현 시 반드시:** `.env.example`에 `RAG_EXTRACT_DIR` 항목이 누락되어 있으므로 함께 추가해야 함.

---

## .dockerignore

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

---

## 실행 방법

```bash
# 이미지 빌드 및 컨테이너 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up --build -d

# 인덱스 수동 빌드 (컨테이너 내부)
docker-compose exec app python scripts/build_index.py

# 로그 확인
docker-compose logs -f app
```

---

## 트레이드오프 및 결정 사항

| 결정 | 선택 | 이유 |
|---|---|---|
| 스테이지 구성 | 멀티 스테이지 | 이미지 경량화, OCP pull 속도 개선 |
| 볼륨 전략 | 역할별 분리 | OCP 마이그레이션 시 PVC 1:1 대응 |
| 환경 구성 | 단일 compose | 과제 범위, 불필요한 복잡도 제거 |
| Python 버전 | 3.13-slim (패치 버전 고정 권장) | 최신 버전, slim으로 이미지 최소화 |
| 비루트 유저 | appuser | 컨테이너 보안 기본 원칙 |
| 소스 복사 방식 | `COPY . /app/` (루트 전체) | StaticFiles 상대 경로 정상 동작 보장 |
| healthcheck | `/api/library` 엔드포인트 | OCP liveness/readiness probe 이전 시 활용 |
