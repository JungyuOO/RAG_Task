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
- `requirements.txt`만 먼저 복사해 pip install 레이어 캐시 활용
- 설치 경로: `/install` (런타임 이미지로 복사할 대상)
- `--no-cache-dir` 옵션으로 pip 캐시 제거

**runtime 스테이지**
- `/install`에서 설치된 패키지만 복사 → 빌드 도구 미포함
- `app/` 소스 복사 (정적 파일 `app/web/`, `app/resources/` 포함)
- 비루트 유저 `appuser` 생성 및 전환 (보안)
- 작업 디렉토리: `/app`
- 노출 포트: `8000`
- 실행 명령: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### 이미지에 포함하지 않는 것
- `.env` 파일 (compose에서 `env_file`로 주입)
- `data/` 디렉토리 (볼륨으로 관리)
- `.git/`, `tests/`, `docs/`, `.venv/`

---

## docker-compose.yml

### 서비스 구성

```
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

### 환경 변수 (.env 경로 설정)

컨테이너 내부 경로와 맞추기 위해 `.env`에 아래 경로 설정 필요:

```
RAG_DATA_DIR=/app/data
RAG_SOURCE_DIR=/app/data/corpus/pdfs
RAG_INDEX_DIR=/app/data/index
RAG_CACHE_DIR=/app/data/cache
RAG_EXTRACT_DIR=/app/data/extracted_markdown
```

---

## .dockerignore

빌드 컨텍스트에서 제외할 항목:

```
.git
.venv
__pycache__
*.pyc
*.pyo
data/
.env
tests/
docs/
*.md (루트의 README, CLAUDE.md 등)
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
| Python 버전 | 3.13-slim | 최신 버전, slim으로 이미지 최소화 |
| 비루트 유저 | appuser | 컨테이너 보안 기본 원칙 |
