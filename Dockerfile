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
