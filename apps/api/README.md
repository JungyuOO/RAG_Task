# apps/api

FastAPI 백엔드 기본 작업공간.

현재 기본 구조:
- `routes/v1` - HTTP route 계층
- `services` - 도메인/오케스트레이션 서비스
- `schemas` - API / 내부 계약
- `repositories` - persistence 계층
- `runtime.py` - runtime wiring
- `main.py` - default app entrypoint

현재 상태:
- backend import path는 `apps.api...` 기준으로 flatten 완료
- default runtime entrypoint는 `apps.api.main:app`
- 기존 중첩 구조 `apps/api/app/api` 는 `apps/api/routes` 로 정리 완료
