# auth feature

예정 UI:

- Cluster URL 입력
- Auth mode 선택
  - Token
  - ID / Password
  - OAuth (future)
- Token 발급 방법 안내
- Connection Test
- 현재 연결된 cluster / user / namespace 표시

주의:
- password는 프런트 영구 저장 금지
- token도 localStorage 장기 저장 기본 비허용
- UI는 세션 연결 상태만 보여주고, 민감정보는 backend에서만 처리
