# OCP Web Connection Notes

## 목적

웹에서 OpenShift 연결 정보를 입력할 때 필요한 기본 운영 메모.

## 권장 입력 항목

- Cluster API URL
- Auth mode
  - Bearer token
  - Username / Password
  - OAuth (future)
- Default namespace
- SSL verify 여부

## 우선 권장 방식

1. **Bearer token 직접 입력**
   - 가장 단순
   - 세션 단위 연결에 적합
   - 현재 scaffold 기준 가장 먼저 지원해야 하는 방식

2. **Username / Password**
   - 백엔드에서 token exchange를 수행해야 함
   - 클러스터 auth 설정에 따라 동작 가능 여부가 달라질 수 있음
   - 현재 scaffold에서는 UI/계약만 열어두고, exchange hook은 후속 구현 대상으로 둠

## Cluster URL 메모

- 일반적으로 API server URL은 `https://api.<cluster-domain>:6443` 형태를 사용
- 사용자는 보통 아래 두 경로로 URL을 알 수 있음:
  - 콘솔의 login command / kubeconfig
  - 운영팀이 전달한 cluster API endpoint

## Token 발급/확인 메모

- 가장 현실적인 초기 UX는 “이미 발급된 token을 붙여넣기”다.
- 운영자가 얻는 대표 경로:
  - OpenShift 콘솔의 login command
  - `oc whoami -t`
  - 서비스 계정 token

## 후속 구현 메모

- password mode는 백엔드에서만 처리
- 비밀번호는 프런트 영구 저장 금지
- profile에는 secret reference만 저장
- 실제 비밀값은 encrypted/session-scoped secret store에 저장

## Official References

- OpenShift OAuth endpoint / default clients overview: https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/authentication_and_authorization/configuring-internal-oauth
- OpenShift API authentication and bearer token usage: https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html-single/oauth_apis/oauth_apis
