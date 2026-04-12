# 고객사 pandas 운영 메뉴얼

## 적용 범위
- 네임스페이스: `demo`
- 대상 애플리케이션: `pandas`

## 기본 점검 절차
고객사 환경에서 pandas 관련 Pod 상태를 먼저 확인합니다.

```bash
oc get pods -n demo | grep pandas
```

필요하면 노드 정보까지 함께 확인합니다.

```bash
oc get pods -n demo -o wide | grep pandas
```

## 특정 Pod YAML 확인
문제가 발생한 Pod 이름을 알고 있으면 YAML을 확인합니다.

```bash
oc get pod <pod_name> -n demo -o yaml
```

추가 상태 확인이 필요하면 describe 명령을 사용합니다.

```bash
oc describe pod <pod_name> -n demo
```

## 경고 이벤트 확인
경고 이벤트는 demo 네임스페이스 기준으로 확인합니다.

```bash
oc get events -n demo --field-selector type=Warning
```

## 운영 메모
- pandas 관련 Pod 조회는 기본적으로 `demo` 네임스페이스 기준으로 설명합니다.
- 고객사 메뉴얼 기준 응답이 필요하면 질문에 `고객사 메뉴얼 기준으로`를 포함합니다.
