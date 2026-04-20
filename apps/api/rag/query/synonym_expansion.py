from __future__ import annotations

import re

ACRONYM_EXPANSIONS: dict[str, str] = {
    "rbac": "rbac role binding cluster role clusterrolebinding unauthenticated groups authentication authorization",
    "oauth": "oauth authentication identity provider serviceaccount events openshift-authentication",
    "pvc": "pvc persistent volume claim",
    "pv": "pv persistent volume",
    "mtu": "mtu maximum transmission unit cluster network",
    "etcd": "etcd key value store",
    "olm": "olm operator lifecycle manager",
    "gitops": "gitops argocd application red hat openshift gitops operator workflow",
    "oc": "oc openshift command line",
    "csi": "csi container storage interface",
    "crd": "crd custom resource definition",
    "scc": "scc security context constraints",
    "nlb": "network load balancer nlb ingress controller",
    "clb": "classic load balancer clb ingress controller",
    "nodeport": "nodeport node port service range",
    "cidr": "cidr network range cluster network",
    "sctp": "sctp stream control transmission protocol enabling",
    "nmstate": "nmstate kubernetes nmstate operator nodenetworkstate",
}

PHRASE_EXPANSIONS: dict[str, str] = {
    "인증": "authentication oauth identity provider openshift-authentication",
    "인가": "authorization rbac role binding cluster role",
    "권한": "authorization rbac cluster role binding permissions",
    "라우트": "route ingress openshift route",
    "깃옵스": "gitops argocd openshift gitops operator",
    "git ops": "gitops argocd openshift gitops operator",
    "업데이트": "update updates upgrade openshift update introduction",
    "업그레이드": "upgrade update updates",
    "이미지 빌드": "image build builds buildconfig understanding",
    "image build": "image build builds buildconfig understanding",
    "레지스트리": "registry integrated image registry",
    "프로젝트": "project projects namespace working",
    "볼륨 마운트": "volume mount mount points cross volume",
    "cross project": "cross project providing access jenkins",
    "cross volume": "cross volume mount jenkins points",
    "토큰 리스트": "oauth access token listing user-owned",
    "토큰 뽑": "oauth access tokens listing",
    "포트 범위": "port range node port service range",
    "네트워크 범위": "network range cluster network cidr",
    "cidr": "cidr network range",
    "활성화": "enable enabling activating",
    "유효 기간": "duration validity period token duration",
    "기존": "existing already configured",
    "새로": "new creating install fresh",
    "설치": "install installing installation",
    "개요": "overview about understanding introduction",
    "machine api": "machine api overview cluster api",
    "내장": "integrated built-in",
    "범위": "range service port configuring",
    "바꾸고": "switch change switching migration",
    "volume mount": "volume mount mount points cross volume",
    "classic load balancer": "classic load balancer clb ingress controller",
    "network load balancer": "network load balancer nlb ingress controller",
}

PHRASE_SUPPRESSED_ACRONYMS: dict[str, set[str]] = {
    "토큰 리스트": {"oauth"},
    "토큰 뽑": {"oauth"},
    "token list": {"oauth"},
    "access token listing": {"oauth"},
}

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9가-힣_-]+")


def expand_acronyms(query: str) -> str:
    text = str(query or "")
    if not text.strip():
        return text
    additions: list[str] = []
    seen: set[str] = set()
    suppressed_tokens: set[str] = set()
    lowered = text.casefold()
    for phrase, expansion in PHRASE_EXPANSIONS.items():
        if phrase.casefold() in lowered and expansion not in additions:
            additions.append(expansion)
            suppressed_tokens.update(PHRASE_SUPPRESSED_ACRONYMS.get(phrase, set()))
    for token in _TOKEN_PATTERN.findall(text.casefold()):
        expansion = ACRONYM_EXPANSIONS.get(token)
        if expansion and token not in seen and token not in suppressed_tokens:
            additions.append(expansion)
            seen.add(token)
    if not additions:
        return text
    return f"{text} " + " ".join(additions)
