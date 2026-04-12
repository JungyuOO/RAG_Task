"""OCP 운영 매뉴얼 가상 문서를 LLM으로 생성하는 스크립트.

Usage:
    python scripts/generate_ocp_manuals.py --version 4.15 --topics pod,service,pvc --output data/corpus/pdfs/generated/
"""
import argparse
import asyncio
import os

MANUAL_TOPICS = [
    {"topic": "Pod 배포 및 관리", "sections": ["프로젝트 생성", "Pod YAML 작성", "배포", "스케일링", "롤백"]},
    {"topic": "Service 네트워킹", "sections": ["Service 유형", "ClusterIP", "NodePort", "LoadBalancer", "Ingress"]},
    {"topic": "PV/PVC 스토리지", "sections": ["StorageClass", "PV 생성", "PVC 바인딩", "동적 프로비저닝"]},
    {"topic": "RBAC 권한 관리", "sections": ["Role", "RoleBinding", "ServiceAccount", "SCC"]},
    {"topic": "Operator 설치 및 관리", "sections": ["OperatorHub", "Subscription", "CSV", "업그레이드"]},
    {"topic": "클러스터 모니터링", "sections": ["Prometheus", "Grafana", "AlertManager", "메트릭 수집"]},
    {"topic": "CI/CD 파이프라인", "sections": ["BuildConfig", "ImageStream", "Pipeline", "Tekton"]},
    {"topic": "보안 정책", "sections": ["NetworkPolicy", "Pod Security", "이미지 서명", "취약점 스캔"]},
]

GENERATION_PROMPT = """당신은 OpenShift Container Platform {version} 운영 전문가입니다.
다음 주제에 대한 실제 현장 운영 매뉴얼을 작성하세요.

주제: {topic}
버전: OCP {version}
섹션: {sections}

## 요구사항:
1. 실제 고객사에서 사용하는 것처럼 구체적인 YAML 예시, 명령어, 설정값을 포함
2. 각 단계별로 주의사항과 트러블슈팅 팁 포함
3. 마크다운 형식으로 작성 (# 제목, ## 소제목, ```yaml 코드블록)
4. 최소 2000자 이상
5. 한국어로 작성
"""


class ManualGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def generate_one(self, topic: str, version: str, doc_type: str = "operation_manual", sections: list = None) -> str:
        sections_str = ", ".join(sections or [])
        system_prompt = GENERATION_PROMPT.format(topic=topic, version=version, sections=sections_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{topic} 운영 매뉴얼을 작성해주세요."},
        ]
        return await self.llm.generate(messages=messages)

    async def generate_batch(self, topics: list, version: str, output_dir: str) -> list:
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for item in topics:
            content = await self.generate_one(item["topic"], version, sections=item.get("sections", []))
            safe_topic = item['topic'].replace(' ', '-').replace('/', '-').replace('\\', '-')
            filename = f"ocp-{version}-{safe_topic}.md"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            results.append(filepath)
        return results


async def main():
    parser = argparse.ArgumentParser(description="OCP 운영 매뉴얼 가상 문서 생성 및 인덱싱")
    parser.add_argument("--version", default="4.15", help="OCP 버전 (예: 4.15)")
    parser.add_argument("--output", default="data/corpus/pdfs/generated/", help="출력 디렉토리")
    parser.add_argument("--topics", default=None, help="쉼표로 구분된 주제 필터 (예: Pod,Service)")
    parser.add_argument("--index", action="store_true", help="생성 후 RAG 인덱싱까지 자동 수행")
    args = parser.parse_args()

    from app.rag.llm import LlmClient
    from app.config import get_settings
    settings = get_settings()
    llm = LlmClient(settings)

    gen = ManualGenerator(llm_client=llm)
    topics = MANUAL_TOPICS
    if args.topics:
        filter_set = set(args.topics.split(","))
        topics = [t for t in topics if any(f in t["topic"] for f in filter_set)]

    files = await gen.generate_batch(topics, args.version, args.output)
    print(f"Generated {len(files)} manuals:")
    for f in files:
        print(f"  {f}")

    if args.index and files:
        from pathlib import Path
        from app.rag.pipeline import RagPipeline
        pipeline = RagPipeline(settings)
        indexing_service = pipeline.indexing_service
        print("\n[인덱싱 시작]")
        for f in files:
            path = Path(f)
            result = indexing_service.index_markdown_file(path, doc_type="operation_manual")
            print(f"  {path.name} → chunks={result['indexed_chunks']}, skipped={result['skipped']}")
        print("인덱싱 완료.")


if __name__ == "__main__":
    asyncio.run(main())
