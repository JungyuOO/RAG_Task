from __future__ import annotations

from pathlib import Path

from apps.api.core.llm_settings import get_chat_llm_settings
from apps.api.persistence.sqlite import (
    SQLiteActionAuditRepository,
    SQLiteActionExecutionRepository,
    SQLiteActionRequestRepository,
    SQLiteBatchJobRepository,
    SQLiteConnectionProfileStore,
)
from apps.api.core.runtime_persistence import load_runtime_persistence_profile
from apps.api.integrations.ocp import (
    ConnectedOcpService,
    LiveOcpChatService,
    OcpActionAuditService,
    OcpActionExecutionService,
    OcpActionPolicyService,
    OcpActionPreviewService,
    OcpActionRequestService,
)
from apps.api.integrations.ocp.auth import OcpConnectionBroker, OcpConnectionVerifier, build_default_lease_scheduler
from apps.api.integrations.ocp.auth.broker import build_default_connection_secret_store
from apps.api.rag.generation.citation_grounding import CitationGroundingValidator
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient
from apps.api.rag.generation.response_cache import ChatResponseCache
from apps.api.rag.generation.unified_copilot_service import UnifiedCopilotService
from apps.api.rag.indexing import BatchIndexingService, BatchIndexJobService
from apps.api.rag.query.intent_agent import IntentAgent
from apps.api.rag.query.question_normalizer import QuestionNormalizer
from apps.api.rag.query.query_rewrite_agent import QueryRewriteAgent
from apps.api.rag.retrieval import DocumentRetriever, PgvectorRetrievalBridge

runtime_persistence_profile = load_runtime_persistence_profile()
STATE_ROOT = runtime_persistence_profile.state_root
STATE_DB_PATH = runtime_persistence_profile.state_db_path

connection_secret_store = build_default_connection_secret_store(
    storage_path=runtime_persistence_profile.secret_storage_path,
    refs_path=runtime_persistence_profile.secret_refs_path,
    backend=runtime_persistence_profile.secret_backend or None,
)
connection_profile_store = SQLiteConnectionProfileStore(
    db_path=STATE_DB_PATH,
    legacy_json_path=runtime_persistence_profile.legacy_json_path("connection_profiles.json"),
)
connection_broker = OcpConnectionBroker(secret_store=connection_secret_store, profile_store=connection_profile_store)
connection_lease_scheduler = build_default_lease_scheduler(broker=connection_broker)
connection_verifier = OcpConnectionVerifier()
connected_ocp_service = ConnectedOcpService()
action_policy_service = OcpActionPolicyService()
action_audit_repository = SQLiteActionAuditRepository(
    db_path=STATE_DB_PATH,
    legacy_json_path=runtime_persistence_profile.legacy_json_path("action_audit.json"),
)
action_audit_service = OcpActionAuditService(repository=action_audit_repository)
action_preview_service = OcpActionPreviewService(policy_service=action_policy_service)
action_request_repository = SQLiteActionRequestRepository(
    db_path=STATE_DB_PATH,
    legacy_json_path=runtime_persistence_profile.legacy_json_path("action_requests.json"),
)
action_request_service = OcpActionRequestService(
    preview_service=action_preview_service,
    repository=action_request_repository,
    audit_service=action_audit_service,
)
action_execution_repository = SQLiteActionExecutionRepository(
    db_path=STATE_DB_PATH,
    legacy_json_path=runtime_persistence_profile.legacy_json_path("action_executions.json"),
)
action_execution_service = OcpActionExecutionService(
    request_service=action_request_service,
    broker=connection_broker,
    repository=action_execution_repository,
    audit_service=action_audit_service,
)
batch_job_repository = SQLiteBatchJobRepository(
    db_path=STATE_DB_PATH,
    legacy_json_path=runtime_persistence_profile.legacy_json_path("batch_jobs.json"),
)
batch_indexing_service = BatchIndexingService()
batch_job_service = BatchIndexJobService(batch_service=batch_indexing_service, job_repository=batch_job_repository)
chat_llm_settings = get_chat_llm_settings()
chat_llm_client = OpenAiCompatibleLlmClient(chat_llm_settings)
intent_agent = IntentAgent(llm_client=chat_llm_client)
question_normalizer = QuestionNormalizer(llm_client=chat_llm_client)
query_rewrite_agent = QueryRewriteAgent(llm_client=chat_llm_client)
citation_validator = CitationGroundingValidator()
chat_response_cache = ChatResponseCache()
pgvector_bridge = PgvectorRetrievalBridge()
document_retriever = DocumentRetriever(use_char_ngram_sparse=chat_llm_settings.use_char_ngram_sparse)
live_ocp_chat_service = LiveOcpChatService(
    connected_ocp_service,
    document_retriever=document_retriever,
    llm_client=chat_llm_client,
)
pgvector_bridge._runtime.set_asymmetric_enabled(chat_llm_settings.use_asymmetric_query_prompt)
unified_copilot_service = UnifiedCopilotService(
    live_chat_service=live_ocp_chat_service,
    document_retriever=document_retriever,
    pgvector_bridge=pgvector_bridge,
    intent_agent=intent_agent,
    question_normalizer=question_normalizer,
    query_rewrite_agent=query_rewrite_agent,
    citation_validator=citation_validator,
    response_cache=chat_response_cache,
    llm_client=chat_llm_client,
)


