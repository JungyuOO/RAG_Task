import type {
  LibraryCatalogResponse,
  LibraryDocumentChunksResponse,
  LibraryDocumentContentResponse,
  LibrarySummaryResponse,
} from "@/features/library/types";

type ApiLibrarySourceBreakdownItem = {
  label: string;
  count: number;
};

type ApiLibrarySummaryResponse = {
  source_root: string;
  extract_root: string;
  corpus_files: number;
  manifest_entries: number;
  extracted_artifacts: number;
  indexed_documents: number;
  indexed_chunks: number;
  batch_jobs: number;
  latest_batch_status: string;
  source_breakdown: ApiLibrarySourceBreakdownItem[];
  indexed_samples: string[];
  message: string;
};

type ApiLibraryDocumentRecord = {
  document_key: string;
  title: string;
  relative_path: string;
  source_type: string;
  group: "official";
  indexed: boolean;
  chunk_count: number;
  original_kind: "markdown" | "pdf";
  original_key: string;
  description: string;
};

type ApiLibraryCatalogResponse = {
  documents: ApiLibraryDocumentRecord[];
  message: string;
};

type ApiLibraryChunkRecord = {
  chunk_id: string;
  chunk_order: number | null;
  page_number: number | null;
  section_title: string;
  block_types: string[];
  preview_text: string;
};

type ApiLibraryDocumentChunksResponse = {
  document_key: string;
  title: string;
  chunk_count: number;
  chunks: ApiLibraryChunkRecord[];
};

type ApiLibraryDocumentContentResponse = {
  document_key: string;
  title: string;
  content: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return JSON.parse(text) as T;
}

function mapSummary(input: ApiLibrarySummaryResponse): LibrarySummaryResponse {
  return {
    sourceRoot: input.source_root,
    extractRoot: input.extract_root,
    corpusFiles: input.corpus_files,
    manifestEntries: input.manifest_entries,
    extractedArtifacts: input.extracted_artifacts,
    indexedDocuments: input.indexed_documents,
    indexedChunks: input.indexed_chunks,
    batchJobs: input.batch_jobs,
    latestBatchStatus: input.latest_batch_status ?? "",
    sourceBreakdown: input.source_breakdown ?? [],
    indexedSamples: input.indexed_samples ?? [],
    message: input.message ?? "",
  };
}

function mapDocument(input: ApiLibraryDocumentRecord) {
  return {
    documentKey: input.document_key,
    title: input.title,
    relativePath: input.relative_path,
    sourceType: input.source_type,
    group: input.group,
    indexed: input.indexed,
    chunkCount: input.chunk_count,
    originalKind: input.original_kind,
    originalKey: input.original_key,
    description: input.description,
  };
}

function mapCatalog(input: ApiLibraryCatalogResponse): LibraryCatalogResponse {
  return {
    documents: (input.documents ?? []).map(mapDocument),
    message: input.message ?? "",
  };
}

function mapChunks(input: ApiLibraryDocumentChunksResponse): LibraryDocumentChunksResponse {
  return {
    documentKey: input.document_key,
    title: input.title,
    chunkCount: input.chunk_count,
    chunks: (input.chunks ?? []).map((chunk) => ({
      chunkId: chunk.chunk_id,
      chunkOrder: chunk.chunk_order,
      pageNumber: chunk.page_number,
      sectionTitle: chunk.section_title,
      blockTypes: chunk.block_types ?? [],
      previewText: chunk.preview_text,
    })),
  };
}

function mapContent(input: ApiLibraryDocumentContentResponse): LibraryDocumentContentResponse {
  return {
    documentKey: input.document_key,
    title: input.title,
    content: input.content,
  };
}

export async function getLibrarySummary(): Promise<LibrarySummaryResponse> {
  const response = await fetch(apiUrl("/api/v1/library/summary"));
  return mapSummary(await readJson<ApiLibrarySummaryResponse>(response));
}

export async function getLibraryCatalog(): Promise<LibraryCatalogResponse> {
  const response = await fetch(apiUrl("/api/v1/library/catalog"));
  return mapCatalog(await readJson<ApiLibraryCatalogResponse>(response));
}

export async function getLibraryDocumentChunks(documentKey: string): Promise<LibraryDocumentChunksResponse> {
  const search = new URLSearchParams({ document_key: documentKey });
  const response = await fetch(apiUrl(`/api/v1/library/chunks?${search.toString()}`));
  return mapChunks(await readJson<ApiLibraryDocumentChunksResponse>(response));
}

export async function getLibraryDocumentContent(documentKey: string): Promise<LibraryDocumentContentResponse> {
  const search = new URLSearchParams({ document_key: documentKey });
  const response = await fetch(apiUrl(`/api/v1/library/document-content?${search.toString()}`));
  return mapContent(await readJson<ApiLibraryDocumentContentResponse>(response));
}

export function libraryDocumentFileUrl(documentKey: string): string {
  const search = new URLSearchParams({ document_key: documentKey });
  return apiUrl(`/api/v1/library/document-file?${search.toString()}`);
}


