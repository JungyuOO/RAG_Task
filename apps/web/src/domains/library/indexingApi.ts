import type { BatchIndexRequest, BatchIndexResponse, BatchJobStatusResponse, BatchIndexItem } from "@/domains/library/types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

type ApiBatchIndexRequest = {
  root_path?: string;
  explicit_source_paths?: string[];
  source_type?: string | null;
  document_group?: string | null;
  version_tag?: string;
  locale?: string;
  max_files?: number;
  include_subdirectories?: boolean;
};

type ApiBatchIndexItem = {
  source_path: string;
  source_type: string;
  indexed: boolean;
  chunks: number;
  error: string;
};

type ApiBatchIndexResponse = {
  discovered_files: number;
  processed_files: number;
  indexed_files: number;
  failed_files: number;
  progress_pct: number;
  current_file: string;
  items: ApiBatchIndexItem[];
};

type ApiBatchJobStatusResponse = {
  job_id: string;
  task_type: string;
  status: string;
  request: ApiBatchIndexRequest;
  result: ApiBatchIndexResponse | null;
  error: string;
  progress_pct: number;
  current_file: string;
  created_at: string;
  updated_at: string;
};

function serializeRequest(request: BatchIndexRequest): ApiBatchIndexRequest {
  return {
    root_path: request.rootPath ?? "",
    explicit_source_paths: request.explicitSourcePaths ?? [],
    source_type: request.sourceType ?? null,
    document_group: request.documentGroup ?? null,
    version_tag: request.versionTag ?? "",
    locale: request.locale ?? "",
    max_files: request.maxFiles ?? 0,
    include_subdirectories: request.includeSubdirectories ?? true,
  };
}

function mapItem(input: ApiBatchIndexItem): BatchIndexItem {
  return {
    sourcePath: input.source_path,
    sourceType: input.source_type as BatchIndexItem["sourceType"],
    indexed: input.indexed,
    chunks: input.chunks,
    error: input.error,
  };
}

function mapBatchResponse(input: ApiBatchIndexResponse): BatchIndexResponse {
  return {
    discoveredFiles: input.discovered_files,
    processedFiles: input.processed_files,
    indexedFiles: input.indexed_files,
    failedFiles: input.failed_files,
    progressPct: input.progress_pct,
    currentFile: input.current_file,
    items: input.items.map(mapItem),
  };
}

function mapBatchJobStatus(input: ApiBatchJobStatusResponse): BatchJobStatusResponse {
  return {
    jobId: input.job_id,
    taskType: input.task_type,
    status: input.status,
    request: {
      rootPath: input.request.root_path,
      explicitSourcePaths: input.request.explicit_source_paths,
      sourceType: (input.request.source_type ?? undefined) as BatchIndexRequest["sourceType"],
      documentGroup: (input.request.document_group ?? undefined) as BatchIndexRequest["documentGroup"],
      versionTag: input.request.version_tag,
      locale: input.request.locale,
      maxFiles: input.request.max_files,
      includeSubdirectories: input.request.include_subdirectories,
    },
    result: input.result ? mapBatchResponse(input.result) : null,
    error: input.error,
    progressPct: input.progress_pct,
    currentFile: input.current_file,
    createdAt: input.created_at,
    updatedAt: input.updated_at,
  };
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return JSON.parse(text) as T;
}

export async function submitBatchIndexJob(request: BatchIndexRequest): Promise<BatchJobStatusResponse> {
  const response = await fetch(apiUrl("/api/v1/index/batch/jobs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(serializeRequest(request)),
  });
  return mapBatchJobStatus(await readJson<ApiBatchJobStatusResponse>(response));
}

export async function getBatchIndexJob(jobId: string): Promise<BatchJobStatusResponse> {
  const response = await fetch(apiUrl(`/api/v1/index/batch/jobs/${jobId}`));
  return mapBatchJobStatus(await readJson<ApiBatchJobStatusResponse>(response));
}

export async function retryFailedBatchIndexJob(jobId: string): Promise<BatchJobStatusResponse> {
  const response = await fetch(apiUrl(`/api/v1/index/batch/jobs/${jobId}/retry-failed`), {
    method: "POST",
  });
  return mapBatchJobStatus(await readJson<ApiBatchJobStatusResponse>(response));
}

export async function cancelBatchIndexJob(jobId: string): Promise<BatchJobStatusResponse> {
  const response = await fetch(apiUrl(`/api/v1/index/batch/jobs/${jobId}/cancel`), {
    method: "POST",
  });
  return mapBatchJobStatus(await readJson<ApiBatchJobStatusResponse>(response));
}

export async function listBatchIndexJobs(limit = 20): Promise<BatchJobStatusResponse[]> {
  const response = await fetch(apiUrl(`/api/v1/index/batch/jobs?limit=${limit}`));
  const payload = await readJson<{ jobs: ApiBatchJobStatusResponse[] }>(response);
  return payload.jobs.map(mapBatchJobStatus);
}



