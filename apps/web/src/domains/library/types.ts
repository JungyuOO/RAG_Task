export type SourceType = "html-single" | "html" | "pdf" | "generated-manual" | "structured-json";
export type DocumentGroup = "official_ocp" | "customer_generated";

export interface BatchIndexRequest {
  rootPath?: string;
  explicitSourcePaths?: string[];
  sourceType?: SourceType;
  documentGroup?: DocumentGroup;
  versionTag?: string;
  locale?: string;
  maxFiles?: number;
  includeSubdirectories?: boolean;
}

export interface BatchIndexItem {
  sourcePath: string;
  sourceType: SourceType;
  indexed: boolean;
  chunks: number;
  error: string;
}

export interface BatchIndexResponse {
  discoveredFiles: number;
  processedFiles: number;
  indexedFiles: number;
  failedFiles: number;
  progressPct: number;
  currentFile: string;
  items: BatchIndexItem[];
}

export interface BatchJobStatusResponse {
  jobId: string;
  taskType: string;
  status: string;
  request: BatchIndexRequest;
  result: BatchIndexResponse | null;
  error: string;
  progressPct: number;
  currentFile: string;
  createdAt: string;
  updatedAt: string;
}

export interface LibrarySourceBreakdownItem {
  label: string;
  count: number;
}

export interface LibrarySummaryResponse {
  sourceRoot: string;
  extractRoot: string;
  corpusFiles: number;
  manifestEntries: number;
  extractedArtifacts: number;
  indexedDocuments: number;
  indexedChunks: number;
  batchJobs: number;
  latestBatchStatus: string;
  sourceBreakdown: LibrarySourceBreakdownItem[];
  indexedSamples: string[];
  message: string;
}

export interface LibraryDocumentRecord {
  documentKey: string;
  title: string;
  relativePath: string;
  sourceType: string;
  group: "official";
  indexed: boolean;
  chunkCount: number;
  originalKind: "markdown" | "pdf";
  originalKey: string;
  description: string;
}

export interface LibraryCatalogResponse {
  documents: LibraryDocumentRecord[];
  message: string;
}

export interface LibraryChunkRecord {
  chunkId: string;
  chunkOrder: number | null;
  pageNumber: number | null;
  sectionTitle: string;
  blockTypes: string[];
  previewText: string;
}

export interface LibraryDocumentChunksResponse {
  documentKey: string;
  title: string;
  chunkCount: number;
  chunks: LibraryChunkRecord[];
}

export interface LibraryDocumentContentResponse {
  documentKey: string;
  title: string;
  content: string;
}

