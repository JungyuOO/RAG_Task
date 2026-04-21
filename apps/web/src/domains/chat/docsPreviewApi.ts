import type { DocumentPreviewResponse } from "@/domains/chat/types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

type ApiDocumentPreviewResponse = {
  source_path: string;
  relative_source_path: string;
  repo_relative_path: string;
  repo_locator: string;
  workspace_root_name: string;
  file_name: string;
  chunk_id: string;
  source_type: string;
  title: string;
  section_title: string;
  section_path: string[];
  anchor: string;
  page_number: number | null;
  line_start: number | null;
  line_end: number | null;
  source_locator: string;
  file_uri: string;
  shell_open_command: string;
  shell_open_label: string;
  vscode_uri: string;
  vscode_uri_with_line: string;
  code_command: string;
  snippet: string;
  lines: string[];
};

function mapPreview(input: ApiDocumentPreviewResponse): DocumentPreviewResponse {
  return {
    sourcePath: input.source_path,
    relativeSourcePath: input.relative_source_path,
    repoRelativePath: input.repo_relative_path,
    repoLocator: input.repo_locator,
    workspaceRootName: input.workspace_root_name,
    fileName: input.file_name,
    chunkId: input.chunk_id,
    sourceType: input.source_type,
    title: input.title,
    sectionTitle: input.section_title,
    sectionPath: input.section_path,
    anchor: input.anchor,
    pageNumber: input.page_number,
    lineStart: input.line_start,
    lineEnd: input.line_end,
    sourceLocator: input.source_locator,
    fileUri: input.file_uri,
    shellOpenCommand: input.shell_open_command,
    shellOpenLabel: input.shell_open_label,
    vscodeUri: input.vscode_uri,
    vscodeUriWithLine: input.vscode_uri_with_line,
    codeCommand: input.code_command,
    snippet: input.snippet,
    lines: input.lines,
  };
}

export async function fetchDocumentSnippet(sourcePath: string, chunkId = ""): Promise<DocumentPreviewResponse> {
  const search = new URLSearchParams({
    source_path: sourcePath,
    chunk_id: chunkId,
  });
  const response = await fetch(apiUrl(`/api/v1/docs-preview/snippet?${search.toString()}`));
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return mapPreview(JSON.parse(text) as ApiDocumentPreviewResponse);
}



