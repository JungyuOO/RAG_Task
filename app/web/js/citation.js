function createCitationRenderer() {
  const seen = new Map();
  let counter = 0;

  function nextLabel(key) {
    if (!seen.has(key)) {
      counter += 1;
      seen.set(key, counter);
    }
    return "[" + seen.get(key) + "]";
  }

  function renderRef(fileName, page, lineStart, lineEnd) {
    const key = [fileName, page, lineStart || "", lineEnd || ""].join("|");
    const label = nextLabel(key);
    const sourcePath = resolveCitationSourcePath(fileName, page);
    const attrs = [
      `type="button"`,
      `class="source-ref-inline"`,
      `data-file="${escapeHtml(fileName)}"`,
      `data-page="${escapeHtml(String(page))}"`,
      `title="${escapeHtml(fileName)} p.${page}${lineStart ? ` L${lineStart}-${lineEnd}` : ""}"`,
    ];
    if (sourcePath) attrs.push(`data-source-path="${escapeHtml(sourcePath)}"`);
    if (lineStart) attrs.push(`data-line-start="${escapeHtml(String(lineStart))}"`);
    if (lineEnd) attrs.push(`data-line-end="${escapeHtml(String(lineEnd))}"`);
    return `<button ${attrs.join(" ")}>${label}</button>`;
  }

  return { renderRef };
}

function resolveCitationSourcePath(fileName, page) {
  const payload = currentContextPayload || {};
  const normalizedFile = String(fileName || "").toLowerCase();
  const normalizedPage = parseInt(page, 10) || 0;

  const answerCitations = Array.isArray(payload.answer_citations) ? payload.answer_citations : [];
  for (const citation of answerCitations) {
    if (String(citation.file_name || "").toLowerCase() !== normalizedFile) continue;
    if ((parseInt(citation.page_number, 10) || 0) !== normalizedPage) continue;
    if (citation.source_path) return String(citation.source_path);
  }

  const previewPages = Array.isArray(payload.preview_pages) ? payload.preview_pages : [];
  for (const pageItem of previewPages) {
    const sourcePath = String(pageItem.source_path || "");
    const candidateFile = sourcePath.split(/[\\/]/).pop() || "";
    if (candidateFile.toLowerCase() !== normalizedFile) continue;
    if ((parseInt(pageItem.page_number, 10) || 0) !== normalizedPage) continue;
    return sourcePath;
  }

  const items = Array.isArray(payload.items) ? payload.items : [];
  for (const item of items) {
    const sourcePath = String(item.source_path || "");
    const candidateFile = sourcePath.split(/[\\/]/).pop() || "";
    if (candidateFile.toLowerCase() !== normalizedFile) continue;
    const pageStart = parseInt(item.page_start, 10) || parseInt(item.page_number, 10) || 0;
    const pageEnd = parseInt(item.page_end, 10) || pageStart;
    if (normalizedPage >= pageStart && normalizedPage <= pageEnd && sourcePath) {
      return sourcePath;
    }
  }

  return "";
}

function resolveCitationAnchor(fileName, page) {
  const payload = currentContextPayload || {};
  const normalizedFile = String(fileName || "").toLowerCase();
  const normalizedPage = parseInt(page, 10) || 0;
  const answerCitations = Array.isArray(payload.answer_citations) ? payload.answer_citations : [];
  for (const citation of answerCitations) {
    if (String(citation.file_name || "").toLowerCase() !== normalizedFile) continue;
    if ((parseInt(citation.page_number, 10) || 0) !== normalizedPage) continue;
    if (citation.block_anchor || citation.html_anchor) {
      return String(citation.block_anchor || citation.html_anchor || "");
    }
  }
  const previewPages = Array.isArray(payload.preview_pages) ? payload.preview_pages : [];
  for (const pageItem of previewPages) {
    const sourcePath = String(pageItem.source_path || "");
    const candidateFile = sourcePath.split(/[\\/]/).pop() || "";
    if (candidateFile.toLowerCase() !== normalizedFile) continue;
    if ((parseInt(pageItem.page_number, 10) || 0) !== normalizedPage) continue;
    return String(pageItem.block_anchor || pageItem.html_anchor || "");
  }
  return "";
}

function renderCitationTags(text) {
  let rendered = String(text || "");
  const renderer = createCitationRenderer();
  const filePattern = "[^\\[\\]\\n]+?\\.[A-Za-z0-9]{2,8}";

  const sourcePattern = /\[source:([^:]+):p(\d+):L(\d+)-(\d+)\]/g;
  rendered = rendered.replace(sourcePattern, (match, fileName, page, lineStart, lineEnd) => {
    return renderer.renderRef(fileName, page, lineStart, lineEnd);
  });

  const groupedPattern = new RegExp(`\\[(${filePattern})\\]\\s*((?:p\\.\\d+(?:-\\d+)?)(?:\\s*,\\s*p\\.\\d+(?:-\\d+)?)*)`, "gi");
  rendered = rendered.replace(groupedPattern, (match, fileName, pagesText) => {
    const parts = [];
    const pageMatches = pagesText.match(/p\.(\d+)(?:-\d+)?/gi) || [];
    pageMatches.forEach((token) => {
      const pageMatch = /p\.(\d+)/i.exec(token);
      if (!pageMatch) return;
      parts.push(renderer.renderRef(fileName, pageMatch[1]));
    });
    return parts.join(" ");
  });

  const bracketPattern = new RegExp(`\\[(${filePattern})\\]\\s*p\\.(\\d+)(?:-(\\d+))?`, "gi");
  rendered = rendered.replace(bracketPattern, (match, fileName, pageStart) => {
    return renderer.renderRef(fileName, pageStart);
  });

  const parenPattern = new RegExp(`\\((${filePattern})\\s+p\\.(\\d+)(?:-(\\d+))?\\)`, "gi");
  rendered = rendered.replace(parenPattern, (match, fileName, pageStart) => {
    return renderer.renderRef(fileName, pageStart);
  });

  return rendered;
}

function bindCitationClicks(container) {
  container.querySelectorAll(".source-ref-inline").forEach((tag) => {
    tag.addEventListener("click", () => {
      const fileName = tag.dataset.file;
      const page = parseInt(tag.dataset.page, 10);
      const blockAnchor = resolveCitationAnchor(fileName, page);
      openAnswerPreviewSource(fileName, page, blockAnchor, tag.dataset.sourcePath || "");
    });
  });
}
