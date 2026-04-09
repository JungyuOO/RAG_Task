// app/web/js/citation.js

function _sourceLabel(fileName, page) {
  const lowered = String(fileName || "").toLowerCase();
  const kind = lowered.includes("customer-guide") ? "고객사 메뉴얼" : "공식문서";
  return `> ${kind} p.${page}`;
}

function _sourceRefTag(_label, fileName, page) {
  const label = _sourceLabel(fileName, page);
  return `<button type="button" class="source-ref-inline" data-file="${fileName}" data-page="${page}" title="${fileName} p.${page}">${label}</button>`;
}

function renderCitationTags(text) {
  let rendered = String(text || "");

  const sourcePattern = /\[source:([^:]+):p(\d+):L(\d+)-(\d+)\]/g;
  rendered = rendered.replace(sourcePattern, (match, fileName, page, lineStart, lineEnd) => {
    return `<button type="button" class="source-ref-inline" data-file="${fileName}" data-page="${page}" data-line-start="${lineStart}" data-line-end="${lineEnd}" title="${fileName} p.${page} L${lineStart}-${lineEnd}">&gt;</button>`;
  });

  const groupedPattern = /\[([^\[\]\n]+?\.pdf)\]\s*((?:p\.\d+(?:-\d+)?)(?:\s*,\s*p\.\d+(?:-\d+)?)*)/gi;
  rendered = rendered.replace(groupedPattern, (match, fileName, pagesText) => {
    const parts = [];
    const pageMatches = pagesText.match(/p\.(\d+)(?:-\d+)?/gi) || [];
    pageMatches.forEach((token) => {
      const pageMatch = /p\.(\d+)/i.exec(token);
      if (!pageMatch) return;
      parts.push(_sourceRefTag(`[${fileName}] ${token}`, fileName, pageMatch[1]));
    });
    return parts.join(" ");
  });

  const bracketPattern = /\[([^\[\]\n]+?\.pdf)\]\s*p\.(\d+)(?:-(\d+))?/gi;
  rendered = rendered.replace(bracketPattern, (match, fileName, pageStart) => {
    return _sourceRefTag(match, fileName, pageStart);
  });

  const parenPattern = /\(([^()\n]+?\.pdf)\s+p\.(\d+)(?:-(\d+))?\)/gi;
  rendered = rendered.replace(parenPattern, (match, fileName, pageStart) => {
    return _sourceRefTag(match, fileName, pageStart);
  });

  return rendered;
}

function bindCitationClicks(container) {
  container.querySelectorAll(".source-ref-inline").forEach((tag) => {
    tag.addEventListener("click", () => {
      const fileName = tag.dataset.file;
      const page = parseInt(tag.dataset.page, 10);
      openAnswerPreviewSource(fileName, page);
    });
  });
}
