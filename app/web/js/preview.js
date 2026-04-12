function syncPreviewPanel() {
  chatLayout.classList.toggle("preview-open", previewAvailable && previewOpen);
  previewPanel.setAttribute("aria-hidden", previewAvailable && previewOpen ? "false" : "true");
}

function hidePreviewPanel() {
  previewAvailable = false;
  previewOpen = false;
  syncPreviewPanel();
}

function showPreviewPanel() {
  previewAvailable = true;
  previewOpen = true;
  syncPreviewPanel();
}

function resetPreview() {
  answerPreviewSummary.textContent = "질문을 보내면 관련 문서 위치를 먼저 보여줍니다.";
  answerPreviewMeta.innerHTML =
    '<div class="answer-preview-title">문서 미리보기 대기 중</div>' +
    '<div class="answer-preview-copy">근거가 잡히면 해당 문서 위치를 표시합니다.</div>';
  answerPageStrip.innerHTML = "";
  answerPreviewFrame.src = "about:blank";
  window.currentPreviewFileName = "";
  hidePreviewPanel();
}

function openPdf(fileName, pageNumber, sourcePath = "") {
  currentPdfFileName = fileName;
  pdfModalTitle.textContent = fileName || "PDF 미리보기";
  const pageFragment = Number(pageNumber) > 0 ? ("&page=" + Number(pageNumber)) : "";
  const params = new URLSearchParams({ file_name: fileName });
  if (sourcePath) params.set("source_path", sourcePath);
  pdfViewer.src = "/api/library/preview?" + params.toString() + "#toolbar=1&navpanes=0&scrollbar=1" + pageFragment;
  pdfModal.classList.add("open");
  pdfModal.setAttribute("aria-hidden", "false");
}

function closePdf() {
  pdfModal.classList.remove("open");
  pdfModal.setAttribute("aria-hidden", "true");
  pdfViewer.src = "";
  currentPdfFileName = "";
}

function downloadPdf() {
  if (!currentPdfFileName) return;
  window.open("/api/library/download?file_name=" + encodeURIComponent(currentPdfFileName), "_blank");
}

function buildPreviewUrl(fileName, pageNumber, anchor, sourcePath = "") {
  const resolvedAnchor = anchor || ("page-" + pageNumber);
  const params = new URLSearchParams({ file_name: fileName, ts: String(Date.now()) });
  if (sourcePath) params.set("source_path", sourcePath);
  return "/api/library/preview-html?" + params.toString() + "#" + encodeURIComponent(resolvedAnchor);
}

function openLibrarySource(fileName, pageNumber, anchor, sourcePath = "") {
  if (String(fileName || "").toLowerCase().endsWith(".md")) {
    window.open(buildPreviewUrl(fileName, pageNumber || 1, anchor || "", sourcePath), "_blank");
    return;
  }
  openPdf(fileName, pageNumber, sourcePath);
}

function loadAnswerPreviewPage(fileName, pageNumber, anchor, sourcePath = "") {
  answerPreviewFrame.src = "about:blank";
  window.setTimeout(() => {
    answerPreviewFrame.src = buildPreviewUrl(fileName, pageNumber, anchor, sourcePath);
  }, 0);
}

function openAnswerPreviewSource(fileName, pageNumber, blockAnchor, sourcePath = "") {
  renderAnswerPreview({
    mode: "rag",
    preview_pages: [
      {
        source_path: sourcePath || fileName,
        page_number: pageNumber,
        block_anchor: blockAnchor || "",
        html_anchor: "page-" + pageNumber,
      },
    ],
  });
}

function renderAnswerPreview(payload) {
  currentContextPayload = payload || null;
  answerPageStrip.innerHTML = "";
  answerPreviewSummary.textContent = "질문 기준 관련 문서 미리보기";
  const previewPages = Array.isArray(payload.preview_pages) ? payload.preview_pages : [];

  if (payload.mode !== "rag" || !previewPages.length) {
    answerPreviewMeta.innerHTML =
      '<div class="answer-preview-title">표시할 근거 위치가 없습니다</div>' +
      '<div class="answer-preview-copy">이번 질문은 미리보기를 표시할 만큼 강한 근거가 잡히지 않았습니다.</div>';
    answerPreviewFrame.src = "about:blank";
    window.currentPreviewFileName = "";
    hidePreviewPanel();
    return;
  }

  const pages = previewPages.map((page) => {
    const sourcePath = String(page.source_path || "");
    const fileName = sourcePath.split(/[\\/]/).pop() || sourcePath;
    return {
      sourcePath,
      fileName,
      pageNumber: page.page_number || 1,
      blockAnchor: String(page.block_anchor || ""),
      htmlAnchor: String(page.html_anchor || ("page-" + (page.page_number || 1))),
    };
  });

  const primary = pages[0];
  window.currentPreviewFileName = primary.fileName;
  const uniqueSources = [];
  const seenSources = new Set();
  pages.forEach((page) => {
    const key = [page.fileName, page.pageNumber, page.sourcePath].join("|");
    if (seenSources.has(key)) return;
    seenSources.add(key);
    uniqueSources.push(page);
  });

  const statsMarkup =
    '<div class="answer-preview-stats">' +
      `<span class="answer-preview-stat"><strong>${uniqueSources.length}</strong> source</span>` +
      `<span class="answer-preview-stat"><strong>${pages.length}</strong> preview page</span>` +
    '</div>';

  const sourceCardsMarkup =
    '<div class="answer-source-card-list">' +
    uniqueSources.slice(0, 4).map((page) =>
      '<button type="button" class="answer-source-card" ' +
      `data-file-name="${escapeHtml(page.fileName)}" ` +
      `data-page-number="${page.pageNumber}" ` +
      `data-source-path="${escapeHtml(page.sourcePath)}" ` +
      `data-block-anchor="${escapeHtml(page.blockAnchor || page.htmlAnchor || "")}">` +
        `<span class="answer-source-card-file">${escapeHtml(page.fileName)}</span>` +
        `<span class="answer-source-card-page">p.${page.pageNumber}</span>` +
      '</button>'
    ).join("") +
    '</div>';

  answerPreviewMeta.innerHTML =
    '<div class="answer-preview-title">' + escapeHtml(primary.fileName) + ' · p.' + primary.pageNumber + '</div>' +
    '<div class="answer-preview-copy">근거 위치를 먼저 보여주고, 필요하면 원본 문서를 바로 열 수 있습니다.</div>' +
    '<div class="answer-preview-actions">' +
      '<button type="button" class="secondary mini-button answer-preview-pdf-button">원본 PDF</button>' +
    '</div>' +
    statsMarkup +
    sourceCardsMarkup;

  const pdfButton = answerPreviewMeta.querySelector(".answer-preview-pdf-button");
  if (pdfButton) {
    pdfButton.textContent = String(primary.fileName || "").toLowerCase().endsWith(".md") ? "원본 문서" : "원본 PDF";
    pdfButton.addEventListener("click", () => openLibrarySource(primary.fileName, primary.pageNumber, primary.blockAnchor || primary.htmlAnchor, primary.sourcePath));
  }

  answerPreviewMeta.querySelectorAll(".answer-source-card").forEach((card) => {
    card.addEventListener("click", () => {
      const fileName = card.dataset.fileName || primary.fileName;
      const pageNumber = parseInt(card.dataset.pageNumber, 10) || 1;
      const blockAnchor = card.dataset.blockAnchor || "";
      const sourcePath = card.dataset.sourcePath || fileName;
      loadAnswerPreviewPage(fileName, pageNumber, blockAnchor, sourcePath);
      answerPageStrip.querySelectorAll(".answer-page-chip").forEach((chip) => chip.classList.remove("active"));
    });
  });

  pages.forEach((page, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "answer-page-chip" + (index === 0 ? " active" : "");
    button.innerHTML =
      '<span class="answer-page-chip-label">' + escapeHtml(page.fileName) + '</span>' +
      '<span class="answer-page-chip-page">p.' + page.pageNumber + '</span>';
    button.addEventListener("click", () => {
      loadAnswerPreviewPage(page.fileName, page.pageNumber, page.blockAnchor || page.htmlAnchor, page.sourcePath);
      answerPageStrip.querySelectorAll(".answer-page-chip").forEach((chip) => chip.classList.remove("active"));
      button.classList.add("active");
    });
    answerPageStrip.appendChild(button);
  });

  showPreviewPanel();
  loadAnswerPreviewPage(primary.fileName, primary.pageNumber, primary.blockAnchor || primary.htmlAnchor, primary.sourcePath);
}
