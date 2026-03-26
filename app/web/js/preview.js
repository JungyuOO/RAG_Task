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
  answerPreviewSummary.textContent = "질문을 보내면 가장 관련된 PDF 페이지를 바로 미리보기로 보여줍니다.";
  answerPreviewMeta.innerHTML = '<div class="answer-preview-title">문서 미리보기 대기 중</div><div class="answer-preview-copy">RAG 근거가 잡히면 해당 PDF 페이지를 표시합니다.</div>';
  answerPageStrip.innerHTML = "";
  answerPreviewFrame.src = "about:blank";
  hidePreviewPanel();
}

function openPdf(fileName) {
  currentPdfFileName = fileName;
  pdfModalTitle.textContent = fileName || "PDF 미리보기";
  pdfViewer.src = "/api/library/preview?file_name=" + encodeURIComponent(fileName) + "#toolbar=1&navpanes=0&scrollbar=1";
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

function buildPreviewUrl(fileName, pageNumber) {
  return "/api/library/preview?file_name=" + encodeURIComponent(fileName)
    + "&preview_page=" + pageNumber
    + "&ts=" + Date.now()
    + "#page=" + pageNumber
    + "&zoom=page-fit&toolbar=1&navpanes=0&scrollbar=1";
}

function loadAnswerPreviewPage(fileName, pageNumber) {
  answerPreviewFrame.src = "about:blank";
  window.setTimeout(() => {
    answerPreviewFrame.src = buildPreviewUrl(fileName, pageNumber);
  }, 0);
}

function renderAnswerPreview(payload) {
  currentContextPayload = payload || null;
  answerPageStrip.innerHTML = "";
  answerPreviewSummary.textContent = "질문 기준 관련 문서 미리보기";
  const previewPages = Array.isArray(payload.preview_pages) ? payload.preview_pages : [];

  if (payload.mode !== "rag" || !previewPages.length) {
    answerPreviewMeta.innerHTML = '<div class="answer-preview-title">표시할 근거 페이지가 없습니다</div><div class="answer-preview-copy">이번 질문은 업로드된 문서 근거가 약해서 우측 미리보기를 표시하지 않습니다.</div>';
    answerPreviewFrame.src = "about:blank";
    hidePreviewPanel();
    return;
  }

  const pages = previewPages.map((page) => {
    const sourcePath = String(page.source_path || "");
    const fileName = sourcePath.split(/[\\/]/).pop() || sourcePath;
    return { fileName, pageNumber: page.page_number || 1 };
  });

  const primary = pages[0];
  answerPreviewMeta.innerHTML = '<div class="answer-preview-title">' + escapeHtml(primary.fileName) + ' · p.' + primary.pageNumber + '</div><div class="answer-preview-copy">답변에 사용된 해당 페이지가 먼저 열리도록 표시합니다.</div>';

  if (pages.length > 1) {
    pages.forEach((page, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "answer-page-chip" + (index === 0 ? " active" : "");
      button.innerHTML = '<span class="answer-page-chip-label">' + escapeHtml(page.fileName) + '</span><span class="answer-page-chip-page">p.' + page.pageNumber + '</span>';
      button.addEventListener("click", () => {
        loadAnswerPreviewPage(page.fileName, page.pageNumber);
        answerPageStrip.querySelectorAll(".answer-page-chip").forEach((chip) => chip.classList.remove("active"));
        button.classList.add("active");
      });
      answerPageStrip.appendChild(button);
    });
  }

  // 패널을 먼저 열어서 레이아웃(너비)이 확정된 후 iframe을 로드한다.
  // 순서가 반대면 iframe이 너비 0 상태에서 PDF를 렌더링하여 극단적으로 축소된다.
  showPreviewPanel();
  loadAnswerPreviewPage(primary.fileName, primary.pageNumber);
}
