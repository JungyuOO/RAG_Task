const reindexBtn = document.getElementById("reindexBtn");
const refreshLibraryBtn = document.getElementById("refreshLibraryBtn");
const uploadProgressArea = document.getElementById("uploadProgressArea");
const libraryScopeBar = document.getElementById("libraryScopeBar");
const libraryToolbar = document.getElementById("libraryToolbar");
const libraryUploadTriggerBtn = document.getElementById("libraryUploadTriggerBtn");
const libraryUploadModal = document.getElementById("libraryUploadModal");
const libraryUploadTitle = document.getElementById("libraryUploadTitle");
const libraryUploadCopy = document.getElementById("libraryUploadCopy");
const libraryUploadVersionField = document.getElementById("libraryUploadVersionField");
const libraryUploadVersionDropdown = document.getElementById("libraryUploadVersionDropdown");
const libraryUploadVersionToggle = document.getElementById("libraryUploadVersionToggle");
const libraryUploadVersionLabel = document.getElementById("libraryUploadVersionLabel");
const libraryUploadVersionMenu = document.getElementById("libraryUploadVersionMenu");
const libraryUploadSelection = document.getElementById("libraryUploadSelection");
const libraryUploadChooseBtn = document.getElementById("libraryUploadChooseBtn");
const libraryUploadSubmitBtn = document.getElementById("libraryUploadSubmitBtn");
const libraryUploadCancelBtn = document.getElementById("libraryUploadCancelBtn");

const DEFAULT_OFFICIAL_VERSIONS = ["4.21", "4.20", "4.19", "4.18", "4.17", "4.16", "4.15"];

let _startupIndexingFile = "";
let _startupPollTimer = null;
let _isReindexSubmitting = false;
let _lastLibraryDocuments = [];
let _libraryScope = "all";
let _officialVersionFilter = "";
let _uploadTarget = { group: "", version: "" };

function _compareVersionsDesc(a, b) {
  const [majorA, minorA] = String(a).split(".").map(Number);
  const [majorB, minorB] = String(b).split(".").map(Number);
  if (majorA !== majorB) return majorB - majorA;
  return minorB - minorA;
}

function _extractVersion(doc) {
  const sourcePath = doc.source_path || doc.file_name || "";
  const match = sourcePath.match(/ocp-(\d+\.\d+)/);
  return match ? match[1] : null;
}

function _isManualDoc(doc) {
  const sourcePath = doc.source_path || "";
  const isGeneratedPdf = sourcePath.includes("/generated_pdf/") || sourcePath.includes("\\generated_pdf\\");
  return isGeneratedPdf || (doc.document_group === "customer_generated" && String(doc.extension || "").toLowerCase() === ".pdf");
}

function _documentVersions(docs) {
  const versions = new Set();
  docs.forEach((doc) => {
    const version = _extractVersion(doc);
    if (version) versions.add(version);
  });
  return Array.from(versions).sort(_compareVersionsDesc);
}

function _availableUploadVersions() {
  const versions = new Set(DEFAULT_OFFICIAL_VERSIONS);
  _lastLibraryDocuments
    .filter((doc) => !_isManualDoc(doc))
    .forEach((doc) => {
      const version = _extractVersion(doc);
      if (version) versions.add(version);
    });
  return Array.from(versions).sort(_compareVersionsDesc);
}

function _groupDocsByVersion(docs) {
  const groups = {};
  const noVersion = [];
  docs.forEach((doc) => {
    const version = _extractVersion(doc);
    if (!version) {
      noVersion.push(doc);
      return;
    }
    if (!groups[version]) groups[version] = [];
    groups[version].push(doc);
  });
  return {
    groups,
    noVersion,
    sortedVersions: Object.keys(groups).sort(_compareVersionsDesc),
  };
}

function _renderScopeBar() {
  if (!libraryScopeBar) return;
  const scopes = [
    { key: "all", label: "전체 보기" },
    { key: "official", label: "OCP 공식 문서" },
    { key: "customer", label: "고객사 메뉴얼" },
  ];

  libraryScopeBar.innerHTML = "";
  scopes.forEach((scope) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "lib-scope-btn" + (_libraryScope === scope.key ? " active" : "");
    button.textContent = scope.label;
    button.addEventListener("click", () => {
      _libraryScope = scope.key;
      if (_libraryScope !== "official") _officialVersionFilter = "";
      closeLibraryUploadModal();
      renderLibrary(_lastLibraryDocuments);
    });
    libraryScopeBar.appendChild(button);
  });
}

function _renderUploadTrigger() {
  if (!libraryUploadTriggerBtn) return;

  if (_libraryScope === "official") {
    libraryUploadTriggerBtn.hidden = false;
    libraryUploadTriggerBtn.textContent = "문서 업로드";
    libraryUploadTriggerBtn.dataset.group = "official_ocp";
    return;
  }

  if (_libraryScope === "customer") {
    libraryUploadTriggerBtn.hidden = false;
    libraryUploadTriggerBtn.textContent = "문서 업로드";
    libraryUploadTriggerBtn.dataset.group = "customer_generated";
    return;
  }

  libraryUploadTriggerBtn.hidden = true;
  libraryUploadTriggerBtn.dataset.group = "";
}

function _closeOfficialVersionMenu() {
  const dropdown = document.getElementById("libraryVersionDropdown");
  const toggle = document.getElementById("libraryVersionToggle");
  if (dropdown) dropdown.classList.remove("open");
  if (toggle) toggle.setAttribute("aria-expanded", "false");
}

function _toggleOfficialVersionMenu() {
  const dropdown = document.getElementById("libraryVersionDropdown");
  const toggle = document.getElementById("libraryVersionToggle");
  if (!dropdown || !toggle) return;
  const isOpen = dropdown.classList.contains("open");
  dropdown.classList.toggle("open", !isOpen);
  toggle.setAttribute("aria-expanded", isOpen ? "false" : "true");
}

function _renderOfficialToolbar(officialDocs) {
  if (!libraryToolbar) return;

  libraryToolbar.innerHTML = "";
  if (_libraryScope !== "official") return;

  const versions = _documentVersions(officialDocs);
  if (_officialVersionFilter && !versions.includes(_officialVersionFilter)) {
    _officialVersionFilter = "";
  }

  const toolbar = document.createElement("div");
  toolbar.className = "library-toolbar-group";
  toolbar.innerHTML =
    '<span class="library-toolbar-label">버전 보기</span>' +
    '<div id="libraryVersionDropdown" class="library-version-dropdown">' +
      '<button id="libraryVersionToggle" class="library-version-toggle" type="button" aria-haspopup="listbox" aria-expanded="false">' +
        '<span id="libraryVersionToggleLabel"></span>' +
        '<span class="library-version-caret" aria-hidden="true">▾</span>' +
      "</button>" +
      '<div id="libraryVersionMenu" class="library-version-menu" role="listbox"></div>' +
    "</div>";
  libraryToolbar.appendChild(toolbar);

  const toggleLabel = document.getElementById("libraryVersionToggleLabel");
  if (toggleLabel) {
    toggleLabel.textContent = _officialVersionFilter ? `OCP ${_officialVersionFilter}` : "전체 보기";
  }

  const menu = document.getElementById("libraryVersionMenu");
  const options = [{ value: "", label: "전체 보기" }].concat(
    versions.map((version) => ({ value: version, label: `OCP ${version}` }))
  );
  options.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "library-version-menu-item" + (_officialVersionFilter === option.value ? " active" : "");
    button.textContent = option.label;
    button.addEventListener("click", () => {
      _officialVersionFilter = option.value;
      _closeOfficialVersionMenu();
      renderLibrary(_lastLibraryDocuments);
    });
    menu.appendChild(button);
  });

  const toggle = document.getElementById("libraryVersionToggle");
  if (toggle) {
    toggle.addEventListener("click", (event) => {
      event.stopPropagation();
      _toggleOfficialVersionMenu();
    });
  }
}

function _setUploadVersion(version) {
  _uploadTarget.version = version;
  if (libraryUploadVersionLabel) {
    libraryUploadVersionLabel.textContent = version || "버전 선택";
  }
}

function _closeUploadVersionMenu() {
  if (!libraryUploadVersionDropdown || !libraryUploadVersionToggle) return;
  libraryUploadVersionDropdown.classList.remove("open");
  libraryUploadVersionToggle.setAttribute("aria-expanded", "false");
}

function _toggleUploadVersionMenu() {
  if (!libraryUploadVersionDropdown || !libraryUploadVersionToggle) return;
  const isOpen = libraryUploadVersionDropdown.classList.contains("open");
  libraryUploadVersionDropdown.classList.toggle("open", !isOpen);
  libraryUploadVersionToggle.setAttribute("aria-expanded", isOpen ? "false" : "true");
}

function _renderUploadVersionMenu() {
  if (!libraryUploadVersionMenu) return;
  libraryUploadVersionMenu.innerHTML = "";
  _availableUploadVersions().forEach((version) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "library-version-menu-item" + (_uploadTarget.version === version ? " active" : "");
    button.textContent = `OCP ${version}`;
    button.addEventListener("click", () => {
      _setUploadVersion(version);
      _closeUploadVersionMenu();
      _renderUploadVersionMenu();
    });
    libraryUploadVersionMenu.appendChild(button);
  });
}

function syncSelectedUploadFiles() {
  if (!libraryUploadSelection || !fileInput) return;

  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    libraryUploadSelection.textContent = "선택된 파일이 없습니다.";
    return;
  }

  const preview = files.slice(0, 2).map((file) => file.name).join(", ");
  const suffix = files.length > 2 ? ` 외 ${files.length - 2}개` : "";
  libraryUploadSelection.textContent = `선택된 파일: ${preview}${suffix}`;
}

function openLibraryUploadModal(group) {
  if (!libraryUploadModal) return;

  const isOfficial = group === "official_ocp";
  const initialVersion = isOfficial
    ? (_officialVersionFilter || _availableUploadVersions()[0] || "")
    : "";
  _uploadTarget = { group, version: initialVersion };

  if (libraryUploadTitle) {
    libraryUploadTitle.textContent = isOfficial ? "OCP 공식 문서 업로드" : "고객사 메뉴얼 업로드";
  }
  if (libraryUploadCopy) {
    libraryUploadCopy.textContent = isOfficial
      ? "업로드할 OCP 버전을 선택한 뒤 파일을 등록합니다."
      : "고객사 메뉴얼 문서를 단일 문서군으로 업로드합니다.";
  }
  if (libraryUploadVersionField) {
    libraryUploadVersionField.classList.toggle("hidden", !isOfficial);
  }

  _setUploadVersion(initialVersion);
  _renderUploadVersionMenu();
  if (fileInput) fileInput.value = "";
  syncSelectedUploadFiles();

  libraryUploadModal.classList.add("open");
  libraryUploadModal.setAttribute("aria-hidden", "false");
}

function closeLibraryUploadModal() {
  if (!libraryUploadModal) return;
  libraryUploadModal.classList.remove("open");
  libraryUploadModal.setAttribute("aria-hidden", "true");
  _closeUploadVersionMenu();
  if (fileInput) fileInput.value = "";
  _uploadTarget = { group: "", version: "" };
  syncSelectedUploadFiles();
}

function updateLibraryStats() {}

let _chunkLoadTimer = null;
let _chunkLoadProgress = 0;
let _activeChunkRequestId = 0;
let _activeChunkAbortController = null;

function _getChunkViewerRefs() {
  const overlay = document.getElementById("chunk-viewer");
  if (!overlay) return { overlay: null, container: null };
  const container = overlay.querySelector("div") || overlay;
  return { overlay, container };
}

function _stopChunkLoadTimer() {
  if (!_chunkLoadTimer) return;
  clearInterval(_chunkLoadTimer);
  _chunkLoadTimer = null;
}

function closeChunkViewer() {
  const { overlay } = _getChunkViewerRefs();
  if (!overlay) return;
  _stopChunkLoadTimer();
  _activeChunkRequestId += 1;
  if (_activeChunkAbortController) {
    _activeChunkAbortController.abort();
    _activeChunkAbortController = null;
  }
  overlay.style.display = "none";
}

function _renderChunkLoadingModal(fileName, progress, message) {
  const { overlay, container } = _getChunkViewerRefs();
  if (!overlay || !container) return;
  container.innerHTML =
    '<div class="chunk-viewer-header">' +
      '<h3>' + escapeHtml(fileName) + ' chunk loading</h3>' +
      '<button type="button" onclick="closeChunkViewer()">Close</button>' +
    '</div>' +
    '<div class="chunk-loading-shell">' +
      '<div class="chunk-loading-copy">' + escapeHtml(message) + '</div>' +
      '<div class="chunk-loading-progress"><div class="chunk-loading-progress-bar" style="width:' + progress + '%"></div></div>' +
      '<div class="chunk-loading-meta">' + progress + '%</div>' +
    '</div>';
  overlay.style.display = "block";
  overlay.onclick = function (event) {
    if (event.target === overlay) closeChunkViewer();
  };
}

function _startChunkLoading(fileName) {
  _chunkLoadProgress = 0;
  _renderChunkLoadingModal(fileName, 0, "Loading chunk list...");
  _stopChunkLoadTimer();
  _chunkLoadTimer = setInterval(() => {
    const increment =
      _chunkLoadProgress < 20 ? 6 :
      _chunkLoadProgress < 40 ? 4 :
      _chunkLoadProgress < 60 ? 2 :
      _chunkLoadProgress < 75 ? 1 :
      0.4;
    _chunkLoadProgress = Math.min(90, Math.round((_chunkLoadProgress + increment) * 10) / 10);
    _renderChunkLoadingModal(fileName, _chunkLoadProgress, "Reading chunks and metadata...");
  }, 220);
}

function _renderChunkListModal(data) {
  const { overlay, container } = _getChunkViewerRefs();
  if (!overlay || !container) return;
  const totalPages = Math.max(1, Math.ceil(data.total / data.page_size));
  container.innerHTML =
    '<div class="chunk-viewer-header">' +
      '<h3>' + escapeHtml(data.file_name) + ' chunk list (' + data.total + ')</h3>' +
      '<button type="button" onclick="closeChunkViewer()">Close</button>' +
    '</div>' +
    '<div class="chunk-list">' +
      data.chunks.map((chunk) =>
        '<div class="chunk-item" data-chunk-id="' + chunk.chunk_id + '">' +
          '<div class="chunk-header">' +
            '<span class="chunk-id">#' + chunk.chunk_id.substring(0, 12) + '...</span>' +
            '<span class="chunk-page">p.' + (chunk.page_number ?? "-") + '</span>' +
            '<span class="chunk-tokens">' + chunk.token_count + ' tokens</span>' +
          '</div>' +
          '<div class="chunk-text">' + escapeHtml(chunk.text) + '</div>' +
        '</div>'
      ).join("") +
    '</div>' +
    '<div class="chunk-pagination">' +
      (data.page > 1 ? '<button type="button" class="chunk-prev-button" data-page="' + (data.page - 1) + '">Prev</button>' : '') +
      '<span>' + data.page + ' / ' + totalPages + '</span>' +
      (data.page < totalPages ? '<button type="button" class="chunk-next-button" data-page="' + (data.page + 1) + '">Next</button>' : '') +
    '</div>';
  overlay.style.display = "block";
  const prevButton = container.querySelector(".chunk-prev-button");
  const nextButton = container.querySelector(".chunk-next-button");
  if (prevButton) {
    prevButton.addEventListener("click", () => openChunkViewer(data.file_name, Number(prevButton.dataset.page || "1")));
  }
  if (nextButton) {
    nextButton.addEventListener("click", () => openChunkViewer(data.file_name, Number(nextButton.dataset.page || "1")));
  }
}

async function openChunkViewer(fileName, page = 1) {
  _activeChunkRequestId += 1;
  const requestId = _activeChunkRequestId;
  if (_activeChunkAbortController) {
    _activeChunkAbortController.abort();
  }
  _activeChunkAbortController = new AbortController();
  _startChunkLoading(fileName);
  await new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(resolve);
    });
  });
  try {
    const response = await fetch(
      "/api/library/" + encodeURIComponent(fileName) + "/chunks?page=" + page + "&page_size=20",
      { signal: _activeChunkAbortController.signal }
    );
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    if (requestId !== _activeChunkRequestId) return;
    _stopChunkLoadTimer();
    _renderChunkLoadingModal(fileName, 100, "Chunk list ready.");
    setTimeout(() => {
      if (requestId !== _activeChunkRequestId) return;
      _renderChunkListModal(data);
    }, 180);
  } catch (error) {
    if (requestId !== _activeChunkRequestId) return;
    if (error && error.name === "AbortError") return;
    _stopChunkLoadTimer();
    _renderChunkLoadingModal(fileName, 100, "Failed to load chunks.");
    console.error("Chunk load failed:", error);
  } finally {
    if (requestId === _activeChunkRequestId) {
      _activeChunkAbortController = null;
    }
  }
}

window.openChunkViewer = openChunkViewer;
window.closeChunkViewer = closeChunkViewer;

function _makeDocRow(doc) {
  const tr = document.createElement("tr");
  const loaders = doc.loaders && doc.loaders.length ? doc.loaders.join(", ") : "미확인";
  const isIndexed = Number(doc.indexed_chunks) > 0;
  const isCurrentlyIndexing = !isIndexed && _startupIndexingFile === doc.file_name;

  let statusBadge = '<span class="status-badge not-indexed">대기 중</span>';
  if (isIndexed) {
    statusBadge = '<span class="status-badge">인덱싱 완료</span>';
  } else if (isCurrentlyIndexing) {
    statusBadge = '<span class="status-badge indexing">인덱싱 중</span>';
  }

  tr.innerHTML =
    '<td><div class="item-title">' + escapeHtml(doc.file_name) + "</div>" +
    '<div class="item-copy">문서군: ' + escapeHtml(doc.document_group || "") + " · 로더: " + escapeHtml(loaders) + "</div></td>" +
    "<td>" + statusBadge + "</td>" +
    "<td>청크 " + doc.indexed_chunks + "<br />페이지 " + doc.indexed_pages + "</td>" +
    "<td>" + escapeHtml(String(doc.extension || "").toUpperCase()) + "</td>" +
    '<td><div class="row-actions">' +
    '<button class="secondary mini-button preview-button" type="button">미리보기</button>' +
    '<button class="secondary mini-button chunks-button" type="button">청크 보기</button>' +
    '<button class="secondary mini-button delete-button" type="button">삭제</button>' +
    "</div></td>";

  tr.querySelector(".preview-button").addEventListener("click", () => openPdf(doc.file_name));
  tr.querySelector(".chunks-button").addEventListener("click", () => openChunkViewer(doc.file_name));
  tr.querySelector(".delete-button").addEventListener("click", () => deleteLibraryFile(doc.file_name));
  return tr;
}

function _makeDocTable(docs) {
  const wrap = document.createElement("div");
  wrap.className = "library-table-wrap";

  const table = document.createElement("table");
  table.className = "library-table";
  table.innerHTML = "<thead><tr><th>파일</th><th>상태</th><th>인덱싱</th><th>형식</th><th>액션</th></tr></thead><tbody></tbody>";

  const tbody = table.querySelector("tbody");
  docs.forEach((doc) => tbody.appendChild(_makeDocRow(doc)));
  wrap.appendChild(table);
  return wrap;
}

function _makeSection(label, docs) {
  const section = document.createElement("section");
  section.className = "library-doc-section";

  const header = document.createElement("div");
  header.className = "library-section-head";
  header.innerHTML =
    '<div style="display:flex;align-items:center;gap:10px;">' +
      '<span style="font-size:13px;font-weight:800;color:#182538;">' + escapeHtml(label) + "</span>" +
      '<span style="font-size:12px;color:#66758a;">' + docs.length + "개 문서</span>" +
    "</div>";

  section.appendChild(header);
  section.appendChild(_makeDocTable(docs));
  libraryList.appendChild(section);
}

function _renderOfficialSections(officialDocs) {
  if (_officialVersionFilter) {
    const filtered = officialDocs.filter((doc) => _extractVersion(doc) === _officialVersionFilter);
    if (!filtered.length) {
      libraryList.innerHTML = '<div class="empty">선택한 버전 문서가 없습니다.</div>';
      return;
    }
    _makeSection(`OCP ${_officialVersionFilter}`, filtered);
    return;
  }

  const { groups, noVersion, sortedVersions } = _groupDocsByVersion(officialDocs);
  if (!sortedVersions.length && !noVersion.length) {
    libraryList.innerHTML = '<div class="empty">등록된 공식 문서가 없습니다.</div>';
    return;
  }

  sortedVersions.forEach((version) => _makeSection(`OCP ${version}`, groups[version]));
  if (noVersion.length) _makeSection("기타", noVersion);
}

function renderLibrary(documents) {
  _lastLibraryDocuments = documents;
  _renderScopeBar();
  _renderUploadTrigger();

  libraryList.innerHTML = "";
  if (libraryToolbar) libraryToolbar.innerHTML = "";

  const officialDocs = documents.filter((doc) => !_isManualDoc(doc));
  const customerDocs = documents.filter((doc) => _isManualDoc(doc));

  if (_libraryScope === "all") {
    if (officialDocs.length) _renderOfficialSections(officialDocs);
    if (customerDocs.length) _makeSection("고객사 메뉴얼", customerDocs);
    if (!officialDocs.length && !customerDocs.length) {
      libraryList.innerHTML = '<div class="empty">등록된 문서가 없습니다.</div>';
    }
    return;
  }

  if (_libraryScope === "official") {
    _renderOfficialToolbar(officialDocs);
    _renderOfficialSections(officialDocs);
    return;
  }

  if (customerDocs.length) {
    _makeSection("고객사 메뉴얼", customerDocs);
  } else {
    libraryList.innerHTML = '<div class="empty">등록된 고객사 메뉴얼이 없습니다.</div>';
  }
}

function _renderUploadState(completedDocs, currentFile, pct, completedCount, totalFiles) {
  renderLibrary(completedDocs);
  if (!currentFile) {
    uploadProgressArea.innerHTML = "";
    return;
  }

  uploadProgressArea.innerHTML =
    '<div id="upload-progress-item" style="padding:14px 16px;border:1px solid #bfd3fb;border-radius:12px;background:#f0f6ff;margin-top:10px;">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">' +
        '<span style="font-size:13px;font-weight:700;color:#182538;">' + escapeHtml(currentFile) + "</span>" +
        '<span id="upload-pct-label" style="font-size:12px;color:#2a57df;font-weight:700;">' + pct + "%</span>" +
      "</div>" +
      '<div style="background:#dde4ef;border-radius:999px;height:6px;overflow:hidden;">' +
        '<div id="upload-pct-bar" style="height:100%;background:#2a57df;border-radius:999px;transition:width 0.15s ease;width:' + pct + '%;"></div>' +
      "</div>" +
      '<div style="font-size:11px;color:#66758a;margin-top:6px;">처리 중 (' + (completedCount + 1) + "/" + totalFiles + ")</div>" +
    "</div>";
}

function _updateProgressBar(pct) {
  const bar = document.getElementById("upload-pct-bar");
  const label = document.getElementById("upload-pct-label");
  if (bar) bar.style.width = pct + "%";
  if (label) label.textContent = pct + "%";
}

function _setLibraryBusyState(isBusy) {
  if (reindexBtn) reindexBtn.disabled = isBusy;
  if (refreshLibraryBtn) refreshLibraryBtn.disabled = false;
  if (libraryUploadTriggerBtn) libraryUploadTriggerBtn.disabled = isBusy;
  if (libraryUploadChooseBtn) libraryUploadChooseBtn.disabled = isBusy;
  if (libraryUploadSubmitBtn) libraryUploadSubmitBtn.disabled = isBusy;
  if (libraryUploadCancelBtn) libraryUploadCancelBtn.disabled = isBusy;
  if (libraryUploadVersionToggle) libraryUploadVersionToggle.disabled = isBusy;
  if (fileInput) fileInput.disabled = isBusy;
  if (uploadDropzone) uploadDropzone.style.pointerEvents = isBusy ? "none" : "auto";
  if (uploadDropzone) uploadDropzone.style.opacity = isBusy ? "0.65" : "1";
}

async function loadLibrary() {
  setLibraryStatus("자료실 상태를 불러오는 중입니다.", "loading", "Loading");
  try {
    const response = await fetch("/api/library");
    if (!response.ok) throw new Error(await extractErrorMessage(response));

    const data = await response.json();
    const startupState = data.startup_indexing || {};
    const reindexState = data.reindexing || {};
    _startupIndexingFile = startupState.current_file || reindexState.current_file || "";

    updateLibraryStats(data);
    renderLibrary(data.indexed_documents || []);

    const startupBusy = startupState.status === "indexing";
    const reindexBusy = reindexState.status === "indexing";
    _setLibraryBusyState(Boolean(startupBusy || reindexBusy || _isReindexSubmitting));

    if (startupBusy || reindexBusy) {
      if (!_startupPollTimer) _startupPollTimer = setInterval(loadLibrary, 2000);
      return;
    }

    if (_startupPollTimer) {
      clearInterval(_startupPollTimer);
      _startupPollTimer = null;
    }
    if (!_isReindexSubmitting) {
      uploadProgressArea.innerHTML = "";
    }
    setLibraryStatus("문서 " + data.total_files + "개를 확인했습니다.", "success", "Ready");
  } catch (error) {
    updateLibraryStats({ total_files: 0, indexed_documents: [] });
    renderLibrary([]);
    _setLibraryBusyState(false);
    setLibraryStatus("자료를 불러오지 못했습니다. " + error.message, "error", "Error");
  }
}

async function deleteLibraryFile(fileName) {
  if (!window.confirm("'" + fileName + "' 문서를 삭제할까요?\n추출된 마크다운도 함께 삭제됩니다.")) return;

  setLibraryStatus(fileName + " 삭제 중입니다.", "loading", "Deleting");
  try {
    const response = await fetch("/api/library?file_name=" + encodeURIComponent(fileName), { method: "DELETE" });
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();
    const markdownText = data.deleted_markdown ? "마크다운 삭제됨" : "마크다운 없음";
    setLibraryStatus(data.deleted_file + " 삭제 완료 · " + markdownText, "success", "Deleted");
    await loadLibrary();
  } catch (error) {
    setLibraryStatus("문서 삭제에 실패했습니다. " + error.message, "error", "Delete Failed");
  }
}

async function uploadFiles() {
  const files = fileInput.files;
  if (!files.length) {
    setLibraryStatus("업로드할 파일을 먼저 선택해 주세요.", "error", "Upload Failed");
    return;
  }
  if (!_uploadTarget.group) {
    setLibraryStatus("업로드 대상 카테고리를 먼저 선택해 주세요.", "error", "Upload Failed");
    return;
  }

  const isOfficial = _uploadTarget.group === "official_ocp";
  const versionValue = isOfficial ? (_uploadTarget.version || "") : "";
  if (isOfficial && !/^4\.\d+$/.test(versionValue)) {
    setLibraryStatus("버전을 먼저 선택해 주세요.", "error", "Upload Failed");
    return;
  }

  const totalFiles = files.length;
  const fileNames = Array.from(files).map((file) => file.name);
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));

  setLibraryStatus("문서 업로드 및 인덱싱을 진행 중입니다.", "loading", "Uploading");
  let completedDocs = _lastLibraryDocuments.filter((doc) => Number(doc.indexed_chunks || 0) > 0);
  let completedCount = 0;
  _renderUploadState(completedDocs, fileNames[0], 0, 0, totalFiles);

  try {
    const query = new URLSearchParams({
      target_group: _uploadTarget.group,
      target_version: versionValue,
    });
    const response = await fetch("/api/library/upload?" + query.toString(), { method: "POST", body: formData });
    if (!response.ok) throw new Error(await extractErrorMessage(response));

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let totalChunks = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        const data = JSON.parse(part.slice(6));

        if (data.type === "progress") {
          _updateProgressBar(data.pct);
          continue;
        }

        if (data.type === "file_indexed") {
          completedCount += 1;
          totalChunks += data.indexed_chunks || 0;
          const docInfo = ((data.library && data.library.indexed_documents) || []).find((doc) => doc.file_name === data.file);
          if (docInfo) completedDocs.push(docInfo);
          if (data.library) updateLibraryStats(data.library);
          const nextFile = fileNames[completedCount] || null;
          _renderUploadState(completedDocs, nextFile, 0, completedCount, totalFiles);
          continue;
        }

        if (data.type === "file_error") {
          completedCount += 1;
          const nextFile = fileNames[completedCount] || null;
          _renderUploadState(completedDocs, nextFile, 0, completedCount, totalFiles);
          setLibraryStatus(data.file + " 오류: " + data.error, "error", "Upload Error");
          continue;
        }

        if (data.type === "done") {
          setLibraryStatus("업로드 및 인덱싱 완료 · 청크 " + totalChunks + "개", "success", "Completed");
        }
      }
    }

    closeLibraryUploadModal();
    await loadLibrary();
  } catch (error) {
    setLibraryStatus("업로드에 실패했습니다. " + error.message, "error", "Upload Failed");
    await loadLibrary();
  }
}

function handleSelectedFiles(files) {
  const dataTransfer = new DataTransfer();
  Array.from(files).forEach((file) => dataTransfer.items.add(file));
  fileInput.files = dataTransfer.files;
  syncSelectedUploadFiles();
}

async function reindexAll() {
  _isReindexSubmitting = true;
  _setLibraryBusyState(true);
  setLibraryStatus("전체 자료실을 다시 인덱싱하는 중입니다.", "loading", "Reindexing");
  if (!_startupPollTimer) _startupPollTimer = setInterval(loadLibrary, 2000);

  try {
    const response = await fetch("/api/reindex", { method: "POST" });
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();
    setLibraryStatus("재인덱싱 완료: 파일 " + data.indexed_files + "개 · 청크 " + data.indexed_chunks + "개", "success", "Completed");
  } catch (error) {
    setLibraryStatus("재인덱싱에 실패했습니다. " + error.message, "error", "Reindex Failed");
  } finally {
    _isReindexSubmitting = false;
    await loadLibrary();
  }
}

if (libraryUploadTriggerBtn) {
  libraryUploadTriggerBtn.addEventListener("click", () => {
    const group = libraryUploadTriggerBtn.dataset.group || "";
    if (group) openLibraryUploadModal(group);
  });
}

if (libraryUploadChooseBtn) {
  libraryUploadChooseBtn.addEventListener("click", () => fileInput.click());
}

if (libraryUploadSubmitBtn) {
  libraryUploadSubmitBtn.addEventListener("click", uploadFiles);
}

if (libraryUploadCancelBtn) {
  libraryUploadCancelBtn.addEventListener("click", closeLibraryUploadModal);
}

if (libraryUploadVersionToggle) {
  libraryUploadVersionToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    _toggleUploadVersionMenu();
  });
}

if (libraryUploadModal) {
  libraryUploadModal.addEventListener("click", (event) => {
    if (event.target === libraryUploadModal) closeLibraryUploadModal();
  });
}

document.addEventListener("click", (event) => {
  const officialDropdown = document.getElementById("libraryVersionDropdown");
  if (officialDropdown && !officialDropdown.contains(event.target)) {
    _closeOfficialVersionMenu();
  }
  if (libraryUploadVersionDropdown && !libraryUploadVersionDropdown.contains(event.target)) {
    _closeUploadVersionMenu();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    _closeOfficialVersionMenu();
    _closeUploadVersionMenu();
    closeLibraryUploadModal();
  }
});
