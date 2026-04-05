const reindexBtn = document.getElementById("reindexBtn");
const refreshLibraryBtn = document.getElementById("refreshLibraryBtn");
const uploadProgressArea = document.getElementById("uploadProgressArea");

let _startupIndexingFile = "";
let _startupPollTimer = null;
let _isReindexSubmitting = false;

function updateLibraryStats(data) {
  const documents = data.indexed_documents || [];
  const totalFiles = Number(data.total_files || documents.length || 0);
  const indexedFiles = documents.filter((doc) => Number(doc.indexed_chunks || 0) > 0).length;
  const totalChunks = documents.reduce((sum, doc) => sum + Number(doc.indexed_chunks || 0), 0);
  if (totalFilesStat) totalFilesStat.textContent = totalFiles.toLocaleString("ko-KR");
  if (indexedFilesStat) indexedFilesStat.textContent = indexedFiles.toLocaleString("ko-KR");
  if (indexedChunksStat) indexedChunksStat.textContent = totalChunks.toLocaleString("ko-KR");
}

function _makeDocRow(doc) {
  const tr = document.createElement("tr");
  const loaders = doc.loaders && doc.loaders.length ? doc.loaders.join(", ") : "미인식";
  const isIndexed = Number(doc.indexed_chunks) > 0;
  const isCurrentlyIndexing = !isIndexed && _startupIndexingFile === doc.file_name;
  let statusBadge;
  if (isIndexed) {
    statusBadge = '<span class="status-badge">인덱싱 완료</span>';
  } else if (isCurrentlyIndexing) {
    statusBadge = '<span class="status-badge indexing">인덱싱 중</span>';
  } else {
    statusBadge = '<span class="status-badge not-indexed">대기 중</span>';
  }
  tr.innerHTML =
    '<td><div class="item-title">' + escapeHtml(doc.file_name) + "</div>" +
    '<div class="item-copy">로드 방식: ' + escapeHtml(loaders) + "</div></td>" +
    "<td>" + statusBadge + "</td>" +
    "<td>청크 " + doc.indexed_chunks + "<br />페이지 " + doc.indexed_pages + "</td>" +
    "<td>" + escapeHtml(String(doc.extension || "").toUpperCase()) + "</td>" +
    '<td><div class="row-actions">' +
    '<button class="secondary mini-button preview-button" type="button">미리보기</button>' +
    '<button class="secondary mini-button chunks-button" type="button">청크 보기</button>' +
    '<button class="secondary mini-button delete-button" type="button">삭제</button>' +
    "</div></td>";
  tr.querySelector(".preview-button").addEventListener("click", () => openPdf(doc.file_name));
  tr.querySelector(".chunks-button").addEventListener("click", () => loadChunks(doc.file_name));
  tr.querySelector(".delete-button").addEventListener("click", () => deleteLibraryFile(doc.file_name));
  return tr;
}

function _extractVersion(doc) {
  // source_path: "data/corpus/pdfs/ocp-4.15/..." → "4.15"
  const sp = doc.source_path || doc.file_name || "";
  const m = sp.match(/ocp-(\d+\.\d+)/);
  return m ? m[1] : null;
}

function _makeVersionTable(docs) {
  const wrap = document.createElement("div");
  wrap.className = "library-table-wrap";
  const table = document.createElement("table");
  table.className = "library-table";
  table.innerHTML = "<thead><tr><th>파일</th><th>상태</th><th>인덱스</th><th>형식</th><th>액션</th></tr></thead><tbody></tbody>";
  const tbody = table.querySelector("tbody");
  docs.forEach((doc) => tbody.appendChild(_makeDocRow(doc)));
  wrap.appendChild(table);
  return wrap;
}

function renderLibrary(documents) {
  libraryList.innerHTML = "";
  if (!documents.length) {
    libraryList.innerHTML = '<div class="empty">업로드된 PDF가 없습니다. 아래 영역에 파일을 올려 주세요.</div>';
    return;
  }

  // 버전별로 그룹핑
  const groups = {};
  const noVersion = [];
  documents.forEach((doc) => {
    const v = _extractVersion(doc);
    if (v) {
      if (!groups[v]) groups[v] = [];
      groups[v].push(doc);
    } else {
      noVersion.push(doc);
    }
  });

  const sortedVersions = Object.keys(groups).sort((a, b) => {
    const [ma, mi_a] = a.split(".").map(Number);
    const [mb, mi_b] = b.split(".").map(Number);
    return ma !== mb ? ma - mb : mi_a - mi_b;
  });

  // 버전 그룹이 있으면 섹션별로 렌더링
  if (sortedVersions.length) {
    sortedVersions.forEach((version) => {
      const section = document.createElement("div");
      section.style.cssText = "margin-bottom: 24px;";
      const header = document.createElement("div");
      header.style.cssText = "display:flex;align-items:center;gap:10px;margin-bottom:10px;";
      header.innerHTML =
        '<span style="font-size:13px;font-weight:800;color:#182538;">OCP ' + escapeHtml(version) + '</span>' +
        '<span style="font-size:12px;color:#66758a;">' + groups[version].length + '개 문서</span>';
      section.appendChild(header);
      section.appendChild(_makeVersionTable(groups[version]));
      libraryList.appendChild(section);
    });
    if (noVersion.length) {
      const section = document.createElement("div");
      section.style.cssText = "margin-bottom: 24px;";
      const header = document.createElement("div");
      header.style.cssText = "display:flex;align-items:center;gap:10px;margin-bottom:10px;";
      header.innerHTML = '<span style="font-size:13px;font-weight:800;color:#182538;">기타</span>';
      section.appendChild(header);
      section.appendChild(_makeVersionTable(noVersion));
      libraryList.appendChild(section);
    }
  } else {
    // 버전 정보 없으면 기존 방식 (단일 테이블)
    libraryList.appendChild(_makeVersionTable(documents));
  }
}

function _renderUploadState(completedDocs, currentFile, pct, completedCount, totalFiles) {
  renderLibrary(completedDocs);

  if (currentFile) {
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
  } else {
    uploadProgressArea.innerHTML = "";
  }
}

function _renderBackgroundIndexingState(state, label) {
  const pct = Number(state.progress_pct || 0);
  const currentChunk = Number(state.current_chunk || 0);
  const totalChunks = Number(state.total_chunks || 0);
  const completedFiles = Number(state.completed_files || 0);
  const totalFiles = Number(state.total_files || 0);
  const currentFile = state.current_file || "준비 중";
  const stageLabel = state.current_stage === "extract"
    ? "추출 중"
    : state.current_stage === "embed"
      ? "임베딩 중"
      : state.current_stage === "done"
        ? "완료 처리 중"
        : "준비 중";

  uploadProgressArea.innerHTML =
    '<div style="padding:14px 16px;border:1px solid #bfd3fb;border-radius:12px;background:#f0f6ff;margin-top:10px;">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">' +
        '<span style="font-size:13px;font-weight:700;color:#182538;">' + escapeHtml(label + " · " + currentFile) + "</span>" +
        '<span style="font-size:12px;color:#2a57df;font-weight:700;">' + pct + "%</span>" +
      "</div>" +
      '<div style="background:#dde4ef;border-radius:999px;height:6px;overflow:hidden;">' +
        '<div style="height:100%;background:#2a57df;border-radius:999px;transition:width 0.15s ease;width:' + pct + '%;"></div>' +
      "</div>" +
      '<div style="font-size:11px;color:#66758a;margin-top:6px;">파일 ' + completedFiles + "/" + totalFiles +
      (totalChunks > 0 ? " · " + stageLabel + " · 청크 " + currentChunk + "/" + totalChunks : " · " + stageLabel) +
      "</div>" +
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
  if (fileInput) fileInput.disabled = isBusy;
  if (refreshLibraryBtn) refreshLibraryBtn.disabled = false;
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

    if (startupBusy) {
      _renderBackgroundIndexingState(startupState, "자동 인덱싱");
      setLibraryStatus("자동 인덱싱 중: " + (startupState.current_file || "준비 중"), "loading", "Startup Indexing");
      if (!_startupPollTimer) {
        _startupPollTimer = setInterval(loadLibrary, 2000);
      }
      return;
    }

    if (reindexBusy) {
      _renderBackgroundIndexingState(reindexState, "전체 재인덱싱");
      setLibraryStatus("전체 재인덱싱 중: " + (reindexState.current_file || "준비 중"), "loading", "Reindexing");
      if (!_startupPollTimer) {
        _startupPollTimer = setInterval(loadLibrary, 2000);
      }
      return;
    }

    if (_startupPollTimer) {
      clearInterval(_startupPollTimer);
      _startupPollTimer = null;
    }
    if (!_isReindexSubmitting) {
      uploadProgressArea.innerHTML = "";
    }
    _startupIndexingFile = "";
    setLibraryStatus("문서 " + data.total_files + "개를 확인했습니다.", "success", "Ready");
  } catch (error) {
    updateLibraryStats({ total_files: 0, indexed_documents: [] });
    renderLibrary([]);
    _setLibraryBusyState(false);
    setLibraryStatus("자료실을 불러오지 못했습니다. " + error.message, "error", "Error");
  }
}

async function deleteLibraryFile(fileName) {
  if (!window.confirm("자료실에서 '" + fileName + "' 문서를 삭제할까요?\n추출된 마크다운도 함께 삭제됩니다.")) return;
  setLibraryStatus(fileName + " 삭제 중입니다.", "loading", "Deleting");
  try {
    const response = await fetch("/api/library?file_name=" + encodeURIComponent(fileName), { method: "DELETE" });
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();
    const markdownText = data.deleted_markdown ? "마크다운 삭제됨" : "마크다운 없음";
    setLibraryStatus(data.deleted_file + " 삭제 완료 · " + markdownText + " · 남은 파일 " + data.indexed_files + "개", "success", "Deleted");
    await loadLibrary();
  } catch (error) {
    setLibraryStatus("문서 삭제에 실패했습니다. " + error.message, "error", "Delete Failed");
  }
}

async function uploadFiles() {
  const files = fileInput.files;
  if (!files.length) return;
  const totalFiles = files.length;
  const fileNames = Array.from(files).map((f) => f.name);
  const formData = new FormData();
  for (const file of files) formData.append("files", file);

  setLibraryStatus("업로드 중입니다. OCR, 청킹, 임베딩, 인덱싱을 처리합니다.", "loading", "Uploading");

  let completedDocs = [];
  try {
    const libraryRes = await fetch("/api/library");
    if (libraryRes.ok) {
      const libraryData = await libraryRes.json();
      completedDocs = (libraryData.indexed_documents || []).filter((d) => d.indexed_chunks > 0);
    }
  } catch (_) {}

  let completedCount = 0;
  _renderUploadState(completedDocs, fileNames[0], 0, 0, totalFiles);

  try {
    const response = await fetch("/api/library/upload", { method: "POST", body: formData });
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
        } else if (data.type === "file_indexed") {
          completedCount += 1;
          totalChunks += data.indexed_chunks || 0;
          const docInfo = ((data.library && data.library.indexed_documents) || []).find((d) => d.file_name === data.file);
          if (docInfo) completedDocs.push(docInfo);
          if (data.library) updateLibraryStats(data.library);
          const nextFile = fileNames[completedCount] || null;
          _renderUploadState(completedDocs, nextFile, 0, completedCount, totalFiles);
          setLibraryStatus(data.file + " 완료 (" + completedCount + "/" + totalFiles + ") · 누적 청크 " + totalChunks + "개", "loading", "Uploading");
        } else if (data.type === "file_error") {
          completedCount += 1;
          const nextFile = fileNames[completedCount] || null;
          _renderUploadState(completedDocs, nextFile, 0, completedCount, totalFiles);
          setLibraryStatus(data.file + " 오류: " + data.error, "error", "Error");
        } else if (data.type === "done") {
          setLibraryStatus("업로드 및 인덱싱 완료: 청크 " + totalChunks + "개", "success", "Completed");
        }
      }
    }
    fileInput.value = "";
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
  uploadFiles();
}

async function reindexAll() {
  _isReindexSubmitting = true;
  _setLibraryBusyState(true);
  setLibraryStatus("전체 라이브러리를 다시 인덱싱하는 중입니다.", "loading", "Reindexing");
  if (!_startupPollTimer) {
    _startupPollTimer = setInterval(loadLibrary, 2000);
  }
  try {
    const response = await fetch("/api/reindex", { method: "POST" });
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();
    setLibraryStatus("재인덱싱 완료: 파일 " + data.indexed_files + "개, 청크 " + data.indexed_chunks + "개", "success", "Completed");
  } catch (error) {
    setLibraryStatus("재인덱싱에 실패했습니다. " + error.message, "error", "Reindex Failed");
  } finally {
    _isReindexSubmitting = false;
    await loadLibrary();
  }
}
