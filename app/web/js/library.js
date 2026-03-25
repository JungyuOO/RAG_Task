function updateLibraryStats(data) {
  const documents = data.indexed_documents || [];
  const totalFiles = Number(data.total_files || documents.length || 0);
  const indexedFiles = documents.length;
  const totalChunks = documents.reduce((sum, doc) => sum + Number(doc.indexed_chunks || 0), 0);
  if (totalFilesStat) totalFilesStat.textContent = totalFiles.toLocaleString("ko-KR");
  if (indexedFilesStat) indexedFilesStat.textContent = indexedFiles.toLocaleString("ko-KR");
  if (indexedChunksStat) indexedChunksStat.textContent = totalChunks.toLocaleString("ko-KR");
}

// 현재 자동 인덱싱 중인 파일명 (폴링 시 갱신)
let _startupIndexingFile = "";

function _makeDocRow(doc) {
  const tr = document.createElement("tr");
  const loaders = doc.loaders && doc.loaders.length ? doc.loaders.join(", ") : "미인식";
  const isIndexed = Number(doc.indexed_chunks) > 0;
  const isCurrentlyIndexing = !isIndexed && _startupIndexingFile === doc.file_name;
  let statusBadge;
  if (isIndexed) {
    statusBadge = '<span class="status-badge">인덱싱 완료</span>';
  } else if (isCurrentlyIndexing) {
    statusBadge = '<span class="status-badge indexing">인덱싱 중…</span>';
  } else {
    statusBadge = '<span class="status-badge not-indexed">대기 중</span>';
  }
  tr.innerHTML =
    '<td><div class="item-title">' + escapeHtml(doc.file_name) + '</div>' +
    '<div class="item-copy">로드 방식: ' + escapeHtml(loaders) + '</div></td>' +
    '<td>' + statusBadge + '</td>' +
    '<td>청크 ' + doc.indexed_chunks + '<br />페이지 ' + doc.indexed_pages + '</td>' +
    '<td>' + escapeHtml(String(doc.extension || "").toUpperCase()) + '</td>' +
    '<td><div class="row-actions">' +
    '<button class="secondary mini-button preview-button" type="button">미리보기</button>' +
    '<button class="secondary mini-button delete-button" type="button">삭제</button>' +
    '</div></td>';
  tr.querySelector(".preview-button").addEventListener("click", () => openPdf(doc.file_name));
  tr.querySelector(".delete-button").addEventListener("click", () => deleteLibraryFile(doc.file_name));
  return tr;
}

function renderLibrary(documents) {
  libraryList.innerHTML = "";
  if (!documents.length) {
    libraryList.innerHTML = '<div class="empty">업로드된 PDF가 없습니다. 위 영역에 파일을 올려 주세요.</div>';
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "library-table-wrap";
  const table = document.createElement("table");
  table.className = "library-table";
  table.innerHTML = '<thead><tr><th>파일</th><th>상태</th><th>인덱싱</th><th>형식</th><th>액션</th></tr></thead><tbody></tbody>';
  const tbody = table.querySelector("tbody");
  documents.forEach((doc) => tbody.appendChild(_makeDocRow(doc)));
  wrap.appendChild(table);
  libraryList.appendChild(wrap);
}

// 업로드 진행 중 libraryList 렌더링:
// completedDocs = 이미 완료된 파일들, currentFile = 지금 처리 중인 파일명, pct = 진행률 0~100
function _renderUploadState(completedDocs, currentFile, pct, completedCount, totalFiles) {
  libraryList.innerHTML = "";

  // 완료된 파일 테이블
  if (completedDocs.length > 0) {
    const wrap = document.createElement("div");
    wrap.className = "library-table-wrap";
    const table = document.createElement("table");
    table.className = "library-table";
    table.innerHTML = '<thead><tr><th>파일</th><th>상태</th><th>인덱싱</th><th>형식</th><th>액션</th></tr></thead><tbody></tbody>';
    const tbody = table.querySelector("tbody");
    completedDocs.forEach((doc) => tbody.appendChild(_makeDocRow(doc)));
    wrap.appendChild(table);
    libraryList.appendChild(wrap);
  }

  // 현재 파일 진행률 바
  if (currentFile) {
    const el = document.createElement("div");
    el.id = "upload-progress-item";
    el.style.cssText =
      "padding:14px 16px;border:1px solid #bfd3fb;border-radius:12px;" +
      "background:#f0f6ff;margin-top:" + (completedDocs.length > 0 ? "10px" : "0") + ";";
    el.innerHTML =
      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">' +
        '<span style="font-size:13px;font-weight:700;color:#182538;">' + escapeHtml(currentFile) + '</span>' +
        '<span id="upload-pct-label" style="font-size:12px;color:#2a57df;font-weight:700;">' + pct + '%</span>' +
      '</div>' +
      '<div style="background:#dde4ef;border-radius:999px;height:6px;overflow:hidden;">' +
        '<div id="upload-pct-bar" style="height:100%;background:#2a57df;border-radius:999px;transition:width 0.15s ease;width:' + pct + '%;"></div>' +
      '</div>' +
      '<div style="font-size:11px;color:#66758a;margin-top:6px;">처리 중 (' + (completedCount + 1) + '/' + totalFiles + ')</div>';
    libraryList.appendChild(el);
  }
}

// 이미 렌더링된 진행률 바만 업데이트 (DOM 재생성 없이)
function _updateProgressBar(pct) {
  const bar = document.getElementById("upload-pct-bar");
  const label = document.getElementById("upload-pct-label");
  if (bar) bar.style.width = pct + "%";
  if (label) label.textContent = pct + "%";
}

let _startupPollTimer = null;

async function loadLibrary() {
  setLibraryStatus("자료실 상태를 불러오는 중입니다.", "loading", "Loading");
  try {
    const response = await fetch("/api/library");
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();

    // 자동 인덱싱 상태 처리
    const si = data.startup_indexing || {};
    _startupIndexingFile = si.current_file || "";

    updateLibraryStats(data);
    renderLibrary(data.indexed_documents || []);

    if (si.status === "indexing") {
      const msg = si.current_file
        ? "자동 인덱싱 중: " + si.current_file + " (" + si.completed + "/" + si.total + ")"
        : "자동 인덱싱 준비 중… (" + si.completed + "/" + si.total + ")";
      setLibraryStatus(msg, "loading", "Indexing");
      // 2초마다 폴링
      if (!_startupPollTimer) {
        _startupPollTimer = setInterval(loadLibrary, 2000);
      }
    } else {
      // 인덱싱 완료 또는 idle — 폴링 중지
      if (_startupPollTimer) {
        clearInterval(_startupPollTimer);
        _startupPollTimer = null;
      }
      _startupIndexingFile = "";
      setLibraryStatus("문서 " + data.total_files + "개를 확인했습니다.", "success", "Ready");
    }
  } catch (error) {
    updateLibraryStats({ total_files: 0, indexed_documents: [] });
    renderLibrary([]);
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

  setLibraryStatus("업로드 중입니다. OCR, 청킹, 인덱싱을 처리합니다.", "loading", "Uploading");

  // 기존 파일 목록을 미리 가져와서 업로드 중에도 계속 표시
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
          // 진행률 바만 빠르게 업데이트
          _updateProgressBar(data.pct);
        } else if (data.type === "file_indexed") {
          completedCount += 1;
          totalChunks += data.indexed_chunks || 0;
          // 완료된 파일 정보를 library 응답에서 가져옴
          const docInfo = (data.library && data.library.indexed_documents || []).find(
            (d) => d.file_name === data.file
          );
          if (docInfo) completedDocs.push(docInfo);
          if (data.library) updateLibraryStats(data.library);
          const nextFile = fileNames[completedCount] || null;
          _renderUploadState(completedDocs, nextFile, 0, completedCount, totalFiles);
          setLibraryStatus(
            data.file + " 완료 (" + completedCount + "/" + totalFiles + ") · 누적 청크 " + totalChunks + "개",
            "loading", "Uploading"
          );
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
  setLibraryStatus("전체 라이브러리를 다시 인덱싱하는 중입니다.", "loading", "Reindexing");
  try {
    const response = await fetch("/api/reindex", { method: "POST" });
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();
    setLibraryStatus("재인덱싱 완료: 파일 " + data.indexed_files + "개, 청크 " + data.indexed_chunks + "개", "success", "Completed");
    await loadLibrary();
  } catch (error) {
    setLibraryStatus("재인덱싱에 실패했습니다. " + error.message, "error", "Reindex Failed");
  }
}
