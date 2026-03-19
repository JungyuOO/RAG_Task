function updateLibraryStats(data) {
  const documents = data.indexed_documents || [];
  const totalFiles = Number(data.total_files || documents.length || 0);
  const indexedFiles = documents.length;
  const totalChunks = documents.reduce((sum, doc) => sum + Number(doc.indexed_chunks || 0), 0);
  if (totalFilesStat) totalFilesStat.textContent = totalFiles.toLocaleString("ko-KR");
  if (indexedFilesStat) indexedFilesStat.textContent = indexedFiles.toLocaleString("ko-KR");
  if (indexedChunksStat) indexedChunksStat.textContent = totalChunks.toLocaleString("ko-KR");
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

  documents.forEach((doc) => {
    const tr = document.createElement("tr");
    const loaders = doc.loaders && doc.loaders.length ? doc.loaders.join(", ") : "미인식";
    tr.innerHTML = '<td><div class="item-title">' + escapeHtml(doc.file_name) + '</div><div class="item-copy">로드 방식: ' + escapeHtml(loaders) + '</div></td><td><span class="status-badge">인덱싱 완료</span></td><td>청크 ' + doc.indexed_chunks + '<br />페이지 ' + doc.indexed_pages + '</td><td>' + escapeHtml(String(doc.extension || "").toUpperCase()) + '</td><td><div class="row-actions"><button class="secondary mini-button preview-button" type="button">미리보기</button><button class="secondary mini-button delete-button" type="button">삭제</button></div></td>';
    tr.querySelector(".preview-button").addEventListener("click", () => openPdf(doc.file_name));
    tr.querySelector(".delete-button").addEventListener("click", () => deleteLibraryFile(doc.file_name));
    tbody.appendChild(tr);
  });

  wrap.appendChild(table);
  libraryList.appendChild(wrap);
}

async function loadLibrary() {
  setLibraryStatus("자료실 상태를 불러오는 중입니다.", "loading", "Loading");
  try {
    const response = await fetch("/api/library");
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();
    updateLibraryStats(data);
    renderLibrary(data.indexed_documents || []);
    setLibraryStatus("문서 " + data.total_files + "개를 확인했습니다.", "success", "Ready");
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
  const formData = new FormData();
  for (const file of files) formData.append("files", file);
  setLibraryStatus("업로드 중입니다. OCR, 청킹, 인덱싱을 처리합니다.", "loading", "Uploading");
  try {
    const response = await fetch("/api/library/upload", { method: "POST", body: formData });
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();
    setLibraryStatus("업로드 및 인덱싱 완료: 청크 " + data.indexed_chunks + "개", "success", "Completed");
    fileInput.value = "";
    await loadLibrary();
  } catch (error) {
    setLibraryStatus("업로드에 실패했습니다. " + error.message, "error", "Upload Failed");
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
