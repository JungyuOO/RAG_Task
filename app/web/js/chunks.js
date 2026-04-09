// app/web/js/chunks.js

let chunkLoadTimer = null;
let chunkLoadProgress = 0;

function getChunkViewerElements() {
  const overlay = document.getElementById("chunk-viewer");
  if (!overlay) return { overlay: null, container: null };
  const container = overlay.querySelector("div") || overlay;
  return { overlay, container };
}

function nextChunkPaintFrame() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(resolve);
    });
  });
}

function stopChunkLoadingTimer() {
  if (!chunkLoadTimer) return;
  clearInterval(chunkLoadTimer);
  chunkLoadTimer = null;
}

function closeChunkViewer() {
  const { overlay } = getChunkViewerElements();
  if (!overlay) return;
  stopChunkLoadingTimer();
  overlay.style.display = "none";
}

function renderChunkLoadingState(fileName, progress, message) {
  const { overlay, container } = getChunkViewerElements();
  if (!overlay || !container) return;

  container.innerHTML = `
    <div class="chunk-viewer-header">
      <h3>${escapeHtml(fileName)} chunk loading</h3>
      <button type="button" onclick="closeChunkViewer()">Close</button>
    </div>
    <div class="chunk-loading-shell">
      <div class="chunk-loading-copy">${escapeHtml(message)}</div>
      <div class="chunk-loading-progress">
        <div class="chunk-loading-progress-bar" style="width:${progress}%"></div>
      </div>
      <div class="chunk-loading-meta">${progress}%</div>
    </div>
  `;

  overlay.style.display = "block";
  overlay.onclick = function (event) {
    if (event.target === overlay) closeChunkViewer();
  };
}

function startChunkLoading(fileName) {
  chunkLoadProgress = 0;
  renderChunkLoadingState(fileName, chunkLoadProgress, "Loading chunk list...");
  stopChunkLoadingTimer();
  chunkLoadTimer = setInterval(() => {
    chunkLoadProgress = Math.min(
      90,
      chunkLoadProgress + (chunkLoadProgress < 30 ? 9 : chunkLoadProgress < 60 ? 6 : 3)
    );
    renderChunkLoadingState(fileName, chunkLoadProgress, "Reading chunks and metadata...");
  }, 220);
}

function finishChunkLoading(fileName, callback) {
  stopChunkLoadingTimer();
  chunkLoadProgress = 100;
  renderChunkLoadingState(fileName, chunkLoadProgress, "Chunk list ready.");
  setTimeout(callback, 180);
}

function failChunkLoading(fileName, error) {
  stopChunkLoadingTimer();
  renderChunkLoadingState(fileName, 100, "Failed to load chunks.");
  console.error("Chunk load failed:", error);
}

async function loadChunks(fileName, page = 1) {
  startChunkLoading(fileName);
  await nextChunkPaintFrame();

  try {
    const res = await fetch(
      `/api/library/${encodeURIComponent(fileName)}/chunks?page=${page}&page_size=20`
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    finishChunkLoading(fileName, () => renderChunkList(data));
  } catch (error) {
    failChunkLoading(fileName, error);
  }
}

function renderChunkList(data) {
  const { overlay, container } = getChunkViewerElements();
  if (!overlay || !container) return;

  const totalPages = Math.max(1, Math.ceil(data.total / data.page_size));
  container.innerHTML = `
    <div class="chunk-viewer-header">
      <h3>${escapeHtml(data.file_name)} chunk list (${data.total})</h3>
      <button type="button" onclick="closeChunkViewer()">Close</button>
    </div>
    <div class="chunk-list">
      ${data.chunks
        .map(
          (chunk) => `
            <div class="chunk-item" data-chunk-id="${chunk.chunk_id}">
              <div class="chunk-header">
                <span class="chunk-id">#${chunk.chunk_id.substring(0, 12)}...</span>
                <span class="chunk-page">p.${chunk.page_number ?? "-"}</span>
                <span class="chunk-tokens">${chunk.token_count} tokens</span>
              </div>
              <div class="chunk-text">${escapeHtml(chunk.text)}</div>
            </div>
          `
        )
        .join("")}
    </div>
    <div class="chunk-pagination">
      ${
        data.page > 1
          ? `<button type="button" onclick="loadChunks(${JSON.stringify(data.file_name)}, ${
              data.page - 1
            })">Prev</button>`
          : ""
      }
      <span>${data.page} / ${totalPages}</span>
      ${
        data.page < totalPages
          ? `<button type="button" onclick="loadChunks(${JSON.stringify(data.file_name)}, ${
              data.page + 1
            })">Next</button>`
          : ""
      }
    </div>
  `;

  overlay.style.display = "block";
  overlay.onclick = function (event) {
    if (event.target === overlay) closeChunkViewer();
  };
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

window.loadChunks = loadChunks;
window.closeChunkViewer = closeChunkViewer;
window.startChunkLoading = startChunkLoading;
