// app/web/js/chunks.js

async function loadChunks(fileName, page = 1) {
  try {
    const res = await fetch(`/api/library/${encodeURIComponent(fileName)}/chunks?page=${page}&page_size=20`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderChunkList(data);
  } catch (err) {
    console.error('청크 로드 실패:', err);
  }
}

function renderChunkList(data) {
  const overlay = document.getElementById('chunk-viewer');
  if (!overlay) return;
  const container = overlay.querySelector('div') || overlay;
  const totalPages = Math.ceil(data.total / data.page_size);
  container.innerHTML = `
    <div class="chunk-viewer-header">
      <h3>${data.file_name} — 청크 목록 (${data.total}개)</h3>
      <button onclick="document.getElementById('chunk-viewer').style.display='none'">닫기</button>
    </div>
    <div class="chunk-list">
      ${data.chunks.map(c => `
        <div class="chunk-item" data-chunk-id="${c.chunk_id}">
          <div class="chunk-header">
            <span class="chunk-id">#${c.chunk_id.substring(0, 12)}…</span>
            <span class="chunk-page">p.${c.page_number ?? '-'}</span>
            <span class="chunk-tokens">${c.token_count} tokens</span>
          </div>
          <div class="chunk-text">${escapeHtml(c.text)}</div>
        </div>
      `).join('')}
    </div>
    <div class="chunk-pagination">
      ${data.page > 1 ? `<button onclick="loadChunks('${data.file_name}', ${data.page - 1})">이전</button>` : ''}
      <span>${data.page} / ${totalPages}</span>
      ${data.page < totalPages ? `<button onclick="loadChunks('${data.file_name}', ${data.page + 1})">다음</button>` : ''}
    </div>
  `;
  overlay.style.display = 'block';
  // 오버레이 배경 클릭 시 닫기
  overlay.onclick = function(e) { if (e.target === overlay) overlay.style.display = 'none'; };
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
