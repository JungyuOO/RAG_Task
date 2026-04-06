// app/web/js/citation.js
// 인용 태그 렌더링 및 PDF 하이라이트 뷰어 연동

/**
 * 텍스트 내 [source:file.pdf:p12:L5-10] 태그를 클릭 가능한 <sup> 태그로 변환
 */
function renderCitationTags(text) {
  const pattern = /\[source:([^:]+):p(\d+):L(\d+)-(\d+)\]/g;
  let index = 0;
  return text.replace(pattern, (match, fileName, page, lineStart, lineEnd) => {
    index++;
    return `<sup class="citation-tag" ` +
      `data-file="${fileName}" ` +
      `data-page="${page}" ` +
      `data-line-start="${lineStart}" ` +
      `data-line-end="${lineEnd}" ` +
      `title="${fileName} p.${page} L${lineStart}-${lineEnd}">[${index}]</sup>`;
  });
}

/**
 * 인용 태그 클릭 이벤트 바인딩
 */
function bindCitationClicks(container) {
  container.querySelectorAll('.citation-tag').forEach(tag => {
    tag.addEventListener('click', () => {
      const fileName = tag.dataset.file;
      const page = parseInt(tag.dataset.page);
      const lineStart = parseInt(tag.dataset.lineStart);
      const lineEnd = parseInt(tag.dataset.lineEnd);
      openHighlightedPreview(fileName, page, lineStart, lineEnd);
    });
  });
}

/**
 * 하이라이트된 PDF 페이지 미리보기 열기
 */
async function openHighlightedPreview(fileName, page, lineStart, lineEnd) {
  const url = `/api/library/highlight?file_name=${encodeURIComponent(fileName)}&page=${page}&line_start=${lineStart}&line_end=${lineEnd}`;
  const previewPanel = document.getElementById('preview-panel');
  if (!previewPanel) return;
  previewPanel.innerHTML = `<img src="${url}" class="highlighted-page" alt="Page ${page}" style="max-width:100%;border:1px solid #ddd;" />`;
  previewPanel.classList.add('active');
  previewPanel.style.display = 'block';
}
