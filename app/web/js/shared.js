const navButtons = Array.from(document.querySelectorAll(".nav-button"));
const chatLayout = document.querySelector(".chat-layout");
const previewPanel = document.querySelector(".preview-main");
const topbarTitle = document.getElementById("topbarTitle");
const topbarCopy = document.getElementById("topbarCopy");
const chatLog = document.getElementById("chatLog");
const statusBox = document.getElementById("status");
const statusLabel = document.getElementById("statusLabel");
const libraryStatusPanel = document.getElementById("libraryStatusPanel");
const historyStatus = document.getElementById("historyStatus");
const messageInput = document.getElementById("messageInput");
const libraryList = document.getElementById("libraryList");
const answerPreviewSummary = document.getElementById("answerPreviewSummary");
const answerPreviewMeta = document.getElementById("answerPreviewMeta");
const answerPageStrip = document.getElementById("answerPageStrip");
const answerPreviewFrame = document.getElementById("answerPreviewFrame");
const sessionHistory = document.getElementById("sessionHistory");
const pdfModal = document.getElementById("pdfModal");
const pdfViewer = document.getElementById("pdfViewer");
const pdfModalTitle = document.getElementById("pdfModalTitle");
const fileInput = document.getElementById("fileInput");
const chatFileInput = document.getElementById("chatFileInput");
const chatAttachmentList = document.getElementById("chatAttachmentList");
const uploadDropzone = document.getElementById("uploadDropzone");
const composerShell = document.getElementById("composerShell");
const sendBtn = document.getElementById("sendBtn");
const attachPdfBtn = document.getElementById("attachPdfBtn");
const clearInputBtn = document.getElementById("clearInputBtn");
const closePreviewBtn = document.getElementById("closePreviewBtn");
const totalFilesStat = document.getElementById("totalFilesStat");
const indexedFilesStat = document.getElementById("indexedFilesStat");
const indexedChunksStat = document.getElementById("indexedChunksStat");
const screens = {
  chat: document.getElementById("screen-chat"),
  library: document.getElementById("screen-library"),
};

const screenMeta = {
  chat: {
    title: "채팅",
    copy: "문서 근거 기반 응답과 일반 fallback 응답을 함께 확인할 수 있습니다.",
  },
  library: {
    title: "자료실",
    copy: "문서 목록과 업로드 상태를 관리합니다.",
  },
};

const STORAGE_KEYS = {
  activeSessionId: "cw-rag.active-session-id",
  pendingChat: "cw-rag.pending-chat",
  draftMessage: "cw-rag.draft-message",
};

let activeSessionId = "";
let currentPdfFileName = "";
let pendingChatFiles = [];
let isSending = false;
let previewAvailable = false;
let previewOpen = false;
let chatPinnedToBottom = true;
let currentContextPayload = null;

function setLibraryStatus(message, tone = "idle", label = "Library Status") {
  statusBox.textContent = message;
  if (statusLabel) statusLabel.textContent = label;
  if (libraryStatusPanel) libraryStatusPanel.dataset.tone = tone;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeAssistantText(value) {
  return String(value || "").replaceAll("**", "").replace(/^#{1,6}\s*/gm, "");
}

function isMarkdownTableRow(line) {
  return /^\s*\|.*\|\s*$/.test(line || "");
}

function isMarkdownTableSeparator(line) {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$/.test(line || "");
}

function parseMarkdownTableRow(line) {
  return String(line || "")
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isCodeFence(line) {
  return /^\s*```/.test(line || "");
}

function isBulletListItem(line) {
  return /^\s*[-*]\s+/.test(line || "");
}

function isOrderedListItem(line) {
  return /^\s*\d+\.\s+/.test(line || "");
}

function getHeadingLevel(line) {
  const match = /^\s*(#{1,3})\s+(.+?)\s*$/.exec(line || "");
  return match ? match[1].length : 0;
}

function buildHeading(line) {
  const match = /^\s*(#{1,3})\s+(.+?)\s*$/.exec(line || "");
  const level = match ? Math.min(match[1].length, 3) : 3;
  const heading = document.createElement("div");
  heading.className = "assistant-heading level-" + level;
  heading.textContent = match ? match[2].trim() : String(line || "").trim();
  return heading;
}

function buildCodeBlock(lines, fenceLine) {
  const wrapper = document.createElement("div");
  wrapper.className = "assistant-code-wrap";
  const code = document.createElement("pre");
  code.className = "assistant-code-block";
  const language = String(fenceLine || "").trim().replace(/^```/, "").trim();
  if (language) {
    wrapper.dataset.language = language;
  }
  code.textContent = lines.join("\n");
  wrapper.appendChild(code);
  return wrapper;
}

function buildList(lines, ordered) {
  const list = document.createElement(ordered ? "ol" : "ul");
  list.className = ordered ? "assistant-list ordered" : "assistant-list";
  lines.forEach((line) => {
    const item = document.createElement("li");
    item.textContent = String(line || "")
      .replace(/^\s*[-*]\s+/, "")
      .replace(/^\s*\d+\.\s+/, "")
      .trim();
    list.appendChild(item);
  });
  return list;
}

function buildMarkdownTable(lines) {
  const table = document.createElement("table");
  table.className = "assistant-table";
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");

  const headerCells = parseMarkdownTableRow(lines[0]);
  const headerRow = document.createElement("tr");
  headerCells.forEach((cell) => {
    const th = document.createElement("th");
    th.textContent = cell;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  lines.slice(2).forEach((line) => {
    const row = document.createElement("tr");
    parseMarkdownTableRow(line).forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = cell;
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  return table;
}

function renderAssistantText(body, value, options = {}) {
  const final = options.final !== false;
  const normalized = normalizeAssistantText(value).replace(/\r\n/g, "\n").trim();
  if (!final) {
    body.textContent = normalized;
    return;
  }

  body.innerHTML = "";
  if (!normalized) return;

  const fragment = document.createDocumentFragment();
  const lines = normalized.split("\n");
  let index = 0;

  while (index < lines.length) {
    if (!lines[index].trim()) {
      index += 1;
      continue;
    }

    if (isCodeFence(lines[index])) {
      const fenceLine = lines[index];
      index += 1;
      const codeLines = [];
      while (index < lines.length && !isCodeFence(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length && isCodeFence(lines[index])) {
        index += 1;
      }
      fragment.appendChild(buildCodeBlock(codeLines, fenceLine));
      continue;
    }

    if (getHeadingLevel(lines[index])) {
      fragment.appendChild(buildHeading(lines[index]));
      index += 1;
      continue;
    }

    if (
      index + 1 < lines.length
      && isMarkdownTableRow(lines[index])
      && isMarkdownTableSeparator(lines[index + 1])
    ) {
      const tableLines = [lines[index], lines[index + 1]];
      index += 2;
      while (index < lines.length && isMarkdownTableRow(lines[index])) {
        tableLines.push(lines[index]);
        index += 1;
      }
      fragment.appendChild(buildMarkdownTable(tableLines));
      continue;
    }

    if (isBulletListItem(lines[index]) || isOrderedListItem(lines[index])) {
      const ordered = isOrderedListItem(lines[index]);
      const listLines = [];
      while (index < lines.length) {
        if (ordered && isOrderedListItem(lines[index])) {
          listLines.push(lines[index]);
          index += 1;
          continue;
        }
        if (!ordered && isBulletListItem(lines[index])) {
          listLines.push(lines[index]);
          index += 1;
          continue;
        }
        break;
      }
      fragment.appendChild(buildList(listLines, ordered));
      continue;
    }

    const textLines = [];
    while (index < lines.length) {
      if (isCodeFence(lines[index])) break;
      if (getHeadingLevel(lines[index])) break;
      const nextStartsTable =
        index + 1 < lines.length
        && isMarkdownTableRow(lines[index])
        && isMarkdownTableSeparator(lines[index + 1]);
      if (nextStartsTable || isBulletListItem(lines[index]) || isOrderedListItem(lines[index])) break;
      textLines.push(lines[index]);
      index += 1;
      if (index < lines.length && !lines[index].trim()) {
        break;
      }
    }

    const block = document.createElement("div");
    block.className = "assistant-text-block";
    block.textContent = textLines.join("\n").trim();
    if (block.textContent) {
      fragment.appendChild(block);
    }
  }

  body.appendChild(fragment);
}

function switchScreen(name) {
  navButtons.forEach((button) => button.classList.toggle("active", button.dataset.screen === name));
  Object.entries(screens).forEach(([key, screen]) => screen.classList.toggle("active", key === name));
  topbarTitle.textContent = screenMeta[name].title;
  topbarCopy.textContent = screenMeta[name].copy;
}

function updateChatPinnedState() {
  const threshold = 40;
  const distanceFromBottom = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight;
  chatPinnedToBottom = distanceFromBottom <= threshold;
}

function scrollChatToBottom(force) {
  if (!force && !chatPinnedToBottom) return;
  chatLog.scrollTop = chatLog.scrollHeight;
  chatPinnedToBottom = true;
}

async function extractErrorMessage(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      const data = await response.json();
      return data.detail || data.message || JSON.stringify(data);
    } catch (error) {
      return "요청에 실패했습니다. (" + response.status + ")";
    }
  }
  const text = await response.text();
  return text ? text.slice(0, 300) : "요청에 실패했습니다. (" + response.status + ")";
}
