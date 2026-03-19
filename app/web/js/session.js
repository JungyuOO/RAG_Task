function generateSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return "session-" + Date.now() + "-" + Math.random().toString(16).slice(2, 10);
}

function ensureSessionId() {
  if (!activeSessionId) {
    activeSessionId = generateSessionId();
    saveActiveSessionId();
  }
  return activeSessionId;
}

function saveActiveSessionId() {
  if (activeSessionId) {
    localStorage.setItem(STORAGE_KEYS.activeSessionId, activeSessionId);
  }
}

function readPendingChatState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.pendingChat);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    localStorage.removeItem(STORAGE_KEYS.pendingChat);
    return null;
  }
}

function persistPendingChatState(state) {
  localStorage.setItem(STORAGE_KEYS.pendingChat, JSON.stringify(state));
}

function updatePendingChatState(patch) {
  const state = readPendingChatState();
  if (!state) return;
  persistPendingChatState({ ...state, ...patch, updated_at: Date.now() });
}

function clearPendingChatState() {
  localStorage.removeItem(STORAGE_KEYS.pendingChat);
}

function resetChatSurface() {
  chatLog.innerHTML = "";
  chatPinnedToBottom = true;
  resetPreview();
  renderChatAttachments();
}

function startNewSession() {
  activeSessionId = generateSessionId();
  saveActiveSessionId();
  clearPendingChatState();
  resetChatSurface();
  renderSessionSelection();
  switchScreen("chat");
  historyStatus.textContent = "새 대화를 시작했습니다.";
  messageInput.focus();
}

function renderSessionSelection() {
  sessionHistory.querySelectorAll(".history-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.sessionId === activeSessionId);
  });
}

function formatSessionTime(value) {
  if (!value) return "";
  const hasTimezone = /[zZ]|[+\-]\d{2}:\d{2}$/.test(value);
  const normalized = value.includes("T") ? value : value.replace(" ", "T");
  const isoValue = hasTimezone ? normalized : normalized + "Z";
  const date = new Date(isoValue);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("ko-KR", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function deleteSession(sessionId) {
  if (!window.confirm("이 세션을 삭제할까요?")) return;
  const response = await fetch("/api/sessions/" + encodeURIComponent(sessionId), { method: "DELETE" });
  if (!response.ok) {
    historyStatus.textContent = "세션 삭제에 실패했습니다.";
    return;
  }
  if (activeSessionId === sessionId) startNewSession();
  historyStatus.textContent = "세션을 삭제했습니다.";
  await loadSessions();
}

function renderSessionHistory(items) {
  sessionHistory.innerHTML = "";
  if (!items.length) {
    sessionHistory.innerHTML = '<div class="empty">아직 저장된 대화가 없습니다. 첫 질문을 보내면 여기에 표시됩니다.</div>';
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "history-item";
    card.dataset.sessionId = item.session_id;
    const title = item.last_user_message || item.title || "새 채팅";
    const timeText = formatSessionTime(item.last_user_at || item.updated_at);
    card.innerHTML = '<div class="history-item-head"><div class="item-title">' + escapeHtml(title) + '</div><button class="history-delete" type="button" title="세션 삭제">x</button></div><div class="item-meta">' + escapeHtml(timeText || "시간 정보 없음") + '</div>';
    card.addEventListener("click", () => loadSession(item.session_id));
    card.querySelector(".history-delete").addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(item.session_id);
    });
    sessionHistory.appendChild(card);
  });
  renderSessionSelection();
}

async function loadSessions() {
  historyStatus.textContent = "대화 이력을 불러오는 중입니다.";
  try {
    const response = await fetch("/api/sessions");
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    const data = await response.json();
    const sessions = data.sessions || [];
    renderSessionHistory(sessions);
    historyStatus.textContent = "저장된 세션 " + sessions.length + "개";
  } catch (error) {
    sessionHistory.innerHTML = '<div class="empty">세션 목록을 불러오지 못했습니다.</div>';
    historyStatus.textContent = "세션 목록 로드 실패: " + error.message;
  }
}

async function loadSession(sessionId) {
  activeSessionId = sessionId;
  saveActiveSessionId();
  resetChatSurface();
  currentContextPayload = null;
  const response = await fetch("/api/sessions/" + encodeURIComponent(sessionId));
  if (!response.ok) throw new Error(await extractErrorMessage(response));
  const data = await response.json();
  const turns = data.turns || [];
  turns.forEach((turn) => appendMessage(turn.role, turn.content, turn.metadata || null));
  const assistantTurns = turns.filter((turn) => turn.role === "assistant" && turn.metadata && Array.isArray(turn.metadata.preview_pages) && turn.metadata.preview_pages.length);
  if (assistantTurns.length) {
    renderAnswerPreview(assistantTurns[assistantTurns.length - 1].metadata);
  }
  renderSessionSelection();
  historyStatus.textContent = turns.length ? "이전 대화를 불러왔습니다." : "빈 세션입니다.";
  switchScreen("chat");
  return data;
}
