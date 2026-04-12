function setComposerPending(pending) {
  isSending = pending;
  composerShell.classList.toggle("pending", pending);
  messageInput.disabled = pending;
  sendBtn.disabled = pending;
  attachPdfBtn.disabled = pending;
  clearInputBtn.disabled = pending;
}

function _primaryOcpResource(metadata) {
  const queryInterpretation = metadata && metadata.query_interpretation ? metadata.query_interpretation : {};
  const resources = Array.isArray(queryInterpretation.resources) ? queryInterpretation.resources : [];
  const primary = String(resources[0] || "").toLowerCase().trim();
  const mapping = {
    pod: "pods",
    deployment: "deployments",
    service: "services",
    route: "routes",
  };
  return mapping[primary] || "";
}

function _primaryOcpResourceName(metadata) {
  const message = String((metadata && metadata.query) || "").trim();
  const match = /\b([a-z0-9][a-z0-9-]{2,})\b/i.exec(message);
  return match ? match[1] : "";
}

function _buildVisibleSourceEntries(metadata) {
  const entries = [];
  const entryByKey = new Map();

  const upsertEntry = (fileName, pageNumber, sourcePath, blockAnchor = "", htmlAnchor = "") => {
    const normalizedFile = String(fileName || "").trim();
    const normalizedPage = parseInt(pageNumber, 10) || 1;
    const normalizedSourcePath = String(sourcePath || "").trim();
    if (!normalizedFile) return;
    const key = [normalizedFile, normalizedPage, normalizedSourcePath].join("|");
    const existing = entryByKey.get(key);
    if (existing) {
      if (!existing.blockAnchor && blockAnchor) existing.blockAnchor = String(blockAnchor);
      if ((!existing.htmlAnchor || existing.htmlAnchor === ("page-" + normalizedPage)) && htmlAnchor) {
        existing.htmlAnchor = String(htmlAnchor);
      }
      if (!existing.sourcePath && normalizedSourcePath) existing.sourcePath = normalizedSourcePath;
      return;
    }
    const entry = {
      fileName: normalizedFile,
      pageNumber: normalizedPage,
      sourcePath: normalizedSourcePath || normalizedFile,
      blockAnchor: String(blockAnchor || ""),
      htmlAnchor: String(htmlAnchor || ("page-" + normalizedPage)),
    };
    entryByKey.set(key, entry);
    entries.push(entry);
  };

  const previewPages = Array.isArray(metadata.preview_pages) ? metadata.preview_pages : [];
  previewPages.forEach((page) => {
    const sourcePath = String(page.source_path || "");
    const fileName = sourcePath.split(/[\\/]/).pop() || sourcePath;
    upsertEntry(fileName, page.page_number, sourcePath, page.block_anchor, page.html_anchor);
  });

  const answerCitations = Array.isArray(metadata.answer_citations) ? metadata.answer_citations : [];
  answerCitations.forEach((citation) => {
    upsertEntry(citation.file_name, citation.page_number, citation.source_path);
  });

  return entries.slice(0, 4);
}

function attachVisibleSourceCards(body, metadata) {
  const entries = _buildVisibleSourceEntries(metadata);
  if (!entries.length) return;
  const hasInlineCitationButton = body.querySelector(".source-ref-inline");
  if (hasInlineCitationButton && entries.length <= 1) {
    return;
  }

  const list = document.createElement("div");
  list.className = "message-source-card-list";

  entries.forEach((entry) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "message-source-card";
    card.innerHTML =
      `<span class="message-source-card-file">${escapeHtml(entry.fileName)}</span>` +
      `<span class="message-source-card-page">p.${entry.pageNumber}</span>`;
    card.addEventListener("click", () => {
      openAnswerPreviewSource(entry.fileName, entry.pageNumber, entry.blockAnchor || entry.htmlAnchor || "", entry.sourcePath);
    });
    list.appendChild(card);
  });

  body.appendChild(list);
}

function attachSourceButton(block, metadata) {
  if (!block || !metadata) {
    return;
  }
  const body = block.querySelector(".message-body");
  if (!body) return;
  body.querySelectorAll(".message-source-card-list").forEach((node) => node.remove());
  body.querySelectorAll(".supporting-card-list").forEach((node) => node.remove());

  attachVisibleSourceCards(body, metadata);

  if (!Array.isArray(metadata.supporting_examples) || !metadata.supporting_examples.length) {
    return;
  }

  const list = document.createElement("div");
  list.className = "supporting-card-list";

  metadata.supporting_examples.forEach((example, index) => {
    const card = document.createElement("div");
    card.className = "supporting-card";

    const header = document.createElement("button");
    header.type = "button";
    header.className = "supporting-card-toggle";
    header.setAttribute("aria-expanded", "false");
    const pageLabel = example.page_start && example.page_end && example.page_start !== example.page_end
      ? `p.${example.page_start}-${example.page_end}`
      : (example.page_start ? `p.${example.page_start}` : "");
    header.innerHTML =
      `<span class="supporting-card-title">${escapeHtml(example.title || `Example ${index + 1}`)}</span>` +
      `<span class="supporting-card-meta">${escapeHtml([example.file_name || "", pageLabel].filter(Boolean).join(" · "))}</span>`;

    const panel = document.createElement("div");
    panel.className = "supporting-card-panel";
    panel.hidden = true;

    const actions = document.createElement("div");
    actions.className = "supporting-card-actions";

    if (example.file_name && example.page_start) {
      const previewBtn = document.createElement("button");
      previewBtn.type = "button";
      previewBtn.className = "secondary mini-button";
      previewBtn.textContent = "근거 보기";
      previewBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        openAnswerPreviewSource(example.file_name, parseInt(example.page_start, 10), example.block_anchor || example.html_anchor || "", example.source_path || example.file_name);
      });
      actions.appendChild(previewBtn);

      const rawPdfBtn = document.createElement("button");
      rawPdfBtn.type = "button";
      rawPdfBtn.className = "secondary mini-button";
      const isMarkdownSource = String(example.file_name || "").toLowerCase().endsWith(".md");
      rawPdfBtn.textContent = isMarkdownSource ? "원본 문서" : "원본 PDF";
      rawPdfBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        openLibrarySource(example.file_name, parseInt(example.page_start, 10), example.block_anchor || example.html_anchor || "", example.source_path || example.file_name);
      });
      actions.appendChild(rawPdfBtn);
    }

    if (example.type === "code") {
      const wrap = document.createElement("div");
      wrap.className = "assistant-code-wrap";
      if (example.language) wrap.dataset.language = example.language;
      const pre = document.createElement("pre");
      pre.className = "assistant-code-block";
      pre.textContent = example.content || "";
      wrap.appendChild(pre);
      panel.appendChild(wrap);
    } else {
      const text = document.createElement("div");
      text.className = "supporting-card-text";
      text.textContent = example.content || "";
      panel.appendChild(text);
    }
    if (actions.childElementCount) {
      panel.appendChild(actions);
    }

    header.addEventListener("click", () => {
      const nextExpanded = header.getAttribute("aria-expanded") !== "true";
      header.setAttribute("aria-expanded", nextExpanded ? "true" : "false");
      panel.hidden = !nextExpanded;
    });

    card.appendChild(header);
    card.appendChild(panel);
    list.appendChild(card);
  });

  body.appendChild(list);
}

function createMessageShell(role, { loading = false } = {}) {
  const block = document.createElement("div");
  block.className = "message " + role + (loading ? " loading" : "");

  const roleNode = document.createElement("div");
  roleNode.className = "message-role";
  roleNode.textContent = role === "user" ? "You" : "CW AI Assistant";

  const body = document.createElement("div");
  body.className = "message-body";

  block.appendChild(roleNode);
  block.appendChild(body);
  return { block, body };
}

function appendMessage(role, text, metadata) {
  const shouldStick = chatPinnedToBottom;
  const { block, body } = createMessageShell(role);
  if (role === "assistant") {
    renderAssistantText(body, text, { final: true });
    attachSourceButton(block, metadata || null);
  } else {
    body.textContent = text;
  }
  chatLog.appendChild(block);
  scrollChatToBottom(shouldStick);
  return { block, body };
}

function appendAssistantLoading() {
  const shouldStick = chatPinnedToBottom;
  const { block, body } = createMessageShell("assistant", { loading: true });
  body.innerHTML = '<div class="loading-indicator"><div class="loading-copy"><div class="loading-title">질문 의도를 분석하는 중입니다.</div><div class="loading-subtitle">관련 문서를 찾고 답변 근거를 준비하고 있습니다.</div></div><div class="loading-meta"><div class="loading-dots" aria-hidden="true"><span></span><span></span><span></span></div><span class="loading-elapsed">0.0초</span></div></div>';
  chatLog.appendChild(block);
  scrollChatToBottom(shouldStick);

  const elapsedNode = body.querySelector(".loading-elapsed");
  const titleNode = body.querySelector(".loading-title");
  const subtitleNode = body.querySelector(".loading-subtitle");
  const startedAt = Date.now();
  let finalized = false;
  const intervalId = window.setInterval(() => {
    if (elapsedNode) {
      elapsedNode.textContent = ((Date.now() - startedAt) / 1000).toFixed(1) + "초";
    }
  }, 100);

  function finalizeLoading() {
    if (finalized) return;
    finalized = true;
    window.clearInterval(intervalId);
    block.classList.remove("loading");
  }

  return {
    block,
    body,
    setStage(message) {
      if (finalized) return;
      if (titleNode) titleNode.textContent = "질문 의도를 분석하는 중입니다.";
      if (subtitleNode) subtitleNode.textContent = message || "답변 준비를 진행하고 있습니다.";
    },
    setText(text) {
      finalizeLoading();
      renderAssistantText(body, text, { final: true });
      scrollChatToBottom(true);
    },
    setPartialText(text) {
      finalizeLoading();
      renderAssistantText(body, text, { final: false });
      scrollChatToBottom(true);
    },
    setError(text) {
      finalizeLoading();
      body.textContent = text;
      scrollChatToBottom(true);
    },
    elapsedSeconds() {
      return ((Date.now() - startedAt) / 1000).toFixed(1);
    },
  };
}

function renderChatAttachments() {
  chatAttachmentList.innerHTML = "";
  pendingChatFiles.forEach((file, index) => {
    const chip = document.createElement("div");
    chip.className = "attachment-chip";
    chip.innerHTML = '<span>' + escapeHtml(file.name) + '</span><button class="attachment-remove" type="button" aria-label="Remove file">x</button>';
    chip.querySelector("button").addEventListener("click", () => {
      pendingChatFiles = pendingChatFiles.filter((_, fileIndex) => fileIndex !== index);
      renderChatAttachments();
    });
    chatAttachmentList.appendChild(chip);
  });
}

function addChatAttachments(files) {
  const nextFiles = Array.from(files).filter((file) => file.name.toLowerCase().endsWith(".pdf"));
  pendingChatFiles = pendingChatFiles.concat(nextFiles);
  renderChatAttachments();
}

async function consumeChatStream(response, assistantState, pendingState) {
  if (!response.body) throw new Error("응답 스트림을 사용할 수 없습니다.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let assistantText = pendingState.partial_response || "";
  let sawDone = false;
  let finalContextPayload = null;
  if (assistantText) assistantState.setPartialText(assistantText);

  while (true) {
    const result = await reader.read();
    if (result.done) break;
    buffer += decoder.decode(result.value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      if (!event.startsWith("data: ")) continue;
      const payload = JSON.parse(event.slice(6));
      if (payload.type === "upload") {
        pendingChatFiles = [];
        renderChatAttachments();
        setLibraryStatus("PDF 업로드 완료: " + (payload.uploaded_files || []).join(", "), "success", "Completed");
        await loadLibrary();
      }
      if (payload.type === "status") {
        if (statusBar) {
          statusBar.show();
          statusBar.addStage(payload.stage, payload.message);
        }
        if (assistantState && typeof assistantState.setStage === "function") {
          assistantState.setStage(payload.message);
        }
      }
      if (payload.type === "context") {
        currentContextPayload = payload;
        if (payload.preview_finalized) {
          finalContextPayload = payload;
          renderAnswerPreview(payload);
        }
      }
      if (payload.type === "token") {
        assistantText += payload.content;
        assistantState.setPartialText(assistantText);
        updatePendingChatState({ partial_response: assistantText });
      }
      if (payload.type === "replace_answer") {
        assistantText = payload.content;
        assistantState.setPartialText(assistantText);
        updatePendingChatState({ partial_response: assistantText });
      }
if (payload.type === "done") {
        sawDone = true;
        assistantState.setText(assistantText);
        attachSourceButton(assistantState.block, finalContextPayload || currentContextPayload);
        if (typeof bindCitationClicks === "function" && assistantState.body) {
          bindCitationClicks(assistantState.body);
        }
        clearPendingChatState();
        setLibraryStatus((payload.cached ? "캐시 응답 완료 (" : "응답 완료 (") + assistantState.elapsedSeconds() + "초)", "success", "Ready");
        if (statusBar) statusBar.complete();
        await loadSessions();
      }
    }
  }

  if (!assistantText.trim() && !sawDone) {
    assistantState.setError("Assistant 응답이 비어 있습니다.");
  }
}

async function retryPendingChat(pendingState) {
  if (!pendingState || !pendingState.session_id || !pendingState.message || isSending) return;
  activeSessionId = pendingState.session_id;
  saveActiveSessionId();
  try {
    await loadSession(activeSessionId);
  } catch (_error) {
    resetChatSurface();
    appendMessage("user", pendingState.message);
  }

  const turns = Array.from(chatLog.querySelectorAll(".message-role")).map((node) => node.textContent);
  if (!turns.length || turns[turns.length - 1] !== "CW AI Assistant") {
    const assistantState = appendAssistantLoading();
    if (statusBar) statusBar.reset();
    setComposerPending(true);
    setLibraryStatus("이전 응답 복구를 시도하고 있습니다.", "loading", "Recovering");
    try {
      const response = await fetch("/api/chat/retry", {
        method: "POST",
        headers: buildOwnerHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          session_id: pendingState.session_id,
          message: pendingState.message,
          file_names: pendingState.file_names || [],
        }),
      });
      if (!response.ok) throw new Error(await extractErrorMessage(response));
      await consumeChatStream(response, assistantState, { ...pendingState, is_retry: true });
    } catch (error) {
      assistantState.setError("Assistant 복구 중 오류가 발생했습니다.\n" + error.message);
      setLibraryStatus("복구에 실패했습니다. " + error.message, "error", "Recovery Failed");
    } finally {
      setComposerPending(false);
    }
  }
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message || isSending) return;

  const sessionId = ensureSessionId();
  const pendingState = {
    session_id: sessionId,
    message,
    file_names: pendingChatFiles.map((file) => file.name),
    partial_response: "",
    created_at: Date.now(),
    updated_at: Date.now(),
  };

  saveActiveSessionId();
  persistPendingChatState(pendingState);
  setComposerPending(true);
  chatPinnedToBottom = true;
  messageInput.value = "";
  localStorage.removeItem(STORAGE_KEYS.draftMessage);
  appendMessage("user", message);
  const assistantState = appendAssistantLoading();
  if (statusBar) statusBar.reset();
  setLibraryStatus(
    pendingChatFiles.length ? "PDF 업로드와 답변 생성을 준비 중입니다." : "Assistant가 답변을 준비 중입니다.",
    "loading",
    "Processing"
  );

  try {
    let response;
    if (pendingChatFiles.length) {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("message", message);
      pendingChatFiles.forEach((file) => formData.append("files", file));
      response = await fetch("/api/chat/upload", {
        method: "POST",
        headers: buildOwnerHeaders(),
        body: formData,
      });
    } else {
      response = await fetch("/api/chat", {
        method: "POST",
        headers: buildOwnerHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ session_id: sessionId, message }),
      });
    }
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    await consumeChatStream(response, assistantState, pendingState);
  } catch (error) {
    assistantState.setError("Assistant 응답 생성에 실패했습니다.\n" + error.message);
    setLibraryStatus("채팅 요청에 실패했습니다. " + error.message, "error", "Chat Failed");
  } finally {
    setComposerPending(false);
    messageInput.focus();
  }
}
