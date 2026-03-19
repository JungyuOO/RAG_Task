function setComposerPending(pending) {
  isSending = pending;
  composerShell.classList.toggle("pending", pending);
  messageInput.disabled = pending;
  sendBtn.disabled = pending;
  attachPdfBtn.disabled = pending;
  clearInputBtn.disabled = pending;
}

function attachSourceButton(block, metadata) {
  if (!metadata || !Array.isArray(metadata.preview_pages) || !metadata.preview_pages.length) return;
  let actions = block.querySelector(".message-actions");
  if (!actions) {
    actions = document.createElement("div");
    actions.className = "message-actions";
    block.appendChild(actions);
  } else {
    actions.innerHTML = "";
  }

  const button = document.createElement("button");
  button.type = "button";
  button.className = "secondary mini-button";
  button.textContent = "자료 보기";
  button.addEventListener("click", () => {
    renderAnswerPreview(metadata);
  });
  actions.appendChild(button);
}

function appendMessage(role, text, metadata) {
  const shouldStick = chatPinnedToBottom;
  const block = document.createElement("div");
  block.className = "message " + role;
  const roleLabel = role === "user" ? "You" : "CW AI Assistant";
  block.innerHTML = '<div class="message-role">' + roleLabel + '</div><div class="message-body"></div>';
  const body = block.querySelector(".message-body");
  if (role === "assistant") {
    renderAssistantText(body, text, { final: true });
  } else {
    body.textContent = text;
  }
  if (role === "assistant") {
    attachSourceButton(block, metadata || null);
  }
  chatLog.appendChild(block);
  scrollChatToBottom(shouldStick);
  return { block, body };
}

function appendAssistantLoading() {
  const shouldStick = chatPinnedToBottom;
  const block = document.createElement("div");
  block.className = "message assistant loading";
  block.innerHTML = '<div class="message-role">CW AI Assistant</div><div class="message-body"><div class="loading-indicator"><div class="loading-copy"><div class="loading-title">답변 생성 중</div><div class="loading-subtitle">문서와 대화 내용을 바탕으로 응답을 준비하고 있습니다.</div></div><div class="loading-meta"><div class="loading-dots" aria-hidden="true"><span></span><span></span><span></span></div><span class="loading-elapsed">0.0초</span></div></div></div>';
  chatLog.appendChild(block);
  scrollChatToBottom(shouldStick);

  const body = block.querySelector(".message-body");
  const elapsedNode = block.querySelector(".loading-elapsed");
  const startedAt = Date.now();
  const intervalId = window.setInterval(() => {
    elapsedNode.textContent = ((Date.now() - startedAt) / 1000).toFixed(1) + "초";
  }, 100);

  return {
    block,
    setText(text) {
      window.clearInterval(intervalId);
      block.classList.remove("loading");
      renderAssistantText(body, text, { final: true });
      scrollChatToBottom(true);
    },
    setPartialText(text) {
      renderAssistantText(body, text, { final: false });
      scrollChatToBottom(true);
    },
    setError(text) {
      window.clearInterval(intervalId);
      block.classList.remove("loading");
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
      if (payload.type === "context") {
        currentContextPayload = payload;
        if (payload.preview_finalized) {
          finalContextPayload = payload;
          renderAnswerPreview(payload);
        }
      }
      if (payload.type === "token") {
        assistantText += payload.content;
        assistantState.setText(assistantText);
        updatePendingChatState({ partial_response: assistantText });
      }
      if (payload.type === "done") {
        sawDone = true;
        attachSourceButton(assistantState.block, finalContextPayload || currentContextPayload);
        clearPendingChatState();
        setLibraryStatus((payload.cached ? "캐시 응답 완료 (" : "응답 완료 (") + assistantState.elapsedSeconds() + "초)", "success", "Ready");
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
  } catch (error) {
    resetChatSurface();
    appendMessage("user", pendingState.message);
  }

  const turns = Array.from(chatLog.querySelectorAll(".message-role")).map((node) => node.textContent);
  if (!turns.length || turns[turns.length - 1] !== "CW AI Assistant") {
    const assistantState = appendAssistantLoading();
    setComposerPending(true);
    setLibraryStatus("새로고침 이후 답변을 복구하는 중입니다.", "loading", "Recovering");
    try {
      const response = await fetch("/api/chat/retry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: pendingState.session_id,
          message: pendingState.message,
          file_names: pendingState.file_names || [],
        }),
      });
      if (!response.ok) throw new Error(await extractErrorMessage(response));
      await consumeChatStream(response, assistantState, { ...pendingState, is_retry: true });
    } catch (error) {
      assistantState.setError("Assistant 복구 응답 중 오류가 발생했습니다.\n" + error.message);
      setLibraryStatus("복구 실패: " + error.message, "error", "Recovery Failed");
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
  setLibraryStatus(pendingChatFiles.length ? "첨부 문서를 반영해 답변을 준비 중입니다." : "Assistant가 답변 생성 중입니다.", "loading", "Processing");

  try {
    let response;
    if (pendingChatFiles.length) {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("message", message);
      pendingChatFiles.forEach((file) => formData.append("files", file));
      response = await fetch("/api/chat/upload", { method: "POST", body: formData });
    } else {
      response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message }),
      });
    }
    if (!response.ok) throw new Error(await extractErrorMessage(response));
    await consumeChatStream(response, assistantState, pendingState);
  } catch (error) {
    assistantState.setError("Assistant 응답 중 오류가 발생했습니다.\n" + error.message);
    setLibraryStatus("응답 생성 실패: " + error.message, "error", "Chat Failed");
  } finally {
    setComposerPending(false);
    messageInput.focus();
  }
}
