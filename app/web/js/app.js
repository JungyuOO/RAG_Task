function bindEventListeners() {
  navButtons.forEach((button) => button.addEventListener("click", () => switchScreen(button.dataset.screen)));
  chatLog.addEventListener("scroll", updateChatPinnedState);
  messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  messageInput.addEventListener("input", () => {
    localStorage.setItem(STORAGE_KEYS.draftMessage, messageInput.value);
  });
  chatFileInput.addEventListener("change", () => {
    if (!chatFileInput.files.length) return;
    addChatAttachments(chatFileInput.files);
    chatFileInput.value = "";
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) uploadFiles();
  });
  uploadDropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadDropzone.classList.add("dragover");
  });
  uploadDropzone.addEventListener("dragleave", () => uploadDropzone.classList.remove("dragover"));
  uploadDropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadDropzone.classList.remove("dragover");
    if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length) {
      handleSelectedFiles(event.dataTransfer.files);
    }
  });

  sendBtn.addEventListener("click", sendMessage);
  attachPdfBtn.addEventListener("click", () => chatFileInput.click());
  closePreviewBtn.addEventListener("click", () => {
    previewOpen = false;
    syncPreviewPanel();
  });
  clearInputBtn.addEventListener("click", () => {
    messageInput.value = "";
    localStorage.removeItem(STORAGE_KEYS.draftMessage);
    messageInput.focus();
  });
  document.getElementById("refreshLibraryBtn").addEventListener("click", loadLibrary);
  document.getElementById("reindexBtn").addEventListener("click", reindexAll);
  document.getElementById("newChatBtn").addEventListener("click", startNewSession);
  document.getElementById("refreshSessionsBtn").addEventListener("click", loadSessions);
  document.getElementById("pdfCloseBtn").addEventListener("click", closePdf);
  document.getElementById("pdfDownloadBtn").addEventListener("click", downloadPdf);
  pdfModal.addEventListener("click", (event) => {
    if (event.target === pdfModal) closePdf();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && pdfModal.classList.contains("open")) closePdf();
    if (event.key === "Escape" && versionDropdown.classList.contains("open")) closeVersionDropdown();
  });
  const versionDropdown = document.getElementById("versionDropdown");
  const versionToggle = document.getElementById("versionToggle");
  const versionToggleLabel = document.getElementById("versionToggleLabel");

  function closeVersionDropdown() {
    versionDropdown.classList.remove("open");
    versionToggle.setAttribute("aria-expanded", "false");
  }

  versionToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    const isOpen = versionDropdown.classList.contains("open");
    if (isOpen) {
      closeVersionDropdown();
    } else {
      versionDropdown.classList.add("open");
      versionToggle.setAttribute("aria-expanded", "true");
    }
  });

  document.querySelectorAll(".version-menu-item").forEach((item) => {
    item.addEventListener("click", () => {
      document.querySelectorAll(".version-menu-item").forEach((b) => b.classList.remove("active"));
      item.classList.add("active");
      const version = item.dataset.version || null;
      selectedVersion = version;
      const label = version ? version : "전체";
      versionToggleLabel.textContent = label;
      if (version) {
        versionToggle.classList.add("active");
      } else {
        versionToggle.classList.remove("active");
      }
      closeVersionDropdown();
    });
  });

  document.addEventListener("click", (event) => {
    if (versionDropdown && !versionDropdown.contains(event.target)) {
      closeVersionDropdown();
    }
  });
}

async function initialize() {
  resetChatSurface();
  syncPreviewPanel();
  messageInput.value = localStorage.getItem(STORAGE_KEYS.draftMessage) || "";

  const savedSessionId = localStorage.getItem(STORAGE_KEYS.activeSessionId) || "";
  if (savedSessionId) {
    activeSessionId = savedSessionId;
  } else {
    activeSessionId = generateSessionId();
    saveActiveSessionId();
  }

  await loadSessions();
  await loadLibrary();

  let loadedExistingSession = false;
  if (activeSessionId) {
    try {
      const session = await loadSession(activeSessionId);
      loadedExistingSession = Array.isArray(session.turns) && session.turns.length > 0;
    } catch (error) {
      resetChatSurface();
    }
  }

  if (!loadedExistingSession) {
    resetChatSurface();
    renderSessionSelection();
  }

  const pendingState = readPendingChatState();
  if (pendingState && pendingState.session_id === activeSessionId) {
    await retryPendingChat(pendingState);
  }
}

bindEventListeners();
initialize();
