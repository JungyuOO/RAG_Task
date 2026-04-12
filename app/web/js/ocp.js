let _ocpResource = "pods";
let _ocpSelectedName = "";
let _ocpFilterText = "";
let _ocpSortMode = "name";
let _ocpLastItems = [];
let _ocpNamespaces = [];

function _toOcpSingular(resource) {
  const mapping = {
    pods: "Pod",
    deployments: "Deployment",
    services: "Service",
    routes: "Route",
    events: "Event",
  };
  return mapping[String(resource || "").trim()] || "Resource";
}

function _composeOcpChatPrompt(resource, name, namespace) {
  const singular = _toOcpSingular(resource);
  const scope = namespace ? `${namespace} namespace에서 ` : "";
  return `${scope}${singular} ${name} 상태 보여줘.`;
}

function _sendOcpPromptToChat(resource, name, namespace) {
  const prompt = _composeOcpChatPrompt(resource, name, namespace);
  switchScreen("chat");
  messageInput.value = prompt;
  localStorage.setItem(STORAGE_KEYS.draftMessage, prompt);
  messageInput.focus();
}

function _normalizeOcpResourceKey(resource) {
  const normalized = String(resource || "").toLowerCase().trim();
  const mapping = {
    pod: "pods",
    pods: "pods",
    deployment: "deployments",
    deployments: "deployments",
    service: "services",
    services: "services",
    route: "routes",
    routes: "routes",
    event: "events",
    events: "events",
  };
  return mapping[normalized] || "";
}

function _yamlScalar(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  const text = String(value);
  if (text === "") return '""';
  if (/[:#\-\n'"\\]/.test(text) || /^\s|\s$/.test(text)) {
    return JSON.stringify(text);
  }
  return text;
}

function _pruneOcpDisplayObject(value, path = []) {
  if (Array.isArray(value)) {
    const items = value
      .map((item, index) => _pruneOcpDisplayObject(item, path.concat(String(index))))
      .filter((item) => item !== undefined);
    return items.length ? items : undefined;
  }

  if (value && typeof value === "object") {
    const result = {};
    for (const [key, item] of Object.entries(value)) {
      if (key === "managedFields") continue;
      if (key === "kubectl.kubernetes.io/last-applied-configuration") continue;
      const next = _pruneOcpDisplayObject(item, path.concat(key));
      if (next === undefined) continue;
      result[key] = next;
    }
    return Object.keys(result).length ? result : undefined;
  }

  if (value === "" || value === null || value === undefined) {
    return undefined;
  }
  return value;
}

function _toYamlLike(value, indent = 0) {
  const pad = "  ".repeat(indent);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    return value.map((item) => {
      if (item && typeof item === "object") {
        return pad + "-\n" + _toYamlLike(item, indent + 1);
      }
      return pad + "- " + _yamlScalar(item);
    }).join("\n");
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) return "{}";
    return entries.map(([key, item]) => {
      if (item && typeof item === "object") {
        return pad + key + ":\n" + _toYamlLike(item, indent + 1);
      }
      return pad + key + ": " + _yamlScalar(item);
    }).join("\n");
  }
  return pad + _yamlScalar(value);
}

function _buildOcpBadge(item) {
  const resource = _ocpResource;
  let label = "";
  let tone = "neutral";

  if (resource === "pods") {
    label = item.phase || "Unknown";
    tone = label === "Running" ? "ok" : (label === "Pending" ? "warn" : "neutral");
  } else if (resource === "deployments") {
    const ready = Number(item.ready_replicas || 0);
    const replicas = Number(item.replicas || 0);
    label = replicas ? `${ready}/${replicas}` : "0/0";
    tone = replicas && ready === replicas ? "ok" : (ready > 0 ? "warn" : "neutral");
  } else if (resource === "services") {
    label = item.type || "Service";
    tone = "info";
  } else if (resource === "routes") {
    label = item.host ? "Exposed" : "Route";
    tone = item.host ? "ok" : "neutral";
  } else if (resource === "events") {
    label = item.type || "Event";
    tone = label === "Warning" ? "warn" : "info";
  }

  return { label, tone };
}

function _buildOcpSummary(items) {
  if (_ocpResource === "pods") {
    const running = items.filter((item) => item.phase === "Running").length;
    const pending = items.filter((item) => item.phase === "Pending").length;
    return [
      { label: "Total", value: String(items.length) },
      { label: "Running", value: String(running) },
      { label: "Pending", value: String(pending) },
    ];
  }
  if (_ocpResource === "deployments") {
    const healthy = items.filter((item) => Number(item.replicas || 0) > 0 && Number(item.ready_replicas || 0) === Number(item.replicas || 0)).length;
    return [
      { label: "Total", value: String(items.length) },
      { label: "Healthy", value: String(healthy) },
      { label: "Scaled", value: String(items.filter((item) => Number(item.replicas || 0) > 0).length) },
    ];
  }
  if (_ocpResource === "services") {
    const clusterIp = items.filter((item) => item.cluster_ip).length;
    return [
      { label: "Total", value: String(items.length) },
      { label: "ClusterIP", value: String(clusterIp) },
      { label: "Types", value: String(new Set(items.map((item) => item.type || "Service")).size) },
    ];
  }
  if (_ocpResource === "events") {
    const warnings = items.filter((item) => item.type === "Warning").length;
    return [
      { label: "Total", value: String(items.length) },
      { label: "Warning", value: String(warnings) },
      { label: "Objects", value: String(new Set(items.map((item) => item.to || "")).size) },
    ];
  }
  const exposed = items.filter((item) => item.host).length;
  return [
    { label: "Total", value: String(items.length) },
    { label: "Exposed", value: String(exposed) },
    { label: "Targets", value: String(new Set(items.map((item) => item.to || "")).size) },
  ];
}

function _renderOcpSummary(items) {
  const summary = document.getElementById("ocpSummaryRow");
  if (!summary) return;
  const stats = _buildOcpSummary(items);
  summary.innerHTML = stats
    .map((stat) => (
      '<div class="ocp-summary-chip">' +
        '<span class="ocp-summary-label">' + escapeHtml(stat.label) + "</span>" +
        '<span class="ocp-summary-value">' + escapeHtml(stat.value) + "</span>" +
      "</div>"
    ))
    .join("");
}

function _renderOcpStatus(data) {
  const body = document.getElementById("ocpStatusBody");
  if (!body) return;
  body.innerHTML =
    '<div class="ocp-status-grid">' +
      '<div class="ocp-status-item"><div class="ocp-status-label">Enabled</div><div class="ocp-status-value">' + (data.enabled ? "Yes" : "No") + "</div></div>" +
      '<div class="ocp-status-item"><div class="ocp-status-label">Namespace</div><div class="ocp-status-value">' + escapeHtml(data.default_namespace || "-") + "</div></div>" +
      '<div class="ocp-status-item" style="grid-column:1 / -1;"><div class="ocp-status-label">API Base URL</div><div class="ocp-status-value">' + escapeHtml(data.base_url || "-") + "</div></div>" +
    "</div>";
}

function _renderOcpNamespaceSuggestions(items) {
  const datalist = document.getElementById("ocpNamespaceSuggestions");
  if (!datalist) return;
  datalist.innerHTML = "";
  items.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    datalist.appendChild(option);
  });
}

function _renderOcpToolbar() {
  const toolbar = document.getElementById("ocpResourceToolbar");
  if (!toolbar) return;
  const resources = [
    ["pods", "Pods"],
    ["deployments", "Deployments"],
    ["services", "Services"],
    ["routes", "Routes"],
    ["events", "Events"],
  ];
  toolbar.innerHTML = "";
  resources.forEach(([key, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ocp-resource-btn" + (_ocpResource === key ? " active" : "");
    button.textContent = label;
    button.addEventListener("click", () => {
      _ocpResource = key;
      _ocpSelectedName = "";
      _renderOcpToolbar();
      loadOcpResources();
    });
    toolbar.appendChild(button);
  });
}

function _filterOcpItems(items) {
  const filter = _ocpFilterText.trim().toLowerCase();
  if (!filter) return items.slice();
  return items.filter((item) => {
    const haystack = [
      item.name || "",
      item.namespace || "",
      item.phase || "",
      item.type || "",
      item.host || "",
      item.node_name || "",
      item.to || "",
    ].join(" ").toLowerCase();
    return haystack.includes(filter);
  });
}

function _sortOcpItems(items) {
  const sorted = items.slice();
  if (_ocpSortMode === "status") {
    sorted.sort((left, right) => {
      const leftBadge = _buildOcpBadge(left).label;
      const rightBadge = _buildOcpBadge(right).label;
      return leftBadge.localeCompare(rightBadge) || String(left.name || "").localeCompare(String(right.name || ""));
    });
    return sorted;
  }
  if (_ocpSortMode === "recent") {
    sorted.sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || "")));
    return sorted;
  }
  sorted.sort((left, right) => String(left.name || "").localeCompare(String(right.name || "")));
  return sorted;
}

function _updateOcpCountLabel(total, visible) {
  const count = document.getElementById("ocpResourceCount");
  if (!count) return;
  count.textContent = visible === total ? `${total} items` : `${visible}/${total} items`;
}

function _renderOcpResources(data) {
  const list = document.getElementById("ocpResourceList");
  if (!list) return;

  _ocpLastItems = Array.isArray(data.items) ? data.items.slice() : [];
  _renderOcpSummary(_ocpLastItems);

  const items = _sortOcpItems(_filterOcpItems(_ocpLastItems));
  _updateOcpCountLabel(_ocpLastItems.length, items.length);

  if (!items.length) {
    list.innerHTML = '<div class="empty">No matching resources.</div>';
    return;
  }

  list.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "ocp-resource-item" + (_ocpSelectedName && _ocpSelectedName === item.name ? " active" : "");

    const badge = _buildOcpBadge(item);
    const meta = [
      item.namespace || "",
      item.node_name || "",
      item.cluster_ip || "",
      item.host || "",
    ].filter(Boolean).join(" · ");

    const extra = [];
    if (item.phase) extra.push("phase: " + item.phase);
    if (item.ready_replicas || item.replicas) {
      extra.push("replicas: " + Number(item.ready_replicas || 0) + "/" + Number(item.replicas || 0));
    }
    if (item.type) extra.push("type: " + item.type);
    if (item.to) extra.push("to: " + item.to);
    if (_ocpResource === "events" && item.host) extra.push("object: " + item.host);

    card.innerHTML =
      '<div class="ocp-resource-row">' +
        '<div class="ocp-resource-name">' + escapeHtml(item.name || "") + "</div>" +
        '<span class="ocp-resource-badge ' + badge.tone + '">' + escapeHtml(badge.label) + "</span>" +
      "</div>" +
      '<div class="ocp-resource-meta">' + escapeHtml(meta || item.kind || "") + "</div>" +
      (extra.length ? '<div class="ocp-resource-extra">' + escapeHtml(extra.join(" · ")) + "</div>" : "");

    card.addEventListener("click", () => {
      _ocpSelectedName = item.name || "";
      _renderOcpResources(data);
      loadOcpYaml(item.name);
    });
    list.appendChild(card);
  });
}

async function loadOcpStatus() {
  const response = await fetch("/api/ocp/status");
  if (!response.ok) throw new Error(await extractErrorMessage(response));
  const data = await response.json();
  _renderOcpStatus(data);
  const namespaceInput = document.getElementById("ocpNamespaceInput");
  if (namespaceInput && !namespaceInput.value) {
    namespaceInput.value = data.default_namespace || "";
  }
  return data;
}

async function loadOcpNamespaces() {
  const response = await fetch("/api/ocp/namespaces");
  if (!response.ok) throw new Error(await extractErrorMessage(response));
  const data = await response.json();
  _ocpNamespaces = Array.isArray(data.items) ? data.items.slice() : [];
  _renderOcpNamespaceSuggestions(_ocpNamespaces);
  return data;
}

async function loadOcpResources() {
  const namespaceInput = document.getElementById("ocpNamespaceInput");
  const namespace = namespaceInput ? namespaceInput.value.trim() : "";
  const list = document.getElementById("ocpResourceList");
  if (list) list.innerHTML = '<div class="empty">Loading resources...</div>';

  const params = new URLSearchParams({ resource: _ocpResource });
  if (namespace) params.set("namespace", namespace);

  const response = await fetch("/api/ocp/resources?" + params.toString());
  if (!response.ok) throw new Error(await extractErrorMessage(response));
  const data = await response.json();
  _renderOcpResources(data);
  return data;
}

async function loadOcpYaml(name) {
  const namespaceInput = document.getElementById("ocpNamespaceInput");
  const namespace = namespaceInput ? namespaceInput.value.trim() : "";
  const frame = document.getElementById("ocpYamlFrame");
  if (!frame) return;

  frame.innerHTML = "<pre>Loading YAML...</pre>";
  const params = new URLSearchParams({ resource: _ocpResource, name });
  if (namespace) params.set("namespace", namespace);

  const response = await fetch("/api/ocp/resource-yaml?" + params.toString());
  if (!response.ok) throw new Error(await extractErrorMessage(response));

  const data = await response.json();
  const displayObject = _pruneOcpDisplayObject(data.object) || {};
  const yamlLike = _toYamlLike(displayObject);
  frame.innerHTML =
    '<div class="ocp-yaml-meta">' +
      '<div class="ocp-yaml-title">' + escapeHtml(data.resource + " · " + data.name) + "</div>" +
      '<div class="ocp-yaml-copy">' + escapeHtml(data.namespace || "") + "</div>" +
      '<div class="ocp-yaml-actions">' +
        '<button type="button" class="secondary mini-button ocp-ask-chat-button">Ask in chat</button>' +
        '<button type="button" class="secondary mini-button ocp-copy-yaml-button">Copy YAML</button>' +
      "</div>" +
    "</div>" +
    "<pre>" + escapeHtml(yamlLike) + "</pre>";

  const askButton = frame.querySelector(".ocp-ask-chat-button");
  const copyButton = frame.querySelector(".ocp-copy-yaml-button");
  if (askButton) {
    askButton.addEventListener("click", () => {
      _sendOcpPromptToChat(data.resource, data.name, data.namespace || "");
    });
  }
  if (copyButton) {
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(yamlLike);
        copyButton.textContent = "Copied";
        setTimeout(() => {
          copyButton.textContent = "Copy YAML";
        }, 1200);
      } catch (_error) {
        copyButton.textContent = "Copy failed";
        setTimeout(() => {
          copyButton.textContent = "Copy YAML";
        }, 1200);
      }
    });
  }
}

function _bindOcpFilterInput() {
  const filterInput = document.getElementById("ocpResourceFilterInput");
  if (!filterInput || filterInput.dataset.bound === "true") return;

  filterInput.dataset.bound = "true";
  filterInput.addEventListener("input", () => {
    _ocpFilterText = filterInput.value || "";
    _renderOcpResources({ items: _ocpLastItems });
  });
}

function _bindOcpSortInput() {
  const sortInput = document.getElementById("ocpSortSelect");
  if (!sortInput || sortInput.dataset.bound === "true") return;

  sortInput.dataset.bound = "true";
  sortInput.addEventListener("change", () => {
    _ocpSortMode = sortInput.value || "name";
    _renderOcpResources({ items: _ocpLastItems });
  });
}

function _bindOcpNamespaceInput() {
  const namespaceInput = document.getElementById("ocpNamespaceInput");
  if (!namespaceInput || namespaceInput.dataset.bound === "true") return;

  namespaceInput.dataset.bound = "true";
  namespaceInput.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    try {
      await loadOcpResources();
    } catch (error) {
      const list = document.getElementById("ocpResourceList");
      if (list) list.innerHTML = '<div class="empty">Failed to load OCP resources. ' + escapeHtml(error.message) + "</div>";
    }
  });
}

async function initializeOcpScreen() {
  if (!document.getElementById("screen-ocp")) return;

  _bindOcpFilterInput();
  _bindOcpSortInput();
  _bindOcpNamespaceInput();
  _renderOcpToolbar();

  const refreshBtn = document.getElementById("ocpRefreshBtn");
  if (refreshBtn && refreshBtn.dataset.bound !== "true") {
    refreshBtn.dataset.bound = "true";
    refreshBtn.addEventListener("click", async () => {
      try {
        await loadOcpStatus();
        await loadOcpResources();
      } catch (error) {
        const list = document.getElementById("ocpResourceList");
        if (list) list.innerHTML = '<div class="empty">Failed to load OCP resources. ' + escapeHtml(error.message) + "</div>";
      }
    });
  }

  try {
    const status = await loadOcpStatus();
    if (status.enabled) {
      await loadOcpNamespaces();
      await loadOcpResources();
    } else {
      const list = document.getElementById("ocpResourceList");
      if (list) list.innerHTML = '<div class="empty">OCP API settings are not configured yet.</div>';
    }
  } catch (error) {
    const list = document.getElementById("ocpResourceList");
    if (list) list.innerHTML = '<div class="empty">Failed to load OCP status. ' + escapeHtml(error.message) + "</div>";
  }
}

async function openOcpExplorer(resource, name = "") {
  const normalized = _normalizeOcpResourceKey(resource);
  if (normalized) {
    _ocpResource = normalized;
    _renderOcpToolbar();
  }

  _ocpSelectedName = String(name || "").trim();
  switchScreen("ocp");

  try {
    await loadOcpStatus();
    await loadOcpNamespaces();
    const data = await loadOcpResources();
    if (_ocpSelectedName && Array.isArray(data.items) && data.items.some((item) => item.name === _ocpSelectedName)) {
      await loadOcpYaml(_ocpSelectedName);
    }
  } catch (error) {
    const list = document.getElementById("ocpResourceList");
    if (list) list.innerHTML = '<div class="empty">Failed to load OCP resources. ' + escapeHtml(error.message) + "</div>";
  }
}
