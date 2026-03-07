/* ═══════════════════════════════════════════════════════════════════════════
   Email Agent — Frontend Application
   Single-page JS app: state → render → event handlers → API calls
   ═══════════════════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  theme: localStorage.getItem("ea-theme") || "light",
  leftTab: "drafting",
  rightTab: "inbox",

  emails: [],
  totalEmails: 0,
  currentPage: 1,
  totalPages: 1,
  lastFetchTs: null,

  selectedId: null,
  selectedEmail: null, // full detail
  isLoadingDetail: false,

  searchQuery: "",
  activeFilters: [], // category slugs to filter

  categories: [],
  todos: [],
  contacts: [],

  draft: null, // generated draft text
  isDrafting: false,
  customInstruction: "",
  showCustomInput: false,

  isFetching: false,
  isLoadingEmails: false,
  llmProvider: "qwen_local_3b",

  // snooze
  snoozeTarget: null, // email id
};

// ── API Client ────────────────────────────────────────────────────────────
const api = {
  async get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    return res.json();
  },
  async post(url, body = {}) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  },
  async put(url, body = {}) {
    const res = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  },
  async del(url) {
    const res = await fetch(url, { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  },
};

// ── Utilities ─────────────────────────────────────────────────────────────
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s || "";
  return d.innerHTML;
}

function debounce(fn, ms) {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

function timeAgo(ts) {
  if (!ts) return "";
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function senderName(raw) {
  if (!raw) return "";
  const m = raw.match(/^([^<]+)/);
  let name = m ? m[1].trim() : raw;
  if (!name) name = raw.split("@")[0] || raw;
  return name;
}

function senderEmail(raw) {
  if (!raw) return raw;
  const m = raw.match(/<([^>]+)>/);
  return m ? m[1].trim() : raw.trim();
}

function simpleDate(raw) {
  if (!raw) return "";
  const m = raw.match(/^([A-Za-z]{3},?\s*\d{1,2}\s+[A-Za-z]{3}\s+\d{4})/);
  return m ? m[1] : raw.split(" ")[0] || "";
}

function toast(message, type = "info") {
  const c = document.getElementById("toast-container");
  const icons = { success: "check_circle", error: "error", info: "info" };
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span class="material-symbols-rounded">${icons[type] || "info"}</span><span>${esc(message)}</span>`;
  c.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateY(8px)";
    setTimeout(() => el.remove(), 200);
  }, 3500);
}

function getCatColor(slug) {
  const cat = state.categories.find((c) => c.slug === slug);
  if (cat) return cat.color;
  const defaults = {
    important: "#ef4444",
    informational: "#3b82f6",
    newsletter: "#8b5cf6",
    "action-required": "#f59e0b",
  };
  return defaults[slug] || "#94a3b8";
}

function getCatName(slug) {
  const cat = state.categories.find((c) => c.slug === slug);
  return cat ? cat.display_name : slug;
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  THEME
// ═══════════════════════════════════════════════════════════════════════════

function applyTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
  const icon = document.getElementById("theme-icon");
  if (icon)
    icon.textContent = state.theme === "dark" ? "light_mode" : "dark_mode";
}

function toggleTheme() {
  state.theme = state.theme === "dark" ? "light" : "dark";
  localStorage.setItem("ea-theme", state.theme);
  applyTheme();
  // Re-render email body iframe if showing detail
  if (state.selectedEmail) renderLeftContent();
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  DATA LOADING
// ═══════════════════════════════════════════════════════════════════════════

async function loadEmails() {
  state.isLoadingEmails = true;
  renderRightContent();
  try {
    const params = new URLSearchParams();
    params.set("page", state.currentPage);
    params.set("per_page", "25");
    if (state.searchQuery) params.set("search", state.searchQuery);
    if (state.activeFilters.length)
      params.set("category", state.activeFilters.join(","));

    const data = await api.get(`/api/emails?${params}`);
    state.emails = data.emails;
    state.totalEmails = data.total;
    state.totalPages = data.total_pages;
    state.lastFetchTs = data.last_fetch_ts;
    updateFetchInfo();
  } catch (e) {
    toast("Failed to load emails: " + e.message, "error");
  } finally {
    state.isLoadingEmails = false;
    renderRightContent();
  }
}

async function loadCategories() {
  try {
    const data = await api.get("/api/categories");
    state.categories = data.categories || [];
  } catch (e) {
    console.error("loadCategories failed:", e);
  }
}

async function loadTodos() {
  try {
    const data = await api.get("/api/todos");
    state.todos = data.todos || [];
  } catch (e) {
    console.error("loadTodos failed:", e);
  }
}

async function loadContacts() {
  try {
    const data = await api.get("/api/contacts?limit=200");
    state.contacts = data.contacts || [];
  } catch (e) {
    console.error("loadContacts failed:", e);
  }
}

async function selectEmail(id) {
  if (state.selectedId === id && state.selectedEmail) return;
  state.selectedId = id;
  state.selectedEmail = null;
  state.draft = null;
  state.showCustomInput = false;
  state.customInstruction = "";
  state.isLoadingDetail = true;
  state.leftTab = "drafting";
  renderLeftTabs();
  renderLeftContent();
  renderRightContent(); // highlight selected
  try {
    const data = await api.get(`/api/emails/${id}`);
    state.selectedEmail = data;
    // Update read status in list
    const listItem = state.emails.find((e) => e.id === id);
    if (listItem) listItem.is_read = true;
  } catch (e) {
    toast("Failed to load email: " + e.message, "error");
  } finally {
    state.isLoadingDetail = false;
    renderLeftContent();
    renderRightContent();
  }
}

async function fetchNewEmails(retrain = false) {
  if (state.isFetching) return;
  state.isFetching = true;
  const btn = document.getElementById("btn-refresh");
  if (btn) btn.classList.add("spinning");
  renderRightContent();
  try {
    const data = await api.post("/api/emails/fetch", { retrain });
    state.lastFetchTs = data.last_fetch_ts;
    state.currentPage = 1;
    await Promise.all([loadEmails(), loadCategories()]);
    updateFetchInfo();
    toast(`Fetched ${data.count} emails`, "success");
  } catch (e) {
    toast("Fetch failed: " + e.message, "error");
  } finally {
    state.isFetching = false;
    if (btn) btn.classList.remove("spinning");
  }
}

function updateFetchInfo() {
  const el = document.getElementById("fetch-info");
  if (el) {
    el.textContent = state.lastFetchTs
      ? `fetched ${timeAgo(state.lastFetchTs)}`
      : "";
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  RENDER — Tabs
// ═══════════════════════════════════════════════════════════════════════════

function renderLeftTabs() {
  document.querySelectorAll("#left-tabs .tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === state.leftTab);
  });
}

function renderRightTabs() {
  document.querySelectorAll("#right-tabs .tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === state.rightTab);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  RENDER — Left panel content
// ═══════════════════════════════════════════════════════════════════════════

function renderLeftContent() {
  const el = document.getElementById("left-content");
  if (state.leftTab === "drafting") {
    renderDrafting(el);
  } else if (state.leftTab === "todo") {
    renderTodo(el);
  } else if (state.leftTab === "categories") {
    renderCategories(el);
  }
}

function renderDrafting(el) {
  if (state.isLoadingDetail) {
    el.innerHTML =
      '<div class="loading-center"><span class="spinner"></span>Loading email…</div>';
    return;
  }

  const email = state.selectedEmail;
  if (!email) {
    el.innerHTML = `
      <div class="empty-state fade-in">
        <span class="material-symbols-rounded">drafts</span>
        <p>Select an email from the inbox to view details and draft replies</p>
      </div>`;
    return;
  }

  // Build quick actions
  const opts = email.decision_options || [];
  const replies = opts.filter((o) => o.type === "reply");
  const todos = opts.filter((o) => o.type === "todo");
  const meetings = opts.filter((o) => o.type === "meeting");

  const catTags = (email.category || "").split(",").filter(Boolean);
  const badgesHtml = catTags
    .map((t) => {
      const c = getCatColor(t.trim());
      return `<span class="badge" style="color:${c};border-color:${c}40;background:${c}18">${esc(getCatName(t.trim()))}</span>`;
    })
    .join("");

  let actionsHtml = "";
  if (replies.length) {
    actionsHtml += `<div class="actions-section"><div class="actions-label">Replies</div><div class="actions-grid">${replies
      .map(
        (r, i) =>
          `<button class="action-btn reply" data-action-type="reply" data-action-idx="${i}" title="${esc(r.context || "")}">${esc(r.label)}</button>`,
      )
      .join("")}</div></div>`;
  }
  if (todos.length) {
    actionsHtml += `<div class="actions-section"><div class="actions-label">Tasks</div><div class="actions-grid">${todos
      .map(
        (t, i) =>
          `<button class="action-btn todo" data-action-type="todo" data-action-idx="${i}" title="${esc(t.context || "")}">${esc(t.label)}</button>`,
      )
      .join("")}</div></div>`;
  }
  if (meetings.length) {
    actionsHtml += `<div class="actions-section"><div class="actions-label">Meeting</div><div class="actions-grid">${meetings
      .map(
        (m, i) =>
          `<button class="action-btn meeting" data-action-type="meeting" data-action-idx="${i}" title="${esc(m.context || "")}">${esc(m.label)}</button>`,
      )
      .join("")}</div></div>`;
  }
  if (!opts.length && email.no_action_message) {
    actionsHtml = `<div class="text-muted text-sm mb-12">${esc(email.no_action_message)}</div>`;
  }

  // Draft section
  let draftHtml = "";
  if (state.isDrafting) {
    draftHtml =
      '<div class="draft-section"><div class="loading-center"><span class="spinner"></span>Generating draft…</div></div>';
  } else if (state.draft) {
    draftHtml = `
      <div class="draft-section fade-in">
        <h4>Draft Reply</h4>
        <textarea class="draft-preview" id="draft-textarea">${esc(state.draft)}</textarea>
        <div class="draft-actions">
          <button class="btn btn-primary" id="btn-send-draft"><span class="material-symbols-rounded">send</span>Send</button>
          <button class="btn btn-ghost" id="btn-discard-draft">Discard</button>
        </div>
      </div>`;
  }

  // Custom instruction
  let customHtml = "";
  if (state.showCustomInput) {
    customHtml = `
      <div class="field mt-12 fade-in">
        <label class="field-label">Custom instruction</label>
        <textarea class="textarea" id="custom-instruction-input" rows="2" placeholder="e.g. Politely decline the invitation…">${esc(state.customInstruction)}</textarea>
        <div class="draft-actions mt-8">
          <button class="btn btn-primary btn-sm" id="btn-custom-draft"><span class="material-symbols-rounded">edit_note</span>Generate</button>
          <button class="btn btn-ghost btn-sm" id="btn-cancel-custom">Cancel</button>
        </div>
      </div>`;
  }

  el.innerHTML = `
    <div class="fade-in">
      <div class="email-detail-header">
        <div class="email-detail-subject">${esc(email.subject)}</div>
        <div class="email-detail-meta">
          <span>${esc(senderName(email.sender))}</span>
          <span style="color:var(--text-muted)">·</span>
          <span style="color:var(--text-muted)">${esc(simpleDate(email.date))}</span>
          ${badgesHtml}
        </div>
      </div>
      <div class="email-body-container" id="email-body-container"></div>
      ${actionsHtml}
      <div class="action-bar">
        <button class="btn btn-secondary btn-sm" id="btn-custom-action"><span class="material-symbols-rounded">edit</span>Custom</button>
        <button class="btn btn-secondary btn-sm" id="btn-forward"><span class="material-symbols-rounded">forward</span>Forward</button>
        <button class="btn btn-secondary btn-sm" id="btn-snooze" style="position:relative"><span class="material-symbols-rounded">snooze</span>Snooze</button>
        <button class="btn btn-danger btn-sm" id="btn-archive"><span class="material-symbols-rounded">archive</span>Archive</button>
      </div>
      ${customHtml}
      ${draftHtml}
    </div>`;

  // Render email body in iframe
  renderEmailBody(email);
}

function renderEmailBody(email) {
  const container = document.getElementById("email-body-container");
  if (!container) return;

  const bodyHtml = email.body_html;
  const plainText = email.body || email.snippet || "";

  if (bodyHtml) {
    const iframe = document.createElement("iframe");
    iframe.sandbox = "allow-same-origin";
    iframe.style.width = "100%";
    iframe.style.display = "block";
    iframe.style.border = "none";
    container.appendChild(iframe);

    const isDark = state.theme === "dark";
    const bgColor = isDark ? "#1e293b" : "#ffffff";
    const textColor = isDark ? "#e2e8f0" : "#1a1a1a";
    const linkColor = isDark ? "#818cf8" : "#6366f1";

    const styleOverride = `<style>
      body { background:${bgColor}!important; color:${textColor}!important; font-family:'Inter',Arial,sans-serif; font-size:14px; line-height:1.6; margin:0; padding:16px; }
      * { color:${textColor}!important; }
      a { color:${linkColor}!important; }
      img { max-width:100%; height:auto; }
      table { max-width:100%!important; }
    </style>`;

    iframe.srcdoc = styleOverride + bodyHtml;
    iframe.onload = () => {
      try {
        const h = iframe.contentDocument.body.scrollHeight;
        iframe.style.height = Math.min(h + 32, 500) + "px";
      } catch (e) {
        iframe.style.height = "300px";
      }
    };
  } else if (plainText) {
    container.innerHTML = `<div class="email-body-plain">${esc(plainText)}</div>`;
  } else {
    container.innerHTML =
      '<div class="email-body-plain text-muted">No content</div>';
  }
}

function renderTodo(el) {
  if (!state.todos.length) {
    el.innerHTML = `
      <div class="empty-state fade-in">
        <span class="material-symbols-rounded">task_alt</span>
        <p>No tasks yet. Quick actions on emails will create todo items here.</p>
      </div>`;
    return;
  }

  el.innerHTML = `<div class="fade-in">${state.todos
    .map(
      (t) => `
    <div class="todo-item">
      <div style="flex:1">
        <div class="todo-text">${esc(t.task)}</div>
        ${t.source_subject ? `<div class="todo-source">from <strong>${esc((t.source_subject || "").slice(0, 55))}</strong> · ${esc(senderName(t.source_sender || ""))}</div>` : ""}
      </div>
      <button class="todo-check" data-todo-id="${t.id}" title="Mark done">
        <span class="material-symbols-rounded">check_circle</span>
      </button>
    </div>
  `,
    )
    .join("")}</div>`;
}

function renderCategories(el) {
  const cats = state.categories;

  // Palette for new label color picker
  const palette = [
    "#38bdf8",
    "#34d399",
    "#a78bfa",
    "#fb923c",
    "#f472b6",
    "#22d3ee",
    "#4ade80",
    "#c084fc",
    "#f87171",
    "#60a5fa",
    "#2dd4bf",
    "#e879f9",
  ];

  el.innerHTML = `
    <div class="fade-in">
      <div class="create-category">
        <div class="field">
          <label class="field-label">New category</label>
          <input class="input" id="new-cat-name" placeholder="Category name" maxlength="40">
        </div>
        <div class="field" style="flex:0 0 auto">
          <label class="field-label">Color</label>
          <input type="color" id="new-cat-color" value="${palette[Math.floor(Math.random() * palette.length)]}" style="width:36px;height:36px;border:none;cursor:pointer;background:none;">
        </div>
        <button class="btn btn-primary btn-sm" id="btn-create-cat" style="margin-bottom:0;align-self:flex-end;height:36px">Create</button>
      </div>
      <div class="category-list">
        ${cats
          .map(
            (c) => `
          <div class="category-item" data-cat-slug="${esc(c.slug)}">
            <span class="category-dot" style="background:${esc(c.color)}"></span>
            <div class="category-info">
              <div class="category-name">${esc(c.display_name)}</div>
              ${c.description ? `<div class="category-desc">${esc(c.description)}</div>` : ""}
            </div>
            <span class="category-count">${c.count || 0}</span>
            <div class="category-actions">
              <button class="icon-btn" data-cat-delete="${esc(c.slug)}" title="Delete">
                <span class="material-symbols-rounded" style="font-size:16px">delete</span>
              </button>
            </div>
          </div>
        `,
          )
          .join("")}
      </div>
    </div>`;
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  RENDER — Right panel content
// ═══════════════════════════════════════════════════════════════════════════

function renderRightContent() {
  const el = document.getElementById("right-content");
  if (state.rightTab === "inbox") {
    renderInbox(el);
  } else if (state.rightTab === "compose") {
    renderCompose(el);
  }
}

function renderInbox(el) {
  // Filter chips
  const enabledCats = state.categories.filter((c) => c.enabled !== 0);
  const filterHtml = enabledCats
    .map((c) => {
      const active = state.activeFilters.includes(c.slug);
      return `<button class="filter-chip ${active ? "active" : ""}" data-filter="${esc(c.slug)}" style="${active ? `border-color:${c.color};color:${c.color}` : ""}">
      ${esc(c.display_name)}<span class="chip-count">${c.count || 0}</span>
    </button>`;
    })
    .join("");

  // Loading
  if (state.isLoadingEmails && !state.emails.length) {
    el.innerHTML = `
      <div class="inbox-header"><h3>Inbox</h3></div>
      ${filterHtml ? `<div class="filter-bar">${filterHtml}</div>` : ""}
      <div class="loading-center"><span class="spinner spinner-lg"></span>Loading emails…</div>`;
    return;
  }

  // Email cards
  const cardsHtml = state.emails
    .map((e) => {
      const isSelected = e.id === state.selectedId;
      const isUnread = !e.is_read;
      const catTags = (e.category || "").split(",").filter(Boolean);
      const badges = catTags
        .map((t) => {
          const c = getCatColor(t.trim());
          return `<span class="badge" style="color:${c};border-color:${c}40;background:${c}18;font-size:10px;padding:1px 6px;">${esc(getCatName(t.trim()))}</span>`;
        })
        .join("");

      return `
      <div class="email-card ${isSelected ? "selected" : ""} ${isUnread ? "unread" : ""}" data-email-id="${esc(e.id)}">
        <div class="email-top-row">
          <div class="email-subject">${esc(e.subject || "(No subject)")}</div>
          <div class="email-badges">${badges}</div>
        </div>
        <div class="email-meta">
          <span class="email-sender-name">${esc(senderName(e.sender))}</span>
          <span>·</span>
          <span>${esc(simpleDate(e.date))}</span>
          ${e.urgent ? '<span class="material-symbols-rounded" style="font-size:14px;color:var(--error)" title="Urgent">priority_high</span>' : ""}
        </div>
        <div class="email-summary">${esc(e.summary || e.snippet || "")}</div>
      </div>`;
    })
    .join("");

  // Pagination
  let pagHtml = "";
  if (state.totalPages > 1) {
    pagHtml = `
      <div class="pagination">
        <button ${state.currentPage <= 1 ? "disabled" : ""} data-page="${state.currentPage - 1}">← Prev</button>
        <span>Page ${state.currentPage} of ${state.totalPages} · ${state.totalEmails} emails</span>
        <button ${state.currentPage >= state.totalPages ? "disabled" : ""} data-page="${state.currentPage + 1}">Next →</button>
      </div>`;
  }

  el.innerHTML = `
    <div class="fade-in">
      <div class="inbox-header">
        <h3>Inbox</h3>
        <span class="inbox-count">${state.totalEmails} email${state.totalEmails !== 1 ? "s" : ""}</span>
      </div>
      ${filterHtml ? `<div class="filter-bar">${filterHtml}</div>` : ""}
      ${cardsHtml || '<div class="empty-state"><span class="material-symbols-rounded">inbox</span><p>No emails found</p></div>'}
      ${pagHtml}
    </div>`;
}

function renderCompose(el) {
  el.innerHTML = `
    <div class="compose-form fade-in">
      <h3 style="margin-bottom:16px">Compose</h3>
      <div class="field autocomplete-container">
        <label class="field-label">To</label>
        <input class="input" id="compose-to" placeholder="Start typing a name or email…" autocomplete="off">
        <div id="compose-autocomplete" class="autocomplete-dropdown hidden"></div>
      </div>
      <div class="field">
        <label class="field-label">Subject</label>
        <input class="input" id="compose-subject" placeholder="Subject">
      </div>
      <div class="field">
        <label class="field-label">What do you want to say?</label>
        <textarea class="textarea" id="compose-context" rows="3" placeholder="e.g. Schedule a meeting for next week, or type a full draft"></textarea>
      </div>
      <div style="display:flex;gap:8px">
        <button class="btn btn-primary" id="btn-compose-draft"><span class="material-symbols-rounded">edit_note</span>Generate draft</button>
        <button class="btn btn-ghost" id="btn-compose-clear">Clear</button>
      </div>
      <div id="compose-draft-area"></div>
    </div>`;

  // Setup autocomplete
  setupComposeAutocomplete();
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  COMPOSE — Autocomplete
// ═══════════════════════════════════════════════════════════════════════════

function setupComposeAutocomplete() {
  const input = document.getElementById("compose-to");
  const dropdown = document.getElementById("compose-autocomplete");
  if (!input || !dropdown) return;

  input.addEventListener(
    "input",
    debounce(() => {
      const q = input.value.trim().toLowerCase();
      if (!q || q.includes("@")) {
        dropdown.classList.add("hidden");
        return;
      }

      const matches = state.contacts
        .filter(
          (c) =>
            (c.name || "").toLowerCase().includes(q) ||
            (c.email || "").toLowerCase().includes(q),
        )
        .slice(0, 6);

      if (!matches.length) {
        dropdown.classList.add("hidden");
        return;
      }

      dropdown.innerHTML = matches
        .map(
          (c) => `
      <div class="autocomplete-item" data-email="${esc(c.email)}">
        <div>${esc(c.name || c.email)}</div>
        <div class="autocomplete-item-email">${esc(c.email)}</div>
      </div>
    `,
        )
        .join("");
      dropdown.classList.remove("hidden");
    }, 200),
  );

  dropdown.addEventListener("click", (e) => {
    const item = e.target.closest(".autocomplete-item");
    if (!item) return;
    input.value = item.dataset.email;
    dropdown.classList.add("hidden");
  });

  // Close on outside click
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".autocomplete-container"))
      dropdown.classList.add("hidden");
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  EVENT HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

// ── Tab clicks ────────────────────────────────────────────────────────────
document.getElementById("left-tabs").addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (!tab) return;
  state.leftTab = tab.dataset.tab;
  renderLeftTabs();
  renderLeftContent();
  if (state.leftTab === "todo") loadTodos().then(() => renderLeftContent());
  if (state.leftTab === "categories")
    loadCategories().then(() => renderLeftContent());
});

document.getElementById("right-tabs").addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (!tab) return;
  state.rightTab = tab.dataset.tab;
  renderRightTabs();
  renderRightContent();
});

// ── Header buttons ────────────────────────────────────────────────────────
document
  .getElementById("btn-refresh")
  .addEventListener("click", () => fetchNewEmails());
document.getElementById("btn-theme").addEventListener("click", toggleTheme);
document.getElementById("btn-auth").addEventListener("click", handleAuth);

document.getElementById("llm-select").addEventListener("change", async (e) => {
  try {
    await api.put("/api/settings/llm", { provider: e.target.value });
    state.llmProvider = e.target.value;
    toast("LLM provider updated", "success");
  } catch (err) {
    toast(err.message, "error");
  }
});

// ── Search ────────────────────────────────────────────────────────────────
const searchInput = document.getElementById("search-input");
const searchClear = document.getElementById("search-clear");

searchInput.addEventListener(
  "input",
  debounce(() => {
    state.searchQuery = searchInput.value.trim();
    state.currentPage = 1;
    searchClear.classList.toggle("hidden", !state.searchQuery);
    loadEmails();
  }, 300),
);

searchClear.addEventListener("click", () => {
  searchInput.value = "";
  state.searchQuery = "";
  state.currentPage = 1;
  searchClear.classList.add("hidden");
  loadEmails();
});

// ── Right panel delegated clicks ──────────────────────────────────────────
document.getElementById("right-content").addEventListener("click", (e) => {
  // Email card click
  const card = e.target.closest(".email-card");
  if (card) {
    selectEmail(card.dataset.emailId);
    return;
  }

  // Filter chip click
  const chip = e.target.closest(".filter-chip");
  if (chip) {
    const slug = chip.dataset.filter;
    const idx = state.activeFilters.indexOf(slug);
    if (idx >= 0) state.activeFilters.splice(idx, 1);
    else state.activeFilters.push(slug);
    state.currentPage = 1;
    loadEmails();
    return;
  }

  // Pagination
  const pageBtn = e.target.closest(".pagination button");
  if (pageBtn && !pageBtn.disabled) {
    state.currentPage = parseInt(pageBtn.dataset.page);
    loadEmails();
    return;
  }

  // Compose: Generate draft
  if (e.target.closest("#btn-compose-draft")) {
    handleComposeDraft();
    return;
  }

  // Compose: Clear
  if (e.target.closest("#btn-compose-clear")) {
    renderCompose(document.getElementById("right-content"));
    return;
  }

  // Compose: Send
  if (e.target.closest("#btn-compose-send")) {
    handleComposeSend();
    return;
  }
});

// ── Left panel delegated clicks ───────────────────────────────────────────
document.getElementById("left-content").addEventListener("click", async (e) => {
  // Quick action buttons
  const actionBtn = e.target.closest(".action-btn");
  if (actionBtn && state.selectedEmail) {
    const type = actionBtn.dataset.actionType;
    const idx = parseInt(actionBtn.dataset.actionIdx);
    const opts = state.selectedEmail.decision_options || [];
    const typeOpts = opts.filter((o) => o.type === type);
    const option = typeOpts[idx];
    if (!option) return;

    if (type === "reply" || type === "meeting") {
      handleDraft(option.context || option.label);
    } else if (type === "todo") {
      try {
        await api.post("/api/todos", {
          task: option.label,
          email_id: state.selectedId,
        });
        toast("Task added", "success");
        loadTodos();
      } catch (err) {
        toast(err.message, "error");
      }
    }
    return;
  }

  // Custom instruction button
  if (e.target.closest("#btn-custom-action")) {
    state.showCustomInput = true;
    renderLeftContent();
    setTimeout(() => {
      const inp = document.getElementById("custom-instruction-input");
      if (inp) inp.focus();
    }, 50);
    return;
  }

  // Custom draft generate
  if (e.target.closest("#btn-custom-draft")) {
    const inp = document.getElementById("custom-instruction-input");
    const instruction = inp ? inp.value.trim() : "";
    if (!instruction) {
      toast("Enter an instruction", "error");
      return;
    }
    state.showCustomInput = false;
    handleDraft(instruction);
    return;
  }

  // Cancel custom
  if (e.target.closest("#btn-cancel-custom")) {
    state.showCustomInput = false;
    renderLeftContent();
    return;
  }

  // Forward
  if (e.target.closest("#btn-forward") && state.selectedEmail) {
    const email = state.selectedEmail;
    state.rightTab = "compose";
    renderRightTabs();
    renderRightContent();
    // Pre-fill compose fields after render
    setTimeout(() => {
      const toEl = document.getElementById("compose-to");
      const subjEl = document.getElementById("compose-subject");
      const ctxEl = document.getElementById("compose-context");
      if (subjEl) {
        let subj = email.subject || "";
        if (!subj.toLowerCase().startsWith("fwd:")) subj = "Fwd: " + subj;
        subjEl.value = subj;
      }
      if (ctxEl) {
        ctxEl.value = `---------- Forwarded message ----------\nFrom: ${email.sender}\nDate: ${email.date}\nSubject: ${email.subject}\n\n${(email.body || "").slice(0, 2000)}`;
      }
    }, 50);
    return;
  }

  // Archive
  if (e.target.closest("#btn-archive") && state.selectedId) {
    try {
      await api.post(`/api/emails/${state.selectedId}/archive`);
      toast("Archived", "success");
      state.emails = state.emails.filter((e) => e.id !== state.selectedId);
      state.selectedId = null;
      state.selectedEmail = null;
      renderLeftContent();
      renderRightContent();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }

  // Snooze
  if (e.target.closest("#btn-snooze") && state.selectedId) {
    showSnoozeDropdown();
    return;
  }

  // Send draft
  if (e.target.closest("#btn-send-draft") && state.selectedId) {
    handleSendDraft();
    return;
  }

  // Discard draft
  if (e.target.closest("#btn-discard-draft")) {
    state.draft = null;
    renderLeftContent();
    return;
  }

  // Todo complete
  const todoCheck = e.target.closest(".todo-check");
  if (todoCheck) {
    const id = parseInt(todoCheck.dataset.todoId);
    try {
      await api.post(`/api/todos/${id}/done`);
      toast("Task completed", "success");
      state.todos = state.todos.filter((t) => t.id !== id);
      renderLeftContent();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }

  // Category delete
  const catDel = e.target.closest("[data-cat-delete]");
  if (catDel) {
    const slug = catDel.dataset.catDelete;
    if (!confirm(`Delete category "${getCatName(slug)}"?`)) return;
    try {
      await api.del(`/api/categories/${slug}`);
      toast("Category deleted", "success");
      await loadCategories();
      renderLeftContent();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }

  // Create category
  if (e.target.closest("#btn-create-cat")) {
    const nameEl = document.getElementById("new-cat-name");
    const colorEl = document.getElementById("new-cat-color");
    const name = nameEl ? nameEl.value.trim() : "";
    const color = colorEl ? colorEl.value : "#94a3b8";
    if (!name) {
      toast("Enter a category name", "error");
      return;
    }
    try {
      await api.post("/api/categories", { name, color });
      toast("Category created", "success");
      await loadCategories();
      renderLeftContent();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// ██  ACTION HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

async function handleDraft(decision) {
  if (!state.selectedId || state.isDrafting) return;
  state.isDrafting = true;
  state.draft = null;
  renderLeftContent();
  try {
    const data = await api.post(`/api/emails/${state.selectedId}/draft`, {
      decision,
    });
    state.draft = data.draft;
  } catch (e) {
    toast("Draft failed: " + e.message, "error");
  } finally {
    state.isDrafting = false;
    renderLeftContent();
  }
}

async function handleSendDraft() {
  const textarea = document.getElementById("draft-textarea");
  const body = textarea ? textarea.value.trim() : state.draft;
  if (!body || !state.selectedId) return;

  try {
    await api.post(`/api/emails/${state.selectedId}/send`, { body });
    toast("Reply sent!", "success");
    state.draft = null;
    renderLeftContent();
  } catch (e) {
    toast("Send failed: " + e.message, "error");
  }
}

async function handleComposeDraft() {
  const to = (document.getElementById("compose-to") || {}).value || "";
  const subject =
    (document.getElementById("compose-subject") || {}).value || "";
  const context =
    (document.getElementById("compose-context") || {}).value || "";

  if (!to.trim() || !subject.trim() || !context.trim()) {
    toast("Fill in To, Subject, and message", "error");
    return;
  }

  const area = document.getElementById("compose-draft-area");
  area.innerHTML =
    '<div class="loading-center mt-16"><span class="spinner"></span>Generating draft…</div>';

  try {
    const data = await api.post("/api/compose/draft", { to, subject, context });
    area.innerHTML = `
      <div class="draft-section mt-16 fade-in">
        <h4>Draft</h4>
        <textarea class="draft-preview" id="compose-draft-text">${esc(data.draft)}</textarea>
        <div class="draft-actions">
          <button class="btn btn-primary" id="btn-compose-send"><span class="material-symbols-rounded">send</span>Send</button>
        </div>
      </div>`;
  } catch (e) {
    area.innerHTML = "";
    toast("Draft failed: " + e.message, "error");
  }
}

async function handleComposeSend() {
  const to = (document.getElementById("compose-to") || {}).value || "";
  const subject =
    (document.getElementById("compose-subject") || {}).value || "";
  const textarea = document.getElementById("compose-draft-text");
  const body = textarea ? textarea.value.trim() : "";

  if (!to || !subject || !body) {
    toast("Missing fields", "error");
    return;
  }

  try {
    await api.post("/api/compose/send", { to, subject, body });
    toast("Email sent!", "success");
    state.rightTab = "inbox";
    renderRightTabs();
    renderRightContent();
  } catch (e) {
    toast("Send failed: " + e.message, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  Snooze dropdown
// ═══════════════════════════════════════════════════════════════════════════

function showSnoozeDropdown() {
  // Remove existing
  document.querySelectorAll(".snooze-dropdown").forEach((d) => d.remove());

  const btn = document.getElementById("btn-snooze");
  if (!btn) return;

  const dropdown = document.createElement("div");
  dropdown.className = "snooze-dropdown";
  dropdown.style.bottom = "100%";
  dropdown.style.left = "0";
  dropdown.style.marginBottom = "4px";

  const options = [
    { label: "In 1 hour", hours: 1 },
    { label: "In 3 hours", hours: 3 },
    { label: "Tomorrow morning", hours: 16 },
    { label: "Next week", hours: 168 },
  ];

  dropdown.innerHTML = options
    .map(
      (o) =>
        `<button class="snooze-option" data-hours="${o.hours}">${o.label}</button>`,
    )
    .join("");

  btn.appendChild(dropdown);

  dropdown.addEventListener("click", async (e) => {
    const opt = e.target.closest(".snooze-option");
    if (!opt) return;
    const hours = parseInt(opt.dataset.hours);
    const until = new Date(Date.now() + hours * 3600000).toISOString();
    dropdown.remove();
    try {
      await api.post(`/api/emails/${state.selectedId}/snooze`, { until });
      toast("Snoozed", "success");
      state.emails = state.emails.filter((e) => e.id !== state.selectedId);
      state.selectedId = null;
      state.selectedEmail = null;
      renderLeftContent();
      renderRightContent();
    } catch (err) {
      toast(err.message, "error");
    }
  });

  // Close on outside click
  setTimeout(() => {
    const close = (e) => {
      if (!dropdown.contains(e.target) && e.target !== btn) {
        dropdown.remove();
        document.removeEventListener("click", close);
      }
    };
    document.addEventListener("click", close);
  }, 10);
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  KEYBOARD SHORTCUTS
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener("keydown", (e) => {
  // Don't fire when typing in inputs
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;

  if (e.key === "/") {
    e.preventDefault();
    searchInput.focus();
    return;
  }

  if (e.key === "j" || e.key === "k") {
    // Navigate emails
    const ids = state.emails.map((e) => e.id);
    const curIdx = ids.indexOf(state.selectedId);
    let next = e.key === "j" ? curIdx + 1 : curIdx - 1;
    if (next < 0) next = 0;
    if (next >= ids.length) next = ids.length - 1;
    if (ids[next]) selectEmail(ids[next]);
    return;
  }

  if (e.key === "Escape") {
    state.selectedId = null;
    state.selectedEmail = null;
    state.draft = null;
    renderLeftContent();
    renderRightContent();
    return;
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// ██  AUTH
// ═══════════════════════════════════════════════════════════════════════════

function updateAuthButton(signedIn) {
  const btn = document.getElementById("btn-auth");
  const label = document.getElementById("auth-label");
  if (!btn || !label) return;
  if (signedIn) {
    label.textContent = "Sign out";
    btn.title = "Sign out of Google";
    btn.className = "btn btn-sm btn-ghost";
  } else {
    label.textContent = "Sign in";
    btn.title = "Sign in with Google";
    btn.className = "btn btn-sm btn-primary";
  }
  btn.dataset.signedIn = signedIn ? "1" : "0";
}

async function handleAuth() {
  const btn = document.getElementById("btn-auth");
  const signedIn = btn && btn.dataset.signedIn === "1";

  if (signedIn) {
    if (
      !confirm(
        "Sign out of Google? You will need to sign in again to fetch emails.",
      )
    )
      return;
    try {
      await api.post("/api/auth/signout");
      updateAuthButton(false);
      toast("Signed out", "success");
    } catch (e) {
      toast("Sign-out failed: " + e.message, "error");
    }
  } else {
    const label = document.getElementById("auth-label");
    if (label) label.textContent = "Signing in…";
    if (btn) btn.disabled = true;
    try {
      await api.post("/api/auth/signin");
      updateAuthButton(true);
      toast("Signed in successfully", "success");
      await fetchNewEmails();
    } catch (e) {
      updateAuthButton(false);
      toast("Sign-in failed: " + e.message, "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ██  INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

async function init() {
  applyTheme();
  renderLeftTabs();
  renderRightTabs();
  renderLeftContent();

  // Load data in parallel
  await Promise.all([
    loadEmails(),
    loadCategories(),
    loadTodos(),
    loadContacts(),
  ]);

  // Re-render with categories loaded (for filter chips + badge colors)
  renderRightContent();
  updateFetchInfo();

  // Load LLM setting
  try {
    const settings = await api.get("/api/settings");
    state.llmProvider = settings.llm_provider;
    document.getElementById("llm-select").value = settings.llm_provider;
  } catch (e) {
    /* ignore */
  }

  // Load auth status
  try {
    const auth = await api.get("/api/auth/status");
    updateAuthButton(auth.signed_in);
  } catch (e) {
    /* ignore */
  }

  // Periodic fetch-info update
  setInterval(updateFetchInfo, 30000);
}

init();
