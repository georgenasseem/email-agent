/* ═══════════════════════════════════════════════════════════════════════
   Dispatch — Frontend Application
   Single-pane navigation: sidebar → list pane → detail pane
   All 10 feedback items incorporated
   ═══════════════════════════════════════════════════════════════════════ */

// ═══════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════
const state = {
  theme: localStorage.getItem("ea-theme") || "light",
  sidebarOpen: false,
  sidebarCollapsed:
    localStorage.getItem("ea-sidebar-collapsed") === null
      ? true
      : localStorage.getItem("ea-sidebar-collapsed") === "1",

  view: "inbox", // 'inbox' | 'todo' | 'meetings' | 'labels'
  detailMode: "empty", // 'email' | 'compose' | 'empty'

  emails: [],
  totalEmails: 0,
  currentPage: 1,
  totalPages: 1,
  lastFetchTs: null,

  selectedId: null,
  selectedEmail: null,
  isLoadingDetail: false,

  searchQuery: "",
  activeFilters: [],

  categories: [],
  todos: [],
  contacts: [],
  meetings: [],
  categorySuggestions: null,

  draft: null,
  isDrafting: false,
  composeContext: null,
  selectedMeeting: null,

  isFetching: false,
  isLoadingEmails: false,
  isRethinking: false,
  llmProvider: "qwen_local_3b",
};

// ═══════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════
const api = {
  async get(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error((await r.json()).error || r.statusText);
    return r.json();
  },
  async post(url, body = {}) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    return d;
  },
  async put(url, body = {}) {
    const r = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    return d;
  },
  async del(url) {
    const r = await fetch(url, { method: "DELETE" });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    return d;
  },
};

// ═══════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════
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
  let n = m ? m[1].trim() : raw;
  if (!n) n = raw.split("@")[0] || raw;
  return n;
}

function senderEmail(raw) {
  if (!raw) return raw;
  const m = raw.match(/<([^>]+)>/);
  return m ? m[1].trim() : raw.trim();
}

function avatarInitials(raw) {
  const n = senderName(raw);
  if (!n) return "?";
  const p = n.trim().split(/\s+/);
  return p.length >= 2
    ? (p[0][0] + p[1][0]).toUpperCase()
    : n.slice(0, 2).toUpperCase();
}

function simpleDate(raw) {
  if (!raw) return "";
  try {
    const d = new Date(raw);
    if (isNaN(d.getTime())) {
      const m = raw.match(/^([A-Za-z]{3},?\s*\d{1,2}\s+[A-Za-z]{3}\s+\d{4})/);
      return m ? m[1] : raw.split(" ")[0] || "";
    }
    const h = d.getHours();
    const min = d.getMinutes().toString().padStart(2, "0");
    const ampm = h >= 12 ? "pm" : "am";
    const h12 = h % 12 || 12;
    const months = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];
    return `${h12}:${min}${ampm} ${d.getDate()}, ${months[d.getMonth()]}`;
  } catch (_) {
    return raw.split(" ")[0] || "";
  }
}

const avatarPalette = [
  "#4285f4",
  "#34a853",
  "#ea4335",
  "#fbbc04",
  "#7b1fa2",
  "#0097a7",
  "#e64a19",
  "#5c6bc0",
  "#00897b",
  "#c2185b",
  "#1e88e5",
  "#43a047",
  "#f4511e",
  "#3949ab",
];
function avatarColor(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = str.charCodeAt(i) + ((h << 5) - h);
  return avatarPalette[Math.abs(h) % avatarPalette.length];
}

function getCatColor(slug) {
  const cat = state.categories.find((c) => c.slug === slug);
  if (cat) return cat.color;
  const d = {
    important: "#d93025",
    informational: "#1a73e8",
    newsletter: "#7c3aed",
    "action-required": "#e37400",
  };
  return d[slug] || "#80868b";
}

function getCatName(slug) {
  const cat = state.categories.find((c) => c.slug === slug);
  return cat ? cat.display_name : slug;
}

function autoColor(str) {
  const p = [
    "#4285f4",
    "#34a853",
    "#ea4335",
    "#fbbc04",
    "#7b1fa2",
    "#0097a7",
    "#e64a19",
    "#5c6bc0",
    "#00897b",
    "#c2185b",
  ];
  let h = 0;
  for (let i = 0; i < str.length; i++) h = str.charCodeAt(i) + ((h << 5) - h);
  return p[Math.abs(h) % p.length];
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

// ═══════════════════════════════════════════════════════════════════════
// THEME
// ═══════════════════════════════════════════════════════════════════════
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
  if (state.selectedEmail) renderDetailPane();
}

// ═══════════════════════════════════════════════════════════════════════
// SIDEBAR (#2 — toggle collapse on desktop, overlay on mobile)
// ═══════════════════════════════════════════════════════════════════════
function applySidebarState() {
  const sb = document.getElementById("sidebar");
  const topbarLeft = document.querySelector(".topbar-left");
  const appName = document.querySelector(".topbar-app-name");
  if (!sb) return;
  if (window.innerWidth > 768) {
    // Desktop: collapse/expand
    sb.classList.toggle("collapsed", state.sidebarCollapsed);
    sb.classList.remove("open");
    document.querySelector(".sidebar-overlay")?.remove();
    // Shrink topbar-left to collapsed width — keeps search bar stable
    if (topbarLeft) {
      topbarLeft.style.width = state.sidebarCollapsed
        ? "var(--sidebar-collapsed-w)"
        : "var(--sidebar-w)";
    }
    // Only hide Dispatch text; AI icon always shows
    if (appName) appName.style.display = state.sidebarCollapsed ? "none" : "";
  }
}

function toggleSidebar() {
  if (window.innerWidth > 768) {
    // Desktop: toggle collapsed
    state.sidebarCollapsed = !state.sidebarCollapsed;
    localStorage.setItem(
      "ea-sidebar-collapsed",
      state.sidebarCollapsed ? "1" : "0",
    );
    applySidebarState();
  } else {
    // Mobile: overlay
    state.sidebarOpen = !state.sidebarOpen;
    const sb = document.getElementById("sidebar");
    if (state.sidebarOpen) {
      sb.classList.add("open");
      const overlay = document.createElement("div");
      overlay.className = "sidebar-overlay";
      overlay.addEventListener("click", closeSidebar);
      document.querySelector(".app-body").appendChild(overlay);
    } else {
      closeSidebar();
    }
  }
}

function closeSidebar() {
  state.sidebarOpen = false;
  document.getElementById("sidebar").classList.remove("open");
  document.querySelector(".sidebar-overlay")?.remove();
}

// ═══════════════════════════════════════════════════════════════════════
// NAVIGATION (#5 — full-width for non-inbox views)
// ═══════════════════════════════════════════════════════════════════════
function navigateTo(view) {
  state.view = view;
  state.detailMode = "empty";
  state.selectedId = null;
  state.selectedEmail = null;
  state.selectedMeeting = null;
  state.draft = null;
  state.composeContext = null;

  // Update sidebar active state
  document.querySelectorAll("#sidebar-nav .nav-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.view === view);
  });

  // Always 2-panel: no full-width
  const body = document.getElementById("app-body");
  body.classList.remove("show-detail");
  body.classList.remove("full-width");

  renderListPane();
  renderDetailPane();

  // Auto-load data for the view
  if (view === "todo")
    loadTodos().then(() => {
      renderListPane();
      renderDetailPane();
    });
  if (view === "labels")
    loadCategories().then(() => {
      renderListPane();
      renderDetailPane();
    });
  if (view === "meetings")
    loadMeetings().then(() => {
      renderListPane();
      renderDetailPane();
    });
}

function showCompose(ctx = null) {
  state.composeContext = ctx;
  state.draft = null;
  openComposeFloat(ctx);
}

function showEmailDetail() {
  state.detailMode = "email";
  const body = document.getElementById("app-body");
  body.classList.add("show-detail");
  renderDetailPane();
}

function goBackToList() {
  const body = document.getElementById("app-body");
  body.classList.remove("show-detail");
}

// ═══════════════════════════════════════════════════════════════════════
// DATA LOADING
// ═══════════════════════════════════════════════════════════════════════
async function loadEmails() {
  state.isLoadingEmails = true;
  renderListPane();
  try {
    const params = new URLSearchParams();
    params.set("page", state.currentPage);
    params.set("per_page", "25");
    if (state.searchQuery) params.set("search", state.searchQuery);
    const serverFilters = state.activeFilters.filter((f) => f !== "__unread");
    if (serverFilters.length) params.set("category", serverFilters.join(","));
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
    renderListPane();
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

// #6 — Meetings hub data loading
async function loadMeetings() {
  try {
    const data = await api.get("/api/meetings?days=14");
    state.meetings = data.meetings || [];
  } catch (e) {
    console.error("loadMeetings failed:", e);
  }
}

async function selectEmail(id) {
  if (state.selectedId === id && state.selectedEmail) {
    showEmailDetail();
    return;
  }
  state.selectedId = id;
  state.selectedEmail = null;
  state.draft = null;
  state.isLoadingDetail = true;
  state.detailMode = "email";
  const body = document.getElementById("app-body");
  body.classList.add("show-detail");
  renderListPane();
  renderDetailPane();
  try {
    const data = await api.get(`/api/emails/${id}`);
    state.selectedEmail = data;
    const listItem = state.emails.find((e) => e.id === id);
    if (listItem) listItem.is_read = true;
  } catch (e) {
    toast("Failed to load email: " + e.message, "error");
  } finally {
    state.isLoadingDetail = false;
    renderListPane();
    renderDetailPane();
    updateFetchInfo();
  }
}

async function fetchNewEmails(retrain = false) {
  if (state.isFetching) return;
  state.isFetching = true;
  const btn = document.getElementById("btn-refresh");
  if (btn) btn.classList.add("fetching");
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
    if (btn) btn.classList.remove("fetching");
  }
}

function updateFetchInfo() {
  const el = document.getElementById("fetch-info");
  if (el)
    el.textContent = state.lastFetchTs
      ? `Updated ${timeAgo(state.lastFetchTs)}`
      : "";
  const badge = document.getElementById("inbox-badge");
  const unread = state.emails.filter((e) => !e.is_read).length;
  if (badge) badge.textContent = unread > 0 ? unread : "";
  document.title = unread > 0 ? `(${unread}) Dispatch` : "Dispatch";
}

// ═══════════════════════════════════════════════════════════════════════
// RENDER — List Pane
// ═══════════════════════════════════════════════════════════════════════
function renderListPane() {
  const el = document.getElementById("list-pane");
  if (state.view === "inbox") renderInboxList(el);
  else if (state.view === "todo") renderTodoList(el);
  else if (state.view === "meetings") renderMeetingsList(el);
  else if (state.view === "labels") renderLabelsList(el);
}

// #8 — simplified email cards: NO subject in list, only sender + snippet
// #9 — compact filter chips: no counts
function renderInboxList(el) {
  const enabledCats = state.categories.filter((c) => c.enabled !== 0);
  const unreadActive = state.activeFilters.includes("__unread");
  const activeCount = state.activeFilters.length;

  const filterOptions =
    `<label class="filter-option"><input type="checkbox" data-filter="__unread" ${unreadActive ? "checked" : ""}/> Unread</label>` +
    enabledCats
      .map((c) => {
        const active = state.activeFilters.includes(c.slug);
        return `<label class="filter-option"><input type="checkbox" data-filter="${esc(c.slug)}" ${active ? "checked" : ""}/> ${esc(c.display_name)}</label>`;
      })
      .join("");

  const filterDropdownHtml = `<div class="filter-dropdown">
    <button class="filter-dropdown-btn" id="filter-toggle">
      <span class="material-symbols-rounded" style="font-size:18px">filter_list</span>
      Filter${activeCount ? ` (${activeCount})` : ""}
    </button>
    <div class="filter-dropdown-menu" id="filter-menu">${filterOptions}</div>
  </div>`;

  if (state.isLoadingEmails && !state.emails.length) {
    el.innerHTML = `
      <div class="list-header"><h2>Inbox</h2>${filterDropdownHtml}</div>
      <div class="loading-center"><span class="spinner spinner-lg"></span></div>`;
    return;
  }

  let displayEmails = state.emails;
  if (state.activeFilters.includes("__unread")) {
    displayEmails = displayEmails.filter((e) => !e.is_read);
  }

  const rowsHtml = displayEmails
    .map((e) => {
      const selected = e.id === state.selectedId;
      const unread = !e.is_read;
      const catTags = (e.category || "").split(",").filter(Boolean);
      const badgesHtml = catTags
        .map((t) => {
          const c = getCatColor(t.trim());
          return `<span class="email-badge" style="color:${c};background:${c}14">${esc(getCatName(t.trim()))}</span>`;
        })
        .join("");

      return `
      <div class="email-row ${selected ? "selected" : ""} ${unread ? "unread" : ""}" data-email-id="${esc(e.id)}">
        <div class="email-info">
          <div class="email-top-row">
            <span class="email-sender">${esc(senderName(e.sender))}</span>
            ${e.urgent ? '<span class="urgent-dot"></span>' : ""}
            <span class="email-date">${esc(simpleDate(e.date))}</span>
            ${unread ? '<span class="unread-dot"></span>' : ""}
          </div>
          ${badgesHtml ? `<div class="email-bottom-row"><span class="email-badges">${badgesHtml}</span></div>` : ""}
        </div>
      </div>`;
    })
    .join("");

  let pagHtml = "";
  if (state.totalPages > 1) {
    pagHtml = `<div class="pagination">
      <button ${state.currentPage <= 1 ? "disabled" : ""} data-page="${state.currentPage - 1}">Prev</button>
      <span>${state.currentPage} of ${state.totalPages}</span>
      <button ${state.currentPage >= state.totalPages ? "disabled" : ""} data-page="${state.currentPage + 1}">Next</button>
    </div>`;
  }

  el.innerHTML = `
    <div class="list-header"><h2>Inbox</h2>${filterDropdownHtml}</div>
    <div class="email-list fade-in">
      ${rowsHtml || '<div class="empty-state"><span class="material-symbols-rounded">inbox</span><p>No emails</p></div>'}
    </div>
    ${pagHtml}`;
}

function renderTodoList(el) {
  const header = `<div class="list-header"><h2>Todo</h2></div>`;
  if (!state.todos.length) {
    el.innerHTML = `${header}
      <div class="empty-state"><span class="material-symbols-rounded">task_alt</span><p>No tasks yet</p></div>`;
    return;
  }

  el.innerHTML = `${header}
    <div class="todo-list fade-in">${state.todos
      .map(
        (t) => `
      <div class="todo-row" data-todo-row-id="${t.id}">
        <div class="todo-info">
          <div class="todo-task">${esc(t.task)}</div>
          ${t.source_subject ? `<div class="todo-source">${esc(t.source_subject.slice(0, 50))} · ${esc(senderName(t.source_sender || ""))}</div>` : ""}
        </div>
        <button class="todo-check" data-todo-id="${t.id}" title="Done">
          <span class="material-symbols-rounded">check_circle</span>
        </button>
      </div>`,
      )
      .join("")}
    </div>`;
}

function renderTodoDetail(el) {
  const count = state.todos.length;
  el.innerHTML = `
    <div class="meeting-detail-pane fade-in">
      <h3>Overview</h3>
      <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:16px">
        <span style="font-size:48px;font-weight:700;color:var(--primary)">${count}</span>
        <span style="font-size:14px;color:var(--on-surface-muted)">${count === 1 ? "task" : "tasks"} remaining</span>
      </div>
      <div class="todo-add-manual" style="display:flex;gap:8px;margin-bottom:20px">
        <input class="input" id="todo-manual-input" placeholder="Add a task…" style="flex:1">
        <button class="btn btn-primary btn-sm" id="btn-todo-manual-add"><span class="material-symbols-rounded">add</span> Add</button>
      </div>
      <h3 style="margin-bottom:8px">Suggested from Emails</h3>
      <div id="todo-suggestions-area"><div class="loading-center"><span class="spinner"></span></div></div>
    </div>`;

  // Load suggestions
  loadTodoSuggestions();
}

async function loadTodoSuggestions() {
  const area = document.getElementById("todo-suggestions-area");
  if (!area) return;
  try {
    const data = await api.get("/api/todos/suggestions");
    const suggestions = data.suggestions || [];
    if (!suggestions.length) {
      area.innerHTML =
        '<p style="font-size:13px;color:var(--on-surface-subtle)">No suggestions from recent emails.</p>';
      return;
    }
    area.innerHTML = suggestions
      .map(
        (s, i) => `
      <div class="todo-suggestion-row" data-suggest-idx="${i}">
        <div class="todo-suggestion-info">
          <div class="todo-suggestion-task">${esc(s.label)}</div>
          <div class="todo-suggestion-source">${esc((s.email_subject || "").slice(0, 50))} · ${esc(senderName(s.email_sender || ""))} · ${esc(simpleDate(s.email_date))}${s.urgent ? ' · <span style="color:var(--error);font-weight:600">Urgent</span>' : ""}</div>
        </div>
        <button class="btn btn-primary btn-sm todo-suggestion-add" data-sug-label="${esc(s.label)}" data-sug-email="${esc(s.email_id)}">
          <span class="material-symbols-rounded">add</span> Add
        </button>
      </div>`,
      )
      .join("");
  } catch (e) {
    area.innerHTML =
      '<p style="font-size:13px;color:var(--on-surface-subtle)">Could not load suggestions.</p>';
  }
}

// Meetings list pane — just a list of meetings
function renderMeetingsList(el) {
  const header = `<div class="list-header"><h2>Meetings</h2></div>`;

  if (!state.meetings.length) {
    el.innerHTML = `${header}
      <div class="empty-state"><span class="material-symbols-rounded">event</span><p>No upcoming meetings</p></div>`;
    return;
  }

  const cardsHtml = state.meetings
    .map((m) => {
      const start = m.start
        ? new Date(m.start.dateTime || m.start.date || m.start)
        : null;
      const end = m.end
        ? new Date(m.end.dateTime || m.end.date || m.end)
        : null;
      const dayStr = start
        ? start.toLocaleDateString("en-US", { weekday: "short" })
        : "";
      const dateNum = start ? start.getDate() : "";
      const timeStr = start
        ? start.toLocaleTimeString("en-US", {
            hour: "numeric",
            minute: "2-digit",
          })
        : "";
      const endStr = end
        ? end.toLocaleTimeString("en-US", {
            hour: "numeric",
            minute: "2-digit",
          })
        : "";
      const attendees = (m.attendees || [])
        .map((a) => a.email || a)
        .slice(0, 3)
        .join(", ");
      const eventId = m.id || "";

      return `
    <div class="meeting-card" data-meeting-id="${esc(eventId)}">
      <div class="meeting-time-block">
        <div class="meeting-day">${esc(dayStr)}</div>
        <div class="meeting-date">${esc(String(dateNum))}</div>
        <div class="meeting-time">${esc(timeStr)}</div>
      </div>
      <div class="meeting-details">
        <div class="meeting-title">${esc(m.summary || "Untitled")}</div>
        <div class="meeting-meta">${esc(timeStr)}${endStr ? ` – ${esc(endStr)}` : ""}${m.location ? ` · ${esc(m.location)}` : ""}</div>
        ${attendees ? `<div class="meeting-attendees">${esc(attendees)}</div>` : ""}
      </div>
      <div class="meeting-actions">
        <button class="meeting-action-btn danger" data-delete-meeting="${esc(eventId)}" title="Delete">
          <span class="material-symbols-rounded" style="font-size:18px">delete</span>
        </button>
      </div>
    </div>`;
    })
    .join("");

  el.innerHTML = `${header}<div class="meetings-content">${cardsHtml}</div>`;
}

// Meetings detail pane — create/edit meeting form + selected meeting detail
function renderMeetingsDetail(el) {
  // If a meeting is selected, show its detail with delete/reschedule
  if (state.selectedMeeting) {
    const m = state.selectedMeeting;
    const start = m.start
      ? new Date(m.start.dateTime || m.start.date || m.start)
      : null;
    const end = m.end ? new Date(m.end.dateTime || m.end.date || m.end) : null;
    const dateStr = start
      ? start.toLocaleDateString("en-US", {
          weekday: "long",
          year: "numeric",
          month: "long",
          day: "numeric",
        })
      : "";
    const timeStr = start
      ? start.toLocaleTimeString("en-US", {
          hour: "numeric",
          minute: "2-digit",
        })
      : "";
    const endStr = end
      ? end.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })
      : "";
    const attendees = (m.attendees || []).map((a) => a.email || a).join(", ");
    const eventId = m.id || "";

    el.innerHTML = `
      <div class="meeting-detail-pane fade-in">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
          <button class="topbar-icon" id="btn-meeting-back" title="Back"><span class="material-symbols-rounded">arrow_back</span></button>
          <h3 style="margin:0">${esc(m.summary || "Untitled Meeting")}</h3>
        </div>
        <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:20px">
          <div style="display:flex;align-items:center;gap:10px">
            <span class="material-symbols-rounded" style="color:var(--primary)">calendar_today</span>
            <span style="font-size:14px;color:var(--on-surface)">${esc(dateStr)}</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px">
            <span class="material-symbols-rounded" style="color:var(--primary)">schedule</span>
            <span style="font-size:14px;color:var(--on-surface)">${esc(timeStr)}${endStr ? ` – ${esc(endStr)}` : ""}</span>
          </div>
          ${m.location ? `<div style="display:flex;align-items:center;gap:10px"><span class="material-symbols-rounded" style="color:var(--primary)">location_on</span><span style="font-size:14px;color:var(--on-surface)">${esc(m.location)}</span></div>` : ""}
          ${attendees ? `<div style="display:flex;align-items:center;gap:10px"><span class="material-symbols-rounded" style="color:var(--primary)">group</span><span style="font-size:14px;color:var(--on-surface)">${esc(attendees)}</span></div>` : ""}
        </div>
        <div style="display:flex;gap:8px;margin-bottom:20px">
          <button class="btn btn-secondary btn-sm" id="btn-reschedule-meeting" data-event-id="${esc(eventId)}"><span class="material-symbols-rounded">edit_calendar</span> Reschedule</button>
          <button class="btn btn-danger btn-sm" id="btn-delete-selected-meeting" data-event-id="${esc(eventId)}" style="border:1px solid var(--divider)"><span class="material-symbols-rounded">delete</span> Delete</button>
        </div>
        <div id="reschedule-area"></div>
      </div>`;
    return;
  }

  el.innerHTML = `
    <div class="meeting-detail-pane fade-in">
      <h3>Create Meeting</h3>
      <div class="meeting-form">
        <div class="form-field">
          <label class="form-label">Title</label>
          <input class="input" id="meeting-new-title" placeholder="Meeting title" value="">
        </div>
        <div class="form-field">
          <label class="form-label">Attendee</label>
          <input class="input" id="meeting-new-attendee" placeholder="attendee@example.com">
        </div>
        <div class="form-field">
          <label class="form-label">Duration</label>
          <select class="select-input select-full" id="meeting-new-duration">
            <option value="15">15 min</option>
            <option value="30" selected>30 min</option>
            <option value="45">45 min</option>
            <option value="60">1 hour</option>
            <option value="90">1.5 hours</option>
          </select>
        </div>
        <div style="display:flex;gap:8px;margin-top:8px">
          <button class="btn btn-primary btn-sm" id="btn-meeting-check-avail"><span class="material-symbols-rounded">search</span> Check Availability</button>
          <button class="btn btn-secondary btn-sm" id="btn-meeting-refresh"><span class="material-symbols-rounded">refresh</span> Refresh</button>
        </div>
        <div id="meeting-slots-area" style="margin-top:16px"></div>
      </div>
    </div>`;
}

function renderLabelsList(el) {
  const cats = state.categories;
  const mainSlugs = [
    "important",
    "informational",
    "newsletter",
    "action-required",
  ];
  const supplementary = cats.filter((c) => !mainSlugs.includes(c.slug));
  const allCats = [
    ...cats.filter((c) => mainSlugs.includes(c.slug)),
    ...supplementary,
  ];

  const header = `<div class="list-header"><h2>Labels</h2></div>`;

  if (!allCats.length) {
    el.innerHTML = `${header}
      <div class="empty-state"><span class="material-symbols-rounded">label</span><p>No labels yet</p></div>`;
    return;
  }

  const rowsHtml = allCats
    .map(
      (c) => `
    <div class="label-row" data-cat-slug="${esc(c.slug)}">
      <span class="label-dot" style="background:${esc(c.color)}"></span>
      <div style="flex:1">
        <div class="label-name">${esc(c.display_name)}</div>
        ${c.description ? `<div class="label-desc">${esc(c.description)}</div>` : ""}
      </div>
      <span class="label-count">${c.count || 0}</span>
      ${!mainSlugs.includes(c.slug) ? `<button class="label-delete" data-cat-delete="${esc(c.slug)}" title="Delete"><span class="material-symbols-rounded" style="font-size:16px">delete</span></button>` : ""}
    </div>`,
    )
    .join("");

  el.innerHTML = `${header}<div class="labels-content fade-in"><div class="label-list">${rowsHtml}</div></div>`;
}

function renderLabelsDetail(el) {
  const suggestHtml =
    state.categorySuggestions == null
      ? `<button class="btn btn-secondary btn-sm" id="btn-load-suggestions" style="width:100%"><span class="material-symbols-rounded">auto_awesome</span> AI Suggestions</button>`
      : state.categorySuggestions.length === 0
        ? `<p style="font-size:13px;color:var(--on-surface-subtle)">No suggestions available</p>`
        : state.categorySuggestions
            .map(
              (s, i) => `
          <div class="suggestion-item">
            <div class="suggestion-info">
              <div class="suggestion-name">${esc(s.proposed_name)}</div>
              <div class="suggestion-reason">${esc(s.reason)}</div>
            </div>
            <button class="btn btn-primary btn-sm" data-suggest-idx="${i}">Add</button>
          </div>`,
            )
            .join("");

  el.innerHTML = `
    <div class="meeting-detail-pane fade-in">
      <h3>New Label</h3>
      <div class="label-create">
        <input class="input" id="new-label-name" placeholder="Label name" maxlength="40">
        <input class="input" id="new-label-desc" placeholder="Description (optional)">
        <button class="btn btn-primary btn-sm" id="btn-create-label">Create Label</button>
      </div>
      <h3 style="margin-top:24px">AI Suggestions</h3>
      <div class="suggestions-area">${suggestHtml}</div>
    </div>`;
}

// ═══════════════════════════════════════════════════════════════════════
// RENDER — Detail Pane
// ═══════════════════════════════════════════════════════════════════════
function renderDetailPane() {
  const el = document.getElementById("detail-pane");
  if (state.view === "todo") {
    renderTodoDetail(el);
    return;
  }
  if (state.view === "meetings") {
    renderMeetingsDetail(el);
    return;
  }
  if (state.view === "labels") {
    renderLabelsDetail(el);
    return;
  }
  if (state.detailMode === "email") renderEmailDetail(el);
  else renderEmptyDetail(el);
}

function renderEmptyDetail(el) {
  el.innerHTML = `<div class="empty-state"><span class="material-symbols-rounded">mail</span><p>Select an email to read</p></div>`;
}

// #10 — Redesigned email detail
function renderEmailDetail(el) {
  if (state.isLoadingDetail) {
    el.innerHTML = `<div class="loading-center"><span class="spinner spinner-lg"></span></div>`;
    return;
  }

  const email = state.selectedEmail;
  if (!email) {
    renderEmptyDetail(el);
    return;
  }

  const opts = email.decision_options || [];
  const replies = opts.filter((o) => o.type === "reply");
  const todos = opts.filter((o) => o.type === "todo");
  const meetings = opts.filter((o) => o.type === "meeting");

  // #10.3 — category badges inline, clickable for dropdown
  const catTags = (email.category || "").split(",").filter(Boolean);
  const badgesHtml = catTags
    .map((t) => {
      const c = getCatColor(t.trim());
      return `<span class="cat-badge" id="cat-badge-click" style="color:${c};background:${c}14" title="Click to change">${esc(getCatName(t.trim()))}<span class="material-symbols-rounded" style="font-size:12px;margin-left:2px">expand_more</span></span>`;
    })
    .join("");

  // #10.5 — separate rows for reply, todo, meeting actions
  let actionsHtml = "";
  if (opts.length) {
    let replyRow = "";
    let todoRow = "";
    let meetingRow = "";

    if (replies.length) {
      replyRow = `<div class="actions-row">${replies
        .map(
          (r, i) =>
            `<button class="action-chip reply" data-action-type="reply" data-action-idx="${i}" title="${esc(r.context || "")}">${esc(r.label)}</button>`,
        )
        .join("")}</div>`;
    }
    if (todos.length) {
      todoRow = `<div class="actions-row">${todos
        .map(
          (t, i) =>
            `<button class="action-chip todo" data-action-type="todo" data-action-idx="${i}" title="${esc(t.context || "")}">${esc(t.label)}</button>`,
        )
        .join("")}</div>`;
    }
    // #10.5 — meeting actions just navigate to meetings tab
    if (meetings.length) {
      meetingRow = `<div class="actions-row">${meetings
        .map(
          (m, i) =>
            `<button class="action-chip meeting" data-action-type="meeting" data-action-idx="${i}" title="${esc(m.context || "")}">${esc(m.label)}</button>`,
        )
        .join("")}</div>`;
    }

    actionsHtml = `<div class="actions-section"><div class="actions-title">Suggested Actions</div>${replyRow}${todoRow}${meetingRow}</div>`;
  } else if (email.no_action_message) {
    actionsHtml = `<div class="actions-section"><p style="font-size:13px;color:var(--on-surface-subtle)">${esc(email.no_action_message)}</p></div>`;
  }

  // #10.1 — toolbar actions moved inline into action-bar at bottom
  // #10.6 — "Custom Instructions" button replaces "Reply"
  el.innerHTML = `
    <div class="detail-toolbar">
      <button class="back-btn" id="btn-back"><span class="material-symbols-rounded">arrow_back</span></button>
    </div>
    <div class="detail-scroll fade-in">
      <h1 class="detail-subject">${esc(email.subject)}</h1>
      <div class="detail-meta">
        <div class="sender-row">
          <span class="sender-avatar" style="background:${avatarColor(senderName(email.sender))}">${esc(avatarInitials(email.sender))}</span>
          <div class="sender-details">
            <div class="sender-name-line">${esc(senderName(email.sender))}</div>
            <div class="sender-email-line">${esc(senderEmail(email.sender))} · ${esc(simpleDate(email.date))}</div>
          </div>
        </div>
        <div class="detail-badges" id="detail-badges">${badgesHtml}</div>
      </div>
      ${email.summary ? `<div class="ai-summary"><span class="material-symbols-rounded">auto_awesome</span><p>${esc(email.summary)}</p></div>` : ""}
      <div class="email-body-wrap" id="email-body-wrap"></div>
      ${actionsHtml}
      <div class="action-bar">
        <button class="btn btn-ghost btn-sm" id="btn-custom-instructions"><span class="material-symbols-rounded">edit_note</span> Custom Instructions</button>
        <button class="btn btn-ghost btn-sm" id="btn-rethink">
          ${state.isRethinking ? '<span class="spinner" style="width:14px;height:14px;border-width:2px"></span>' : '<span class="material-symbols-rounded">refresh</span>'}
          ${state.isRethinking ? "Thinking…" : "Rethink"}
        </button>
        <span class="action-bar-spacer"></span>
        <button class="tool-btn" id="btn-mark-unread" title="Mark unread"><span class="material-symbols-rounded">mark_email_unread</span></button>
        <button class="tool-btn" id="btn-snooze" title="Snooze" style="position:relative"><span class="material-symbols-rounded">snooze</span></button>
        <button class="tool-btn" id="btn-forward" title="Forward"><span class="material-symbols-rounded">forward</span></button>
        <button class="tool-btn danger" id="btn-archive" title="Archive"><span class="material-symbols-rounded">archive</span></button>
      </div>
      <div id="inline-reply-area"></div>
      <div id="draft-area"></div>
    </div>`;

  renderEmailBody(email);
}

// #10.2 — only latest email body, not full thread
function renderEmailBody(email) {
  const container = document.getElementById("email-body-wrap");
  if (!container) return;

  const bodyHtml = email.body_html;
  const plainText = email.body || email.snippet || "";

  if (bodyHtml) {
    const iframe = document.createElement("iframe");
    iframe.sandbox = "allow-same-origin";
    iframe.style.cssText = "width:100%;display:block;border:none;";
    container.appendChild(iframe);

    const bgColor = state.theme === "dark" ? "#1f1f1f" : "#ffffff";
    const textColor = state.theme === "dark" ? "#e3e3e3" : "#1f1f1f";
    const linkColor = state.theme === "dark" ? "#8ab4f8" : "#1a73e8";

    const styleTag = `<style>
      body { background:${bgColor}!important; color:${textColor}!important; font-family:'Inter',sans-serif; font-size:14px; line-height:1.6; margin:0; padding:16px 0; }
      * { color:${textColor}!important; }
      a { color:${linkColor}!important; }
      img { max-width:100%; height:auto; }
      table { max-width:100%!important; }
      /* Hide quoted/forwarded thread text */
      .gmail_quote, blockquote, .moz-cite-prefix { display:none!important; }
    </style>`;

    iframe.srcdoc = styleTag + bodyHtml;
    iframe.onload = () => {
      try {
        const h = iframe.contentDocument.body.scrollHeight;
        const maxH = window.innerHeight - 460;
        iframe.style.height = Math.min(h + 32, maxH) + "px";
      } catch (_) {
        iframe.style.height = "300px";
      }
    };
  } else if (plainText) {
    // Strip quoted text patterns from plain text (#10.2)
    let cleaned = plainText;
    const quoteIdx = cleaned.search(/\n\s*On .+ wrote:\s*\n/);
    if (quoteIdx > 0) cleaned = cleaned.substring(0, quoteIdx).trim();
    container.innerHTML = `<div class="email-body-plain">${esc(cleaned)}</div>`;
  } else {
    container.innerHTML = `<div class="email-body-plain" style="color:var(--on-surface-subtle)">No content</div>`;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// FLOATING COMPOSE — Gmail-style (#9)
// ═══════════════════════════════════════════════════════════════════════
let composeAttachments = [];

function openComposeFloat(ctx = null) {
  const floatEl = document.getElementById("compose-float");
  floatEl.classList.remove("hidden", "minimized", "expanded");
  const titleEl = document.getElementById("compose-float-title");

  ctx = ctx || {};
  const isReply = !!ctx.replyToId;
  titleEl.textContent = isReply ? `Reply: ${ctx.subject || ""}` : "New Message";
  composeAttachments = [];

  const body = document.getElementById("compose-float-body");
  body.innerHTML = `
    <div class="form-field autocomplete-wrap">
      <label class="form-label">To</label>
      <input class="input" id="compose-to" placeholder="Recipient" autocomplete="off" value="${esc(ctx.to || "")}">
      <div id="compose-autocomplete" class="autocomplete-drop hidden"></div>
    </div>
    <div class="form-field autocomplete-wrap">
      <label class="form-label">CC</label>
      <input class="input" id="compose-cc" placeholder="CC (optional)" autocomplete="off" value="${esc(ctx.cc || "")}">
      <div id="compose-cc-autocomplete" class="autocomplete-drop hidden"></div>
    </div>
    <div class="form-field">
      <label class="form-label">Subject</label>
      <input class="input" id="compose-subject" placeholder="Subject" value="${esc(ctx.subject || "")}">
    </div>
    <div class="form-field">
      <label class="form-label">Tone</label>
      <select class="select-input select-full" id="compose-tone">
        <option value="">Auto</option>
        <option value="professional">Professional</option>
        <option value="friendly">Friendly</option>
        <option value="casual">Casual</option>
        <option value="formal">Formal</option>
      </select>
    </div>
    <div class="form-field">
      <label class="form-label">Message</label>
      <textarea class="textarea" id="compose-context" rows="4" placeholder="What do you want to say?">${esc(ctx.context || "")}</textarea>
    </div>
    <div class="form-field">
      <div class="attach-dropzone" id="compose-dropzone">
        <span class="material-symbols-rounded" style="font-size:20px;vertical-align:middle">attach_file</span>
        Drop files here or click to attach
        <input type="file" id="compose-file-input" multiple style="display:none">
      </div>
      <div class="compose-attachments" id="compose-attachments"></div>
    </div>
    <div class="compose-actions">
      <button class="btn btn-primary" id="btn-compose-draft"><span class="material-symbols-rounded">edit_note</span> AI Draft</button>
      <button class="btn btn-ghost btn-sm" id="btn-compose-clear" style="margin-left:auto">Clear</button>
    </div>
    <div class="compose-draft-area" id="compose-draft-area" ${isReply ? `data-reply-to-id="${esc(ctx.replyToId)}" data-reply-to-thread="${esc(ctx.replyToThread || "")}"` : ""}></div>`;

  setupComposeAutocomplete();
  setupComposeAttachments();

  if (ctx.autoAction === "draft") {
    setTimeout(() => handleComposeDraft(), 100);
  }
}

function closeComposeFloat() {
  const floatEl = document.getElementById("compose-float");
  floatEl.classList.add("hidden");
  composeAttachments = [];
}

function setupComposeAttachments() {
  const dropzone = document.getElementById("compose-dropzone");
  const fileInput = document.getElementById("compose-file-input");
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    addFiles(e.dataTransfer.files);
  });

  fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
    fileInput.value = "";
  });
}

function addFiles(fileList) {
  for (const f of fileList) {
    composeAttachments.push(f);
  }
  renderAttachments();
}

function renderAttachments() {
  const container = document.getElementById("compose-attachments");
  if (!container) return;
  container.innerHTML = composeAttachments
    .map(
      (f, i) => `
    <div class="compose-attachment">
      <span class="material-symbols-rounded" style="font-size:14px">description</span>
      ${esc(f.name)}
      <span class="remove-attach" data-attach-idx="${i}"><span class="material-symbols-rounded">close</span></span>
    </div>`,
    )
    .join("");

  container.querySelectorAll(".remove-attach").forEach((btn) => {
    btn.addEventListener("click", () => {
      composeAttachments.splice(parseInt(btn.dataset.attachIdx), 1);
      renderAttachments();
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════
// COMPOSE — Autocomplete
// ═══════════════════════════════════════════════════════════════════════
function setupComposeAutocomplete() {
  setupFieldAutocomplete("compose-to", "compose-autocomplete");
  setupFieldAutocomplete("compose-cc", "compose-cc-autocomplete");
}

function setupFieldAutocomplete(inputId, dropdownId) {
  const input = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);
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
      <div class="ac-item" data-email="${esc(c.email)}">
        <div class="ac-item-name">${esc(c.name || c.email)}</div>
        <div class="ac-item-email">${esc(c.email)}</div>
      </div>`,
        )
        .join("");
      dropdown.classList.remove("hidden");
    }, 200),
  );

  dropdown.addEventListener("click", (e) => {
    const item = e.target.closest(".ac-item");
    if (item) {
      input.value = item.dataset.email;
      dropdown.classList.add("hidden");
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".autocomplete-wrap"))
      dropdown.classList.add("hidden");
  });
}

// ═══════════════════════════════════════════════════════════════════════
// ACTION HANDLERS
// ═══════════════════════════════════════════════════════════════════════
async function handleArchive() {
  if (!state.selectedId) return;
  try {
    await api.post(`/api/emails/${state.selectedId}/archive`);
    toast("Archived", "success");
    state.emails = state.emails.filter((e) => e.id !== state.selectedId);
    state.selectedId = null;
    state.selectedEmail = null;
    state.detailMode = "empty";
    goBackToList();
    renderListPane();
    renderDetailPane();
  } catch (e) {
    toast(e.message, "error");
  }
}

async function handleSnooze(hours) {
  if (!state.selectedId) return;
  const until = new Date(Date.now() + hours * 3600000).toISOString();
  try {
    await api.post(`/api/emails/${state.selectedId}/snooze`, { until });
    const labels = {
      1: "1 hour",
      3: "3 hours",
      16: "tomorrow",
      168: "next week",
    };
    toast(`Snoozed for ${labels[hours] || hours + "h"}`, "success");
    state.emails = state.emails.filter((e) => e.id !== state.selectedId);
    state.selectedId = null;
    state.selectedEmail = null;
    state.detailMode = "empty";
    goBackToList();
    renderListPane();
    renderDetailPane();
  } catch (e) {
    toast(e.message, "error");
  }
}

async function handleMarkUnread() {
  if (!state.selectedId) return;
  const listItem = state.emails.find((e) => e.id === state.selectedId);
  if (listItem) listItem.is_read = false;
  updateFetchInfo();
  try {
    await api.post(`/api/emails/${state.selectedId}/mark-unread`);
  } catch (_) {}
  toast("Marked as unread", "success");
  await loadEmails();
  renderListPane();
}

async function handleRethink() {
  if (!state.selectedId || state.isRethinking) return;
  state.isRethinking = true;
  renderDetailPane();
  try {
    const data = await api.post(`/api/emails/${state.selectedId}/rethink`);
    state.selectedEmail = data;
    const listItem = state.emails.find((e) => e.id === state.selectedId);
    if (listItem) {
      listItem.category = data.category;
      listItem.urgent = data.urgent;
    }
    toast("Actions regenerated", "success");
  } catch (e) {
    toast("Rethink failed: " + e.message, "error");
  } finally {
    state.isRethinking = false;
    renderDetailPane();
    renderListPane();
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
    const area = document.getElementById("inline-reply-area");
    if (area) area.innerHTML = "";
    const draftArea = document.getElementById("draft-area");
    if (draftArea) draftArea.innerHTML = "";
    renderDetailPane();
  } catch (e) {
    toast("Send failed: " + e.message, "error");
  }
}

async function handleComposeDraft() {
  const to = (document.getElementById("compose-to") || {}).value || "";
  const cc = (document.getElementById("compose-cc") || {}).value || "";
  const subject =
    (document.getElementById("compose-subject") || {}).value || "";
  const context =
    (document.getElementById("compose-context") || {}).value || "";
  const tone = (document.getElementById("compose-tone") || {}).value || "";

  if (!to.trim() || !subject.trim() || !context.trim()) {
    toast("Fill in To, Subject, and Message", "error");
    return;
  }

  const area = document.getElementById("compose-draft-area");
  if (!area) return;
  const replyToId = area.dataset.replyToId || "";

  area.innerHTML =
    '<div class="loading-center"><span class="spinner"></span> Drafting…</div>';
  area.scrollIntoView({ behavior: "smooth", block: "nearest" });

  try {
    let data;
    if (replyToId) {
      data = await api.post(`/api/emails/${replyToId}/draft`, {
        decision: tone ? `[Tone: ${tone}] ${context}` : context,
      });
    } else {
      data = await api.post("/api/compose/draft", {
        to,
        cc,
        subject,
        context,
        tone,
      });
    }
    area.innerHTML = `
      <div class="draft-section fade-in">
        <h4>Draft</h4>
        <textarea class="draft-textarea" id="compose-draft-text">${esc(data.draft)}</textarea>
        <div class="draft-actions">
          <button class="btn btn-primary" id="btn-compose-send"><span class="material-symbols-rounded">send</span> Send</button>
        </div>
      </div>`;
    area.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    area.innerHTML = "";
    toast("Draft failed: " + e.message, "error");
  }
}

async function handleComposeSend() {
  const to = (document.getElementById("compose-to") || {}).value || "";
  const cc = (document.getElementById("compose-cc") || {}).value || "";
  const subject =
    (document.getElementById("compose-subject") || {}).value || "";
  const textarea = document.getElementById("compose-draft-text");
  const body = textarea ? textarea.value.trim() : "";
  const area = document.getElementById("compose-draft-area");
  const replyToId = area ? area.dataset.replyToId : "";

  if (!to || !subject || !body) {
    toast("Missing fields", "error");
    return;
  }

  try {
    if (replyToId) {
      await api.post(`/api/emails/${replyToId}/send`, { body });
    } else {
      // Use FormData if there are attachments
      if (composeAttachments.length) {
        const fd = new FormData();
        fd.append("to", to);
        fd.append("cc", cc);
        fd.append("subject", subject);
        fd.append("body", body);
        composeAttachments.forEach((f) => fd.append("attachments", f));
        const r = await fetch("/api/compose/send", {
          method: "POST",
          body: fd,
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || r.statusText);
      } else {
        await api.post("/api/compose/send", { to, cc, subject, body });
      }
    }
    toast("Email sent!", "success");
    closeComposeFloat();
  } catch (e) {
    toast("Send failed: " + e.message, "error");
  }
}

// #10.6 + #10.7 — Custom Instructions: shows inline reply area in email detail
function showInlineReply(prefill = "") {
  const area = document.getElementById("inline-reply-area");
  if (!area || !state.selectedEmail) return;

  const email = state.selectedEmail;
  area.innerHTML = `
    <div class="inline-reply fade-in">
      <div class="inline-reply-header">
        <span class="inline-reply-title">Custom Instructions</span>
        <button class="inline-reply-close" id="btn-close-inline-reply"><span class="material-symbols-rounded" style="font-size:18px">close</span></button>
      </div>
      <div class="form-field">
        <label class="form-label">Tone</label>
        <select class="select-input select-full" id="inline-tone">
          <option value="">Auto</option>
          <option value="professional">Professional</option>
          <option value="friendly">Friendly</option>
          <option value="casual">Casual</option>
          <option value="formal">Formal</option>
        </select>
      </div>
      <div class="form-field">
        <label class="form-label">What should the reply say?</label>
        <textarea class="textarea" id="inline-instructions" rows="3" placeholder="e.g. Accept the meeting, thank them, ask about parking">${esc(prefill)}</textarea>
      </div>
      <div style="display:flex;gap:8px;margin-top:4px">
        <button class="btn btn-primary btn-sm" id="btn-inline-draft"><span class="material-symbols-rounded">edit_note</span> Generate Draft</button>
      </div>
      <div id="inline-draft-result"></div>
    </div>`;

  area.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function handleInlineDraft() {
  if (!state.selectedId) return;
  const instructions =
    (document.getElementById("inline-instructions") || {}).value || "";
  const tone = (document.getElementById("inline-tone") || {}).value || "";
  if (!instructions.trim()) {
    toast("Enter instructions for the reply", "error");
    return;
  }

  const result = document.getElementById("inline-draft-result");
  if (!result) return;
  result.innerHTML =
    '<div class="loading-center"><span class="spinner"></span> Drafting…</div>';

  try {
    const decision = tone ? `[Tone: ${tone}] ${instructions}` : instructions;
    const data = await api.post(`/api/emails/${state.selectedId}/draft`, {
      decision,
    });
    result.innerHTML = `
      <div class="draft-section fade-in">
        <h4>Draft</h4>
        <textarea class="draft-textarea" id="draft-textarea">${esc(data.draft)}</textarea>
        <div class="draft-actions">
          <button class="btn btn-primary" id="btn-send-draft"><span class="material-symbols-rounded">send</span> Send</button>
          <button class="btn btn-ghost btn-sm" id="btn-discard-draft">Discard</button>
        </div>
      </div>`;
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    result.innerHTML = "";
    toast("Draft failed: " + e.message, "error");
  }
}

// #10.7 — Quick reply action generates inline draft instead of navigating to compose
async function handleQuickReplyAction(option) {
  if (!state.selectedId || !state.selectedEmail) return;

  const draftArea = document.getElementById("draft-area");
  if (!draftArea) return;

  draftArea.innerHTML =
    '<div class="loading-center"><span class="spinner"></span> Drafting…</div>';
  draftArea.scrollIntoView({ behavior: "smooth", block: "nearest" });

  try {
    const data = await api.post(`/api/emails/${state.selectedId}/draft`, {
      decision: option.context || option.label,
    });
    draftArea.innerHTML = `
      <div class="draft-section fade-in">
        <h4>Draft — ${esc(option.label)}</h4>
        <textarea class="draft-textarea" id="draft-textarea">${esc(data.draft)}</textarea>
        <div class="draft-actions">
          <button class="btn btn-primary" id="btn-send-draft"><span class="material-symbols-rounded">send</span> Send</button>
          <button class="btn btn-ghost btn-sm" id="btn-discard-draft">Discard</button>
        </div>
      </div>`;
    draftArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    draftArea.innerHTML = "";
    toast("Draft failed: " + e.message, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// MEETING SCHEDULING (compose view)
// ═══════════════════════════════════════════════════════════════════════
async function handleScheduleMeeting() {
  const area = document.getElementById("compose-meeting-area");
  if (!area) return;
  const subject =
    (document.getElementById("compose-subject") || {}).value || "Meeting";
  const defaultAttendee =
    (document.getElementById("compose-to") || {}).value || "";

  area.innerHTML = `
    <div class="meeting-panel fade-in">
      <div class="meeting-header"><span class="material-symbols-rounded">event</span><strong>Schedule Meeting</strong></div>
      <div class="form-field">
        <label class="form-label">Attendee</label>
        <input class="input" id="meeting-attendee" value="${esc(defaultAttendee)}" placeholder="attendee@example.com">
      </div>
      <div class="form-field">
        <label class="form-label">Duration</label>
        <select class="select-input select-full" id="meeting-duration-pre">
          <option value="15">15 min</option>
          <option value="30" selected>30 min</option>
          <option value="45">45 min</option>
          <option value="60">1 hour</option>
          <option value="90">1.5 hours</option>
        </select>
      </div>
      <div style="display:flex;gap:8px;margin-top:4px">
        <button class="btn btn-primary btn-sm" id="btn-check-availability"><span class="material-symbols-rounded">search</span> Check Availability</button>
        <button class="btn btn-ghost btn-sm" id="btn-cancel-meeting-pre">Cancel</button>
      </div>
    </div>`;

  document
    .getElementById("btn-cancel-meeting-pre")
    ?.addEventListener("click", () => {
      area.innerHTML = "";
    });

  document
    .getElementById("btn-check-availability")
    ?.addEventListener("click", async () => {
      const attendee =
        (document.getElementById("meeting-attendee") || {}).value?.trim() || "";
      const duration =
        (document.getElementById("meeting-duration-pre") || {}).value || "30";

      area.innerHTML =
        '<div class="loading-center"><span class="spinner"></span> Checking…</div>';

      try {
        const url =
          `/api/compose/free-slots?duration=${encodeURIComponent(duration)}` +
          (attendee ? `&attendee=${encodeURIComponent(attendee)}` : "");
        const [slotsData, zoomData] = await Promise.all([
          api.get(url),
          api.get("/api/compose/zoom-available"),
        ]);
        const slots = slotsData.slots || [];
        const hasZoom = zoomData.available;

        if (!slots.length) {
          area.innerHTML = `<div class="meeting-panel fade-in"><p style="color:var(--on-surface-subtle);font-size:13px">No available slots found.</p><button class="btn btn-ghost btn-sm" id="btn-cancel-meeting-empty" style="margin-top:8px">Close</button></div>`;
          document
            .getElementById("btn-cancel-meeting-empty")
            ?.addEventListener("click", () => {
              area.innerHTML = "";
            });
          return;
        }

        const slotsHtml = slots
          .map(
            (s, i) =>
              `<label class="slot-option"><input type="radio" name="meeting-slot" value="${i}" ${i === 0 ? "checked" : ""}><span>${esc(s.label)}</span></label>`,
          )
          .join("");

        area.innerHTML = `
        <div class="meeting-panel fade-in">
          <div class="meeting-header"><span class="material-symbols-rounded">event</span><strong>Pick a time</strong></div>
          <div class="form-field">
            <label class="form-label">Title</label>
            <input class="input" id="meeting-title" value="${esc(subject)}" placeholder="Meeting title">
          </div>
          <div class="form-field">
            <label class="form-label">Duration</label>
            <select class="select-input select-full" id="meeting-duration">
              <option value="15" ${duration === "15" ? "selected" : ""}>15 min</option>
              <option value="30" ${duration === "30" ? "selected" : ""}>30 min</option>
              <option value="45" ${duration === "45" ? "selected" : ""}>45 min</option>
              <option value="60" ${duration === "60" ? "selected" : ""}>1 hour</option>
              <option value="90" ${duration === "90" ? "selected" : ""}>1.5 hours</option>
            </select>
          </div>
          <div class="form-field">
            <label class="form-label">Available Slots</label>
            <div class="slot-grid">${slotsHtml}</div>
          </div>
          ${hasZoom ? `<label class="zoom-option"><input type="checkbox" id="meeting-add-zoom" checked><span class="material-symbols-rounded" style="font-size:16px;color:#2D8CFF">videocam</span> Add Zoom link</label>` : ""}
          <div style="display:flex;gap:8px;margin-top:16px">
            <button class="btn btn-primary btn-sm" id="btn-confirm-meeting"><span class="material-symbols-rounded">check</span> Create</button>
            <button class="btn btn-ghost btn-sm" id="btn-cancel-meeting">Cancel</button>
          </div>
        </div>`;

        area.dataset.slotsJson = JSON.stringify(slots);
        area.dataset.attendee = attendee;

        document
          .getElementById("btn-cancel-meeting")
          ?.addEventListener("click", () => {
            area.innerHTML = "";
          });
      } catch (e) {
        area.innerHTML = "";
        toast("Could not load calendar: " + e.message, "error");
      }
    });
}

async function handleConfirmMeeting() {
  const area = document.getElementById("compose-meeting-area");
  if (!area) return;

  const slots = JSON.parse(area.dataset.slotsJson || "[]");
  const attendee = area.dataset.attendee || "";
  const selectedRadio = area.querySelector(
    'input[name="meeting-slot"]:checked',
  );
  const slot = slots[selectedRadio ? parseInt(selectedRadio.value) : 0];
  if (!slot) return;

  const title =
    (document.getElementById("meeting-title") || {}).value || "Meeting";
  const duration = parseInt(
    (document.getElementById("meeting-duration") || {}).value || "30",
  );
  const addZoom = document.getElementById("meeting-add-zoom")?.checked ?? false;

  const btn = document.getElementById("btn-confirm-meeting");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML =
      '<span class="spinner" style="width:14px;height:14px;border-width:2px"></span> Creating…';
  }

  try {
    const data = await api.post("/api/compose/create-meeting", {
      summary: title,
      start: slot.start,
      duration,
      attendees: attendee ? [attendee] : [],
      add_zoom: addZoom,
    });

    area.innerHTML = `
      <div class="meeting-panel meeting-success fade-in">
        <div class="meeting-header"><span class="material-symbols-rounded">check_circle</span><strong>Meeting Scheduled</strong></div>
        <p style="margin-top:8px;font-size:13px;color:var(--on-surface-variant)">${esc(title)} · ${esc(slot.label)}</p>
        ${data.zoom_join ? `<p style="font-size:12px;color:var(--on-surface-subtle);margin-top:4px">Zoom: <a href="${esc(data.zoom_join)}" target="_blank" rel="noopener noreferrer">${esc(data.zoom_join)}</a></p>` : ""}
        ${data.html_link ? `<a href="${esc(data.html_link)}" target="_blank" rel="noopener noreferrer" style="font-size:12px;margin-top:4px;display:inline-block">Open in Calendar ↗</a>` : ""}
      </div>`;
    toast("Meeting created" + (data.zoom_join ? " with Zoom" : ""), "success");
  } catch (e) {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML =
        '<span class="material-symbols-rounded">check</span> Create';
    }
    toast("Failed: " + e.message, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// NEW MEETING from meetings hub (#6)
// ═══════════════════════════════════════════════════════════════════════
// Meeting creation now happens directly in the meetings detail pane
function showNewMeetingForm() {
  // Already rendered in the detail pane
  const titleInput = document.getElementById("meeting-new-title");
  if (titleInput) titleInput.focus();
}

// Check availability from meetings detail pane
async function handleMeetingCheckAvailability() {
  const area = document.getElementById("meeting-slots-area");
  if (!area) return;
  const title =
    (document.getElementById("meeting-new-title") || {}).value || "Meeting";
  const attendee =
    (document.getElementById("meeting-new-attendee") || {}).value?.trim() || "";
  const duration =
    (document.getElementById("meeting-new-duration") || {}).value || "30";

  area.innerHTML =
    '<div class="loading-center"><span class="spinner"></span> Checking…</div>';

  try {
    const url =
      `/api/compose/free-slots?duration=${encodeURIComponent(duration)}` +
      (attendee ? `&attendee=${encodeURIComponent(attendee)}` : "");
    const [slotsData, zoomData] = await Promise.all([
      api.get(url),
      api.get("/api/compose/zoom-available"),
    ]);
    const slots = slotsData.slots || [];
    const hasZoom = zoomData.available;

    if (!slots.length) {
      area.innerHTML = `<p style="color:var(--on-surface-subtle);font-size:13px;padding:8px 0">No available slots found.</p>`;
      return;
    }

    const slotsHtml = slots
      .map(
        (s, i) =>
          `<label class="slot-option"><input type="radio" name="meeting-slot" value="${i}" ${i === 0 ? "checked" : ""}><span>${esc(s.label)}</span></label>`,
      )
      .join("");

    area.innerHTML = `
      <div class="form-field">
        <label class="form-label">Available Slots</label>
        <div class="slot-grid">${slotsHtml}</div>
      </div>
      ${hasZoom ? `<label class="zoom-option"><input type="checkbox" id="meeting-add-zoom" checked><span class="material-symbols-rounded" style="font-size:16px;color:#2D8CFF">videocam</span> Add Zoom link</label>` : ""}
      <div style="display:flex;gap:8px;margin-top:12px">
        <button class="btn btn-primary btn-sm" id="btn-meeting-confirm"><span class="material-symbols-rounded">check</span> Create</button>
        <button class="btn btn-ghost btn-sm" id="btn-meeting-cancel-slots">Cancel</button>
      </div>`;

    area.dataset.slotsJson = JSON.stringify(slots);
    area.dataset.attendee = attendee;
    area.dataset.title = title;
    area.dataset.duration = duration;
  } catch (e) {
    area.innerHTML = "";
    toast("Could not load calendar: " + e.message, "error");
  }
}

async function handleMeetingConfirmFromDetail() {
  const area = document.getElementById("meeting-slots-area");
  if (!area) return;

  const slots = JSON.parse(area.dataset.slotsJson || "[]");
  const attendee = area.dataset.attendee || "";
  const title = area.dataset.title || "Meeting";
  const duration = parseInt(area.dataset.duration || "30");
  const selectedRadio = area.querySelector(
    'input[name="meeting-slot"]:checked',
  );
  const slot = slots[selectedRadio ? parseInt(selectedRadio.value) : 0];
  if (!slot) return;

  const addZoom = document.getElementById("meeting-add-zoom")?.checked ?? false;

  const btn = document.getElementById("btn-meeting-confirm");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML =
      '<span class="spinner" style="width:14px;height:14px;border-width:2px"></span> Creating…';
  }

  try {
    const data = await api.post("/api/compose/create-meeting", {
      summary: title,
      start: slot.start,
      duration,
      attendees: attendee ? [attendee] : [],
      add_zoom: addZoom,
    });

    area.innerHTML = `
      <div class="meeting-panel meeting-success fade-in" style="margin-top:8px">
        <div class="meeting-header"><span class="material-symbols-rounded">check_circle</span><strong>Meeting Scheduled</strong></div>
        <p style="margin-top:8px;font-size:13px;color:var(--on-surface-variant)">${esc(title)} · ${esc(slot.label)}</p>
        ${data.zoom_join ? `<p style="font-size:12px;color:var(--on-surface-subtle);margin-top:4px">Zoom: <a href="${esc(data.zoom_join)}" target="_blank" rel="noopener noreferrer">${esc(data.zoom_join)}</a></p>` : ""}
        ${data.html_link ? `<a href="${esc(data.html_link)}" target="_blank" rel="noopener noreferrer" style="font-size:12px;margin-top:4px;display:inline-block">Open in Calendar ↗</a>` : ""}
      </div>`;
    toast("Meeting created" + (data.zoom_join ? " with Zoom" : ""), "success");
    await loadMeetings();
    renderListPane();
  } catch (e) {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML =
        '<span class="material-symbols-rounded">check</span> Create';
    }
    toast("Failed: " + e.message, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// SNOOZE DROPDOWN
// ═══════════════════════════════════════════════════════════════════════

// Reschedule meeting — shows availability picker
async function handleRescheduleMeeting() {
  const m = state.selectedMeeting;
  if (!m) return;
  const area = document.getElementById("reschedule-area");
  if (!area) return;

  const allAttendees = (m.attendees || [])
    .map((a) => a.email || a)
    .filter(Boolean);
  const attendeeParam = allAttendees.length
    ? `&attendee=${encodeURIComponent(allAttendees.join(","))}`
    : "";
  area.innerHTML =
    '<div class="loading-center"><span class="spinner"></span> Checking availability for all attendees…</div>';

  try {
    const [slotsData, zoomData] = await Promise.all([
      api.get(`/api/compose/free-slots?duration=30${attendeeParam}`),
      api.get("/api/compose/zoom-available"),
    ]);
    const slots = slotsData.slots || [];
    const hasZoom = zoomData.available;
    if (!slots.length) {
      area.innerHTML =
        '<p style="font-size:13px;color:var(--on-surface-subtle);padding:8px 0">No available slots found.</p>';
      return;
    }
    const slotsHtml = slots
      .map(
        (s, i) =>
          `<label class="slot-option"><input type="radio" name="reschedule-slot" value="${i}" ${i === 0 ? "checked" : ""}><span>${esc(s.label)}</span></label>`,
      )
      .join("");

    area.innerHTML = `
      <div class="meeting-form" style="margin-top:8px">
        <div class="form-field">
          <label class="form-label">New Time</label>
          <div class="slot-grid">${slotsHtml}</div>
        </div>
        ${hasZoom ? `<label class="zoom-option"><input type="checkbox" id="reschedule-zoom" checked><span class="material-symbols-rounded" style="font-size:16px;color:#2D8CFF">videocam</span> Add Zoom link</label>` : ""}
        <div style="display:flex;gap:8px;margin-top:12px">
          <button class="btn btn-primary btn-sm" id="btn-reschedule-confirm"><span class="material-symbols-rounded">check</span> Reschedule</button>
          <button class="btn btn-ghost btn-sm" id="btn-reschedule-cancel">Cancel</button>
        </div>
      </div>`;
    area.dataset.slotsJson = JSON.stringify(slots);
    area.dataset.attendee = allAttendees.join(",");
  } catch (e) {
    area.innerHTML = "";
    toast("Could not load calendar: " + e.message, "error");
  }
}

async function handleRescheduleConfirm() {
  const m = state.selectedMeeting;
  if (!m) return;
  const area = document.getElementById("reschedule-area");
  if (!area) return;

  const slots = JSON.parse(area.dataset.slotsJson || "[]");
  const attendee = area.dataset.attendee || "";
  const selectedRadio = area.querySelector(
    'input[name="reschedule-slot"]:checked',
  );
  const slot = slots[selectedRadio ? parseInt(selectedRadio.value) : 0];
  if (!slot) return;

  const addZoom = document.getElementById("reschedule-zoom")?.checked ?? false;

  try {
    // Delete old meeting first
    await api.del(`/api/meetings/${encodeURIComponent(m.id)}`);
    // Create new one with all original attendees
    const allAttendeeEmails = (m.attendees || [])
      .map((a) => a.email || a)
      .filter(Boolean);
    const data = await api.post("/api/compose/create-meeting", {
      summary: m.summary || "Meeting",
      start: slot.start,
      duration: 30,
      attendees: allAttendeeEmails,
      add_zoom: addZoom,
    });
    toast("Meeting rescheduled", "success");
    state.selectedMeeting = null;
    await loadMeetings();
    renderListPane();
    renderDetailPane();
  } catch (e) {
    toast("Reschedule failed: " + e.message, "error");
  }
}
function showSnoozeMenu() {
  document.querySelectorAll(".snooze-menu").forEach((d) => d.remove());
  const btn = document.getElementById("btn-snooze");
  if (!btn) return;

  const menu = document.createElement("div");
  menu.className = "snooze-menu";
  const options = [
    { label: "In 1 hour", hours: 1 },
    { label: "In 3 hours", hours: 3 },
    { label: "Tomorrow morning", hours: 16 },
    { label: "Next week", hours: 168 },
  ];
  menu.innerHTML = options
    .map(
      (o) =>
        `<button class="snooze-opt" data-hours="${o.hours}">${o.label}</button>`,
    )
    .join("");
  btn.appendChild(menu);

  menu.addEventListener("click", (e) => {
    const opt = e.target.closest(".snooze-opt");
    if (opt) {
      menu.remove();
      handleSnooze(parseInt(opt.dataset.hours));
    }
  });

  setTimeout(() => {
    const close = (e) => {
      if (!menu.contains(e.target) && e.target !== btn) {
        menu.remove();
        document.removeEventListener("click", close);
      }
    };
    document.addEventListener("click", close);
  }, 10);
}

// ═══════════════════════════════════════════════════════════════════════
// CATEGORY DROPDOWN (#10.3 — inline dropdown on badge click)
// ═══════════════════════════════════════════════════════════════════════
function showCategoryDropdown() {
  const badges = document.getElementById("detail-badges");
  if (!badges || !state.selectedEmail) return;

  // Close if already open
  const existing = badges.querySelector(".cat-dropdown");
  if (existing) {
    existing.remove();
    return;
  }

  const email = state.selectedEmail;
  const curCat = email.category || "informational";
  const mainSlugs = [
    "important",
    "informational",
    "newsletter",
    "action-required",
  ];
  const supplementary = state.categories.filter(
    (c) => !mainSlugs.includes(c.slug),
  );
  const curMain = curCat.split(",")[0];
  const curTags = curCat.split(",").slice(1);

  const mainItems = mainSlugs
    .map((s) => {
      const c = state.categories.find((x) => x.slug === s);
      const name = c ? c.display_name : s;
      const color = getCatColor(s);
      const active = curMain === s;
      return `<div class="cat-dropdown-item ${active ? "active" : ""}" data-cat-main="${esc(s)}">
      <span class="cat-dropdown-dot" style="background:${color}"></span>
      <span>${esc(name)}</span>
      ${active ? '<span class="material-symbols-rounded cat-dropdown-check">check</span>' : ""}
    </div>`;
    })
    .join("");

  const tagItems = supplementary
    .map((c) => {
      const active = curTags.includes(c.slug);
      return `<div class="cat-dropdown-item ${active ? "active" : ""}" data-cat-tag="${esc(c.slug)}">
      <span class="cat-dropdown-dot" style="background:${c.color}"></span>
      <span>${esc(c.display_name)}</span>
      ${active ? '<span class="material-symbols-rounded cat-dropdown-check">check</span>' : ""}
    </div>`;
    })
    .join("");

  const dropdown = document.createElement("div");
  dropdown.className = "cat-dropdown fade-in";
  dropdown.innerHTML = `
    <div class="cat-dropdown-label">Category</div>
    ${mainItems}
    ${tagItems ? `<div class="cat-dropdown-label" style="margin-top:6px;border-top:1px solid var(--divider);padding-top:6px">Tags</div>${tagItems}` : ""}`;

  badges.appendChild(dropdown);

  dropdown.addEventListener("click", async (e) => {
    const mainItem = e.target.closest("[data-cat-main]");
    const tagItem = e.target.closest("[data-cat-tag]");

    if (mainItem) {
      const mainVal = mainItem.dataset.catMain;
      const suppChecks = badges.querySelectorAll(
        ".cat-dropdown-item[data-cat-tag].active",
      );
      const suppTags = Array.from(suppChecks).map((el) => el.dataset.catTag);
      const newCat = [mainVal, ...suppTags].join(",");
      await saveCategory(newCat);
      dropdown.remove();
    }

    if (tagItem) {
      const slug = tagItem.dataset.catTag;
      tagItem.classList.toggle("active");
      const mainActive = badges.querySelector(
        ".cat-dropdown-item[data-cat-main].active",
      );
      const mainVal = mainActive ? mainActive.dataset.catMain : curMain;
      const activeTags = Array.from(
        badges.querySelectorAll(".cat-dropdown-item[data-cat-tag].active"),
      ).map((el) => el.dataset.catTag);
      const newCat = [mainVal, ...activeTags].join(",");
      await saveCategory(newCat);
    }
  });

  // Close on outside click
  setTimeout(() => {
    const close = (e) => {
      if (
        !dropdown.contains(e.target) &&
        !e.target.closest("#cat-badge-click")
      ) {
        dropdown.remove();
        document.removeEventListener("click", close);
      }
    };
    document.addEventListener("click", close);
  }, 10);
}

async function saveCategory(newCat) {
  try {
    await api.put(`/api/emails/${state.selectedId}/category`, {
      category: newCat,
    });
    if (state.selectedEmail) state.selectedEmail.category = newCat;
    const listItem = state.emails.find((e) => e.id === state.selectedId);
    if (listItem) listItem.category = newCat;
    toast("Category updated", "success");
    await loadEmails();
    renderDetailPane();
  } catch (e) {
    toast(e.message, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════════════

// ── Sidebar nav ──────────────────────────────────────────────────────
document.getElementById("sidebar-nav").addEventListener("click", (e) => {
  const item = e.target.closest(".nav-item");
  if (!item) return;
  navigateTo(item.dataset.view);
  closeSidebar();
});

// ── Topbar buttons ───────────────────────────────────────────────────
document.getElementById("btn-menu").addEventListener("click", toggleSidebar);
document.getElementById("btn-compose").addEventListener("click", () => {
  showCompose();
  closeSidebar();
});
document
  .getElementById("btn-refresh")
  .addEventListener("click", () => fetchNewEmails());
document.getElementById("btn-theme").addEventListener("click", toggleTheme);
document.getElementById("btn-auth").addEventListener("click", handleAuth);

// ── Compose float controls ───────────────────────────────────────────
document
  .getElementById("compose-float-close")
  .addEventListener("click", closeComposeFloat);
document
  .getElementById("compose-float-minimize")
  .addEventListener("click", () => {
    const el = document.getElementById("compose-float");
    el.classList.toggle("minimized");
    el.classList.remove("expanded");
  });
document
  .getElementById("compose-float-expand")
  .addEventListener("click", () => {
    const el = document.getElementById("compose-float");
    el.classList.toggle("expanded");
    el.classList.remove("minimized");
  });
document
  .getElementById("compose-float-header")
  .addEventListener("click", (e) => {
    if (e.target.closest(".compose-float-btn")) return;
    const el = document.getElementById("compose-float");
    if (el.classList.contains("minimized")) el.classList.remove("minimized");
  });

// ── Compose float body delegated clicks ──────────────────────────────
document.getElementById("compose-float-body").addEventListener("click", (e) => {
  if (e.target.closest("#btn-compose-draft")) {
    handleComposeDraft();
    return;
  }
  if (e.target.closest("#btn-compose-send")) {
    handleComposeSend();
    return;
  }
  if (e.target.closest("#btn-compose-clear")) {
    state.composeContext = null;
    openComposeFloat();
    return;
  }
});

document.getElementById("llm-select").addEventListener("change", async (e) => {
  try {
    await api.put("/api/settings/llm", { provider: e.target.value });
    state.llmProvider = e.target.value;
    toast("AI engine updated", "success");
  } catch (err) {
    toast(err.message, "error");
  }
});

// ── Search ───────────────────────────────────────────────────────────
const searchInput = document.getElementById("search-input");
const searchClear = document.getElementById("search-clear");

searchInput.addEventListener(
  "input",
  debounce(() => {
    state.searchQuery = searchInput.value.trim();
    state.currentPage = 1;
    searchClear.classList.toggle("hidden", !state.searchQuery);
    if (state.view === "inbox") loadEmails();
  }, 300),
);

searchClear.addEventListener("click", () => {
  searchInput.value = "";
  state.searchQuery = "";
  state.currentPage = 1;
  searchClear.classList.add("hidden");
  if (state.view === "inbox") loadEmails();
});

// ── List pane delegated clicks ───────────────────────────────────────
document.getElementById("list-pane").addEventListener("click", async (e) => {
  // Email row
  const row = e.target.closest(".email-row");
  if (row) {
    selectEmail(row.dataset.emailId);
    return;
  }

  // Filter dropdown toggle
  const filterToggle = e.target.closest("#filter-toggle");
  if (filterToggle) {
    const menu = document.getElementById("filter-menu");
    if (menu) menu.classList.toggle("open");
    return;
  }

  // Filter option checkbox
  const filterOpt = e.target.closest(".filter-option input");
  if (filterOpt) {
    const slug = filterOpt.dataset.filter;
    if (slug === "__unread") {
      const idx = state.activeFilters.indexOf("__unread");
      if (idx >= 0) state.activeFilters.splice(idx, 1);
      else state.activeFilters.push("__unread");
      renderListPane();
      return;
    }
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

  // Todo check
  const todoCheck = e.target.closest(".todo-check");
  if (todoCheck) {
    const id = parseInt(todoCheck.dataset.todoId);
    try {
      await api.post(`/api/todos/${id}/done`);
      toast("Task completed", "success");
      state.todos = state.todos.filter((t) => t.id !== id);
      renderListPane();
      renderDetailPane();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }

  // Label delete (still in list pane)
  const catDel = e.target.closest("[data-cat-delete]");
  if (catDel) {
    const slug = catDel.dataset.catDelete;
    if (!confirm(`Delete label "${getCatName(slug)}"?`)) return;
    try {
      await api.del(`/api/categories/${slug}`);
      toast("Label deleted", "success");
      await Promise.all([loadCategories(), loadEmails()]);
      renderListPane();
      renderDetailPane();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }

  // Delete meeting (in list pane)
  const delMeeting = e.target.closest("[data-delete-meeting]");
  if (delMeeting) {
    e.stopPropagation();
    const eventId = delMeeting.dataset.deleteMeeting;
    if (!confirm("Delete this meeting?")) return;
    try {
      await api.del(`/api/meetings/${encodeURIComponent(eventId)}`);
      toast("Meeting deleted", "success");
      state.selectedMeeting = null;
      await loadMeetings();
      renderListPane();
      renderDetailPane();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }

  // Select meeting (in list pane)
  const meetingCard = e.target.closest(".meeting-card");
  if (meetingCard && state.view === "meetings") {
    const eventId = meetingCard.dataset.meetingId;
    const m = state.meetings.find((m) => (m.id || "") === eventId);
    if (m) {
      state.selectedMeeting = m;
      renderDetailPane();
    }
    return;
  }
});

// ── Detail pane delegated clicks ─────────────────────────────────────
document.getElementById("detail-pane").addEventListener("click", async (e) => {
  // Back button (mobile)
  if (e.target.closest("#btn-back")) {
    goBackToList();
    return;
  }

  // #10.1 — toolbar actions now in action-bar
  if (e.target.closest("#btn-archive")) {
    handleArchive();
    return;
  }
  if (e.target.closest("#btn-snooze")) {
    showSnoozeMenu();
    return;
  }
  if (e.target.closest("#btn-mark-unread")) {
    handleMarkUnread();
    return;
  }

  // #10.3 — category badge click opens dropdown
  if (e.target.closest("#cat-badge-click")) {
    showCategoryDropdown();
    return;
  }

  // Forward
  if (e.target.closest("#btn-forward") && state.selectedEmail) {
    const email = state.selectedEmail;
    let subj = email.subject || "";
    if (!subj.toLowerCase().startsWith("fwd:")) subj = "Fwd: " + subj;
    showCompose({
      to: "",
      subject: subj,
      context: `---------- Forwarded message ----------\nFrom: ${email.sender}\nDate: ${email.date}\nSubject: ${email.subject}\n\n${(email.body || "").slice(0, 2000)}`,
    });
    return;
  }

  // #10.6 — Custom Instructions button
  if (e.target.closest("#btn-custom-instructions") && state.selectedEmail) {
    showInlineReply();
    return;
  }

  // Rethink
  if (e.target.closest("#btn-rethink")) {
    handleRethink();
    return;
  }

  // Send draft (inline)
  if (e.target.closest("#btn-send-draft")) {
    handleSendDraft();
    return;
  }
  if (e.target.closest("#btn-discard-draft")) {
    state.draft = null;
    const area = document.getElementById("inline-reply-area");
    if (area) area.innerHTML = "";
    const draftArea = document.getElementById("draft-area");
    if (draftArea) draftArea.innerHTML = "";
    return;
  }

  // Inline reply draft generate
  if (e.target.closest("#btn-inline-draft")) {
    handleInlineDraft();
    return;
  }

  // Close inline reply
  if (e.target.closest("#btn-close-inline-reply")) {
    const area = document.getElementById("inline-reply-area");
    if (area) area.innerHTML = "";
    return;
  }

  // #10.7 — Quick action chips: reply generates inline, meeting goes to meetings tab
  const actionChip = e.target.closest(".action-chip");
  if (actionChip && state.selectedEmail) {
    const type = actionChip.dataset.actionType;
    const idx = parseInt(actionChip.dataset.actionIdx);
    const opts = state.selectedEmail.decision_options || [];
    const typeOpts = opts.filter((o) => o.type === type);
    const option = typeOpts[idx];
    if (!option) return;

    if (type === "reply") {
      // #10.7 — generate draft inline instead of navigating to compose
      handleQuickReplyAction(option);
    } else if (type === "meeting") {
      // #10.5 — route to meetings tab instead of showing meeting in email
      navigateTo("meetings");
      toast("Use the Meetings hub to schedule", "info");
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

  // Compose actions — now handled in floating compose body
  if (e.target.closest("#btn-compose-draft")) {
    handleComposeDraft();
    return;
  }
  if (e.target.closest("#btn-compose-send")) {
    handleComposeSend();
    return;
  }
  if (e.target.closest("#btn-compose-clear")) {
    state.composeContext = null;
    openComposeFloat();
    return;
  }
  if (e.target.closest("#btn-schedule-meeting")) {
    handleScheduleMeeting();
    return;
  }
  if (e.target.closest("#btn-confirm-meeting")) {
    handleConfirmMeeting();
    return;
  }

  // ── Meeting detail pane buttons ──────────────────────────────────
  if (e.target.closest("#btn-meeting-check-avail")) {
    handleMeetingCheckAvailability();
    return;
  }
  if (e.target.closest("#btn-meeting-refresh")) {
    await loadMeetings();
    renderListPane();
    renderDetailPane();
    toast("Meetings refreshed", "success");
    return;
  }
  if (e.target.closest("#btn-meeting-confirm")) {
    handleMeetingConfirmFromDetail();
    return;
  }
  if (e.target.closest("#btn-meeting-cancel-slots")) {
    const area = document.getElementById("meeting-slots-area");
    if (area) area.innerHTML = "";
    return;
  }

  // ── Meeting detail selected meeting buttons ──────────────────────
  if (e.target.closest("#btn-meeting-back")) {
    state.selectedMeeting = null;
    renderDetailPane();
    return;
  }
  if (e.target.closest("#btn-delete-selected-meeting")) {
    const eventId = e.target.closest("#btn-delete-selected-meeting").dataset
      .eventId;
    if (!confirm("Delete this meeting?")) return;
    try {
      await api.del(`/api/meetings/${encodeURIComponent(eventId)}`);
      toast("Meeting deleted", "success");
      state.selectedMeeting = null;
      await loadMeetings();
      renderListPane();
      renderDetailPane();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }
  if (e.target.closest("#btn-reschedule-meeting")) {
    handleRescheduleMeeting();
    return;
  }
  if (e.target.closest("#btn-reschedule-confirm")) {
    handleRescheduleConfirm();
    return;
  }
  if (e.target.closest("#btn-reschedule-cancel")) {
    const area = document.getElementById("reschedule-area");
    if (area) area.innerHTML = "";
    return;
  }

  // ── Todo detail pane buttons ─────────────────────────────────────
  if (e.target.closest("#btn-todo-manual-add")) {
    const inp = document.getElementById("todo-manual-input");
    const task = inp ? inp.value.trim() : "";
    if (!task) {
      toast("Enter a task", "error");
      return;
    }
    try {
      await api.post("/api/todos", { task, email_id: "" });
      toast("Task added", "success");
      inp.value = "";
      await loadTodos();
      renderListPane();
      renderDetailPane();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }
  const sugAddBtn = e.target.closest(".todo-suggestion-add");
  if (sugAddBtn) {
    const label = sugAddBtn.dataset.sugLabel;
    const emailId = sugAddBtn.dataset.sugEmail;
    try {
      await api.post("/api/todos", { task: label, email_id: emailId });
      toast("Task added", "success");
      await loadTodos();
      renderListPane();
      renderDetailPane();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }

  // ── Labels detail pane buttons ───────────────────────────────────
  if (e.target.closest("#btn-create-label")) {
    const nameEl = document.getElementById("new-label-name");
    const descEl = document.getElementById("new-label-desc");
    const name = nameEl ? nameEl.value.trim() : "";
    if (!name) {
      toast("Enter a label name", "error");
      return;
    }
    const color = autoColor(name);
    const description = descEl ? descEl.value.trim() : "";
    try {
      await api.post("/api/categories", { name, color, description });
      toast("Label created", "success");
      await Promise.all([loadCategories(), loadEmails()]);
      renderListPane();
      renderDetailPane();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }

  // AI suggestions load (in labels detail)
  if (e.target.closest("#btn-load-suggestions")) {
    const btn = e.target.closest("#btn-load-suggestions");
    btn.disabled = true;
    btn.innerHTML =
      '<span class="spinner" style="width:14px;height:14px"></span> Loading…';
    try {
      const data = await api.get("/api/categories/suggest");
      state.categorySuggestions = data.suggestions || [];
      renderDetailPane();
    } catch (err) {
      toast("Could not load suggestions: " + err.message, "error");
      state.categorySuggestions = null;
      renderDetailPane();
    }
    return;
  }

  // Add suggested category (in labels detail)
  const suggestBtn = e.target.closest("[data-suggest-idx]");
  if (suggestBtn) {
    const idx = parseInt(suggestBtn.dataset.suggestIdx);
    const suggestion = (state.categorySuggestions || [])[idx];
    if (!suggestion) return;
    try {
      await api.post("/api/categories", {
        name: suggestion.proposed_name,
        color: autoColor(suggestion.proposed_name),
        description: suggestion.reason || "",
      });
      toast(`Label "${suggestion.proposed_name}" added`, "success");
      state.categorySuggestions = state.categorySuggestions.filter(
        (_, i) => i !== idx,
      );
      await Promise.all([loadCategories(), loadEmails()]);
      renderListPane();
      renderDetailPane();
    } catch (err) {
      toast(err.message, "error");
    }
    return;
  }
});

// ═══════════════════════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS
// ═══════════════════════════════════════════════════════════════════════
document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;

  // Modifier combos first
  if (
    (e.ctrlKey || e.metaKey) &&
    e.shiftKey &&
    (e.key === "K" || e.key === "k")
  ) {
    e.preventDefault();
    fetchNewEmails(true);
    return;
  }

  if (e.key === "/") {
    e.preventDefault();
    searchInput.focus();
    return;
  }
  if (e.key === "n") {
    showCompose();
    return;
  }
  if (e.key === "r" && state.selectedId) {
    handleRethink();
    return;
  }
  if (e.key === "e" && state.selectedId) {
    handleArchive();
    return;
  }

  if (e.key === "j" || e.key === "k") {
    const ids = state.emails.map((em) => em.id);
    const cur = ids.indexOf(state.selectedId);
    let next = e.key === "j" ? cur + 1 : cur - 1;
    if (next < 0) next = 0;
    if (next >= ids.length) next = ids.length - 1;
    if (ids[next]) selectEmail(ids[next]);
    return;
  }

  if (e.key === "Escape") {
    // Close compose float if open
    const floatEl = document.getElementById("compose-float");
    if (floatEl && !floatEl.classList.contains("hidden")) {
      closeComposeFloat();
      return;
    }
    state.selectedId = null;
    state.selectedEmail = null;
    state.draft = null;
    state.detailMode = "empty";
    state.composeContext = null;
    goBackToList();
    renderListPane();
    renderDetailPane();
    return;
  }
});

// ═══════════════════════════════════════════════════════════════════════
// AUTH
// ═══════════════════════════════════════════════════════════════════════
function updateAuthButton(signedIn) {
  const btn = document.getElementById("btn-auth");
  const label = document.getElementById("auth-label");
  if (!btn || !label) return;
  label.textContent = signedIn ? "Sign out" : "Sign in";
  btn.title = signedIn ? "Sign out of Google" : "Sign in with Google";
  btn.dataset.signedIn = signedIn ? "1" : "0";
}

async function handleAuth() {
  const btn = document.getElementById("btn-auth");
  const signedIn = btn && btn.dataset.signedIn === "1";

  if (signedIn) {
    if (!confirm("Sign out of Google?")) return;
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

// ═══════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════
async function init() {
  applyTheme();
  applySidebarState();
  renderListPane();
  renderDetailPane();

  await Promise.all([
    loadEmails(),
    loadCategories(),
    loadTodos(),
    loadContacts(),
  ]);
  renderListPane();
  updateFetchInfo();

  // #7 — load persisted settings including last_fetch_ts
  try {
    const settings = await api.get("/api/settings");
    state.llmProvider = settings.llm_provider;
    document.getElementById("llm-select").value = settings.llm_provider;
    // If last_fetch_ts came from settings (persisted across restarts)
    if (settings.last_fetch_ts && !state.lastFetchTs) {
      state.lastFetchTs = settings.last_fetch_ts;
      updateFetchInfo();
    }
  } catch (_) {}

  try {
    const auth = await api.get("/api/auth/status");
    updateAuthButton(auth.signed_in);
  } catch (_) {}

  setInterval(updateFetchInfo, 30000);

  // Close filter dropdown on outside click
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".filter-dropdown")) {
      const menu = document.getElementById("filter-menu");
      if (menu) menu.classList.remove("open");
    }
  });
}

init();
