// The demo page. Talks only to server.py; every action is a memory tool call on the server,
// so the page can't bypass any memory rule. No framework: plain DOM + Cytoscape for the graph.

const $ = (id) => document.getElementById(id);
const state = { user: null, org: null, agents: {}, chat: null, cy: null, busy: false, key: null, serverKey: false };

// Small helpers
function saveLocal(key, value) {          // per-browser convenience only; the page works without it
  try { value === undefined ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch (e) {}
}
function recallLocal(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
// The API key is kept in this tab only (sessionStorage) and is gone when the tab closes.
function saveSession(key, value) {
  try { value === undefined ? sessionStorage.removeItem(key) : sessionStorage.setItem(key, value); } catch (e) {}
}
function recallSession(key) { try { return sessionStorage.getItem(key); } catch (e) { return null; } }
state.key = recallSession("apikey");

async function api(method, path, body) {
  const headers = { "Content-Type": "application/json", "X-User": state.user || "", "X-Org": state.org || "" };
  if (state.key) headers["X-Api-Key"] = state.key;
  const res = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

const roleOf = (agentId) => (agentId || "").split("-").pop();
const short = (s, n = 60) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");
const mine = (agentId) => Object.values(state.agents).includes(agentId);

// Entry
async function showEntry() {
  const { orgs } = await api("GET", "/api/orgs");
  $("entry-org").replaceChildren(...orgs.map((o) => el("option", { value: o }, o)));
  $("entry-name").value = state.user || "";
  if (state.org) $("entry-org").value = state.org;
  $("entry").classList.remove("hidden");
  $("entry-name").focus();
}

$("entry-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  state.user = $("entry-name").value.trim();
  state.org = $("entry-org").value;
  try {
    await enter();
    saveLocal("user", state.user); saveLocal("org", state.org);
    $("entry").classList.add("hidden");
  } catch (err) { alert(err.message); }
});

$("switch").addEventListener("click", () => { state.chat = null; showEntry(); });

async function enter() {
  const out = await api("POST", "/api/enter");
  state.agents = out.agents;
  $("who").textContent = `${out.user} · ${out.org}`;
  state.chat = recallLocal(`chat:${state.org}:${state.user}`);
  if (state.chat && !out.chats.some((c) => c.id === state.chat)) state.chat = null;
  if (!state.chat && out.chats.length) state.chat = out.chats[0].id;
  await Promise.all([renderChats(out.chats), refreshPanels()]);
  await openChat(state.chat);
}

// Chats
async function renderChats(chats) {
  chats = chats || (await api("GET", "/api/chats")).chats;
  $("chats").replaceChildren(...chats.map((c) => el("li", {
    class: c.id === state.chat ? "current" : "",
    title: c.title || "(empty chat)",
    onclick: () => openChat(c.id),
  }, c.title || "(empty chat)")));
  if (!chats.length) $("chats").append(el("li", { class: "muted" }, "No chats yet"));
}

$("new-chat").addEventListener("click", async () => {
  const { id } = await api("POST", "/api/chats");
  await openChat(id);
  await renderChats();
});

async function openChat(chat) {
  state.chat = chat;
  $("input").disabled = $("send").disabled = !chat;
  if (!chat) {
    $("messages").replaceChildren(el("p", { class: "muted center" }, "Start a new chat, or open one on the left."));
    return;
  }
  saveLocal(`chat:${state.org}:${state.user}`, chat);
  const { messages } = await api("GET", `/api/chats/${chat}`);
  $("messages").replaceChildren(...renderChat(messages));
  if (!messages.length) $("messages").append(el("p", { class: "muted center" }, "Ask your agents anything. The Researcher answers first and passes work on when needed."));
  await renderChats();
  scrollDown();
  $("input").focus();
}

// The chat reads like one chatbot: per turn (a user message up to the next one), only the last
// agent reply is a bubble. Everything else in the turn (each agent's tool calls, saved by the
// server as a "tool" message, and in-between replies and handoff notes) goes into one collapsed
// "which agents did what" panel at the bottom of the turn.
function renderChat(messages) {
  const out = [];
  let turn = [];
  const flush = () => {
    if (!turn.length) return;
    const replies = turn.filter((m) => m.role === "agent" && !m.text.startsWith("Handoff from the "));
    const answer = replies[replies.length - 1];
    if (answer) out.push(el("div", { class: "msg agent" }, el("span", { class: "from" }, "assistant"), answer.text));
    const steps = turn.filter((m) => m !== answer);
    if (steps.length) out.push(renderSteps(steps));
    turn = [];
  };
  for (const m of messages) {
    if (m.role === "user") { flush(); out.push(el("div", { class: "msg user" }, m.text)); }
    else turn.push(m);
  }
  flush();
  return out;
}

function renderSteps(steps) {
  const runs = [];
  const body = [];
  for (const m of steps) {
    if (m.role === "tool") {
      let a = {};
      try { a = JSON.parse(m.text); } catch (e) { continue; }
      if (!a.calls) continue;                       // older records had another shape
      runs.push(a);
      body.push(el("div", { class: "step" }, el("strong", {}, roleOf(m.agent)), ` (${a.stop})`,
        ...(a.web_search ? [el("div", { class: "call" }, "web_search (on Anthropic's servers)")] : []),
        ...(a.calls.length ? a.calls.map((c) => el("div", { class: "call" + (c.ok ? "" : " err") },
          `${c.tool}(${JSON.stringify(c.args)}) → ${c.result}`)) : [el("div", { class: "call" }, "no tools used")])));
    } else {
      const handoff = m.text.startsWith("Handoff from the ");
      body.push(el("div", { class: "step" }, el("strong", {}, roleOf(m.agent)),
        handoff ? " → handoff" : " said", el("div", { class: "call" }, m.text)));
    }
  }
  const calls = runs.reduce((n, a) => n + a.calls.length, 0);
  return el("details", { class: "steps" },
    el("summary", {}, `Which agents did what (${runs.length} agent run(s), ${calls} tool call(s))`), ...body);
}

$("send-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("input").value.trim();
  if (!text || state.busy || !state.chat) return;
  if (!state.key && !state.serverKey) {
    renderKey("Paste your Anthropic API key here first.", true);
    $("key-input").focus();
    return;
  }
  state.busy = true; $("send").disabled = true; $("input").value = "";
  const empty = $("messages").querySelector(".center");
  if (empty) empty.remove();
  $("messages").append(el("div", { class: "msg user" }, text));
  const pending = el("div", { class: "msg agent pending" }, "The agents are working…");
  $("messages").append(pending); scrollDown();
  try {
    const out = await api("POST", `/api/chats/${state.chat}/messages`, { text });
    const { messages } = await api("GET", `/api/chats/${state.chat}`);
    $("messages").replaceChildren(...renderChat(messages));
    if (out.stop !== "done") $("messages").append(el("p", { class: "small muted" }, `(stopped: ${out.stop})`));
  } catch (err) {
    pending.textContent = "Error: " + err.message;
    if (/API key/i.test(err.message)) renderKey(err.message, true);
  } finally {
    state.busy = false; $("send").disabled = false;
    scrollDown();
    await Promise.all([renderChats(), refreshPanels()]);
  }
});

$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("send-form").requestSubmit(); }
});

function scrollDown() { $("messages").scrollTop = $("messages").scrollHeight; }

// Side panels
async function refreshPanels() {
  const [flagged, status] = await Promise.all([api("GET", "/api/flagged"), api("GET", "/api/status")]);

  const items = Object.entries(flagged.flagged).flatMap(([role, list]) => list.map((i) => ({ ...i, role })));
  $("flagged").replaceChildren(...items.map((i) => el("li", {
    class: i.wrong.length ? "wrong" : "changed",
    title: i.id,
  }, `${i.role}: ${short(i.text, 70)} — ${i.wrong.length ? "built on a wrong fact: redo" : "a source changed: review"}`)));
  $("flag-count").textContent = items.length;
  $("flag-count").classList.toggle("hidden", !items.length);
  $("status").textContent = `${status.mode} · $${status.spend.toFixed(4)} / $${status.cap.toFixed(2)}`;
  if (state.serverKey !== status.server_key) { state.serverKey = status.server_key; renderKey(); }
  await drawGraph();
}

// Graph
const COLORS = { active: "#2f9e44", flagged: "#e03131", retracted: "#868e96", superseded: "#ced4da" };
const SHAPES = { Fact: "ellipse", Decision: "diamond", Procedure: "hexagon", Entity: "round-rectangle",
  Agent: "triangle", Message: "tag" };

// The graph panel can be hidden (it takes a lot of room). Open by default on wide screens, closed on
// small ones; the choice is remembered in this browser. While hidden, nothing is fetched.
function graphOpen() { return !$("main").classList.contains("graph-closed"); }

function setGraphOpen(open) {
  $("main").classList.toggle("graph-closed", !open);
  saveLocal("graph", open ? "open" : "closed");
  if (open) drawGraph().then(() => { state.cy.resize(); state.cy.fit(undefined, 20); });
}

$("open-graph").addEventListener("click", () => setGraphOpen(true));
$("close-graph").addEventListener("click", () => setGraphOpen(false));
{
  const saved = recallLocal("graph");
  $("main").classList.toggle("graph-closed",
    saved ? saved === "closed" : window.matchMedia("(max-width: 900px)").matches);
}

// The handle on the graph panel's left edge sets its width. The width is clamped so neither the chat
// nor the graph can be squeezed out, and is remembered in this browser.
const GRAPH_MIN = 280;
const CHAT_MIN = 360 + 220;   // the chat column plus the left panel
function setGraphWidth(px) {
  const max = Math.max(GRAPH_MIN, window.innerWidth - CHAT_MIN);
  $("main").style.setProperty("--graph-w", `${Math.round(Math.min(max, Math.max(GRAPH_MIN, px)))}px`);
  if (state.cy) state.cy.resize();
}
{
  const saved = parseInt(recallLocal("graph-width"), 10);
  if (saved) setGraphWidth(saved);
  const handle = $("resize-graph");
  handle.addEventListener("pointerdown", (ev) => {
    handle.setPointerCapture(ev.pointerId);
    handle.classList.add("dragging");
    const move = (e) => setGraphWidth(window.innerWidth - e.clientX);
    const stop = () => {
      handle.classList.remove("dragging");
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", stop);
      handle.removeEventListener("pointercancel", stop);
      saveLocal("graph-width", parseInt($("main").style.getPropertyValue("--graph-w"), 10));
      if (state.cy) state.cy.fit(undefined, 20);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
  });
  handle.addEventListener("dblclick", () => {   // back to the default width
    $("main").style.removeProperty("--graph-w");
    saveLocal("graph-width", undefined);
    if (state.cy) { state.cy.resize(); state.cy.fit(undefined, 20); }
  });
}

async function drawGraph() {
  if (!graphOpen()) return;
  const view = await api("GET", "/api/graph");
  const elements = [
    ...view.nodes.map((n) => ({ data: { ...n, name: short(n.text || n.id, 28) } })),
    ...view.edges.map((e, i) => ({ data: { id: `e${i}`, source: e.source, target: e.target, type: e.type } })),
  ];
  if (!state.cy) {
    state.cy = cytoscape({
      container: $("graph"),
      wheelSensitivity: 0.2,
      style: [
        { selector: "node", style: { label: "data(name)", "font-size": 9, "text-wrap": "wrap", "text-max-width": 90,
            "text-valign": "bottom", "text-margin-y": 3, width: 22, height: 22, "background-color": "#adb5bd",
            shape: (n) => SHAPES[n.data("label")] || "ellipse" } },
        { selector: "node[status]", style: { "background-color": (n) => COLORS[n.data("status")] || "#adb5bd" } },
        { selector: "node[label = 'Entity']", style: { "background-color": "#4dabf7" } },
        { selector: "node[label = 'Agent']", style: { "background-color": "#7950f2" } },
        { selector: "node[label = 'Message']", style: { "background-color": "#ffd43b" } },
        { selector: "node[visibility = 'private']", style: { "border-width": 2, "border-style": "dashed", "border-color": "#1d2330" } },
        { selector: "edge", style: { width: 1.2, "line-color": "#ced4da", "target-arrow-color": "#ced4da",
            "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(type)", "font-size": 7, color: "#868e96" } },
        { selector: "edge[type = 'DERIVED_FROM']", style: { "line-color": "#495057", "target-arrow-color": "#495057", width: 2 } },
        { selector: ".hit", style: { "border-width": 4, "border-color": "#f08c00", "border-style": "solid" } },
        { selector: "edge.hit", style: { "line-color": "#f08c00", "target-arrow-color": "#f08c00", width: 3 } },
        { selector: ":selected", style: { "border-width": 3, "border-color": "#3b5bdb" } },
      ],
    });
    state.cy.on("tap", "node", (e) => inspect(e.target.data()));
  }
  state.cy.elements().remove();
  state.cy.add(elements);
  state.cy.layout({ name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 70 }).run();
}

$("refresh-graph").addEventListener("click", drawGraph);

// No claims yet (Planner, Writer: decisions and drafts are never counted) means no score, not 100%.
function accuracyLabel(a) {
  if (!a.claims) return "no claims";
  return (a.accuracy < 0.75 ? "⚠ " : "") + Math.round(a.accuracy * 100) + "%";
}

// Inspector: read-only buttons, run as the user's own Researcher. Changes to memory are made
// by the agents in the chat, never from here.

async function tool(name, args) {
  return api("POST", `/api/tools/${name}`, { role: "researcher", chat: state.chat, args });
}

function inspect(node) {
  const box = $("inspector");
  box.classList.remove("muted");
  const lines = [el("h4", {}, `${node.label} ${node.id}`), el("div", {}, node.text || "")];
  if (node.status) lines.push(el("div", { class: "small muted" },
    `status: ${node.status} · ${node.visibility} · by ${node.author}${mine(node.author) ? " (yours)" : ""}`));
  if (node.label === "Agent") lines.push(el("div", { class: "small muted" }, `owner ${node.owner} · accuracy ${accuracyLabel(node)} (${node.mistakes ?? 0} wrong of ${node.claims ?? 0} claims)`));
  const actions = el("div", { class: "actions" });
  const item = ["Fact", "Decision", "Procedure"].includes(node.label);
  if (item) {
    actions.append(
      el("button", { class: "ghost small", onclick: () => explain(node) }, "Explain"),
      el("button", { class: "ghost small", onclick: () => blast(node) }, "Blast radius"));
  }
  box.replaceChildren(...lines, actions, el("div", { id: "inspect-out" }));
}

function out(...children) { const o = $("inspect-out"); if (o) o.replaceChildren(...children); }

async function explain(node) {
  try {
    const { chain } = await tool("explain", { item_id: node.id });
    out(el("strong", { class: "small" }, "Built on:"),
      el("ol", {}, ...chain.map((c) => el("li", {}, `${short(c.text, 60)} — ${roleOf(c.agent)}, ${c.status}`))));
    highlight(chain.map((c) => c.id));
  } catch (err) { out(el("p", { class: "call err" }, err.message)); }
}

async function blast(node) {
  try {
    const { affected } = await tool("blast_radius", { item_id: node.id });
    out(el("strong", { class: "small" }, affected.length ? `${affected.length} item(s) built on this:` : "Nothing is built on this."),
      el("ol", {}, ...affected.map((a) => el("li", {}, `${short(a.text, 60)} — ${roleOf(a.agent)}, depth ${a.depth}`))));
    highlight([node.id, ...affected.map((a) => a.id)]);
  } catch (err) { out(el("p", { class: "call err" }, err.message)); }
}

function highlight(ids) {
  const cy = state.cy;
  cy.elements().removeClass("hit");
  const set = new Set(ids);
  cy.nodes().filter((n) => set.has(n.id())).addClass("hit");
  cy.edges().filter((e) => e.data("type") === "DERIVED_FROM" && set.has(e.data("source")) && set.has(e.data("target"))).addClass("hit");
}

// API key box (bottom left)
function renderKey(message, bad) {
  const status = $("key-status");
  $("keybox").classList.toggle("need", !state.key && !state.serverKey);
  if (message) status.textContent = message;
  else if (state.key) status.textContent = `Key saved for this tab (ends ${state.key.slice(-4)}).`;
  else if (state.serverKey) status.textContent = "This server has its own key. Paste yours to use it instead.";
  else status.textContent = "Paste your Anthropic API key to start.";
  status.classList.toggle("bad", !!bad);
  $("key-clear").classList.toggle("hidden", !state.key);
}

$("key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const candidate = $("key-input").value.trim();
  if (!candidate) return;
  const previous = state.key;
  state.key = candidate;
  renderKey("Checking with Anthropic...");
  try {
    await api("POST", "/api/key/check");
    saveSession("apikey", candidate);
    $("key-input").value = "";
    renderKey();
    if (state.user) await refreshPanels();
  } catch (err) {
    state.key = previous;
    renderKey(err.message, true);
  }
});

$("key-clear").addEventListener("click", async () => {
  state.key = null;
  saveSession("apikey", undefined);
  renderKey();
  if (state.user) await refreshPanels();
});

// Start
renderKey();
state.user = recallLocal("user");
state.org = recallLocal("org");
(state.user && state.org ? enter().catch(showEntry) : showEntry());
