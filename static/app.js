// The demo page. Talks only to server.py; every action is a memory tool call on the server,
// so the page can't bypass any memory rule. No framework: plain DOM + Cytoscape for the graph.

const $ = (id) => document.getElementById(id);
const state = { user: null, org: null, agents: {}, chat: null, cy: null, busy: false };

// --- small helpers ---

function saveLocal(key, value) {          // per-browser convenience only; the page works without it
  try { value === undefined ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch (e) {}
}
function recallLocal(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", "X-User": state.user || "", "X-Org": state.org || "" },
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

// --- entry ---

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

// --- chats ---

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
  if (!messages.length) $("messages").append(el("p", { class: "muted center" }, "Ask your team anything. The Researcher answers first and passes work on when needed."));
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
  state.busy = true; $("send").disabled = true; $("input").value = "";
  const empty = $("messages").querySelector(".center");
  if (empty) empty.remove();
  $("messages").append(el("div", { class: "msg user" }, text));
  const pending = el("div", { class: "msg agent pending" }, "The team is working…");
  $("messages").append(pending); scrollDown();
  setBusyAgents(true);
  try {
    const out = await api("POST", `/api/chats/${state.chat}/messages`, { text });
    const { messages } = await api("GET", `/api/chats/${state.chat}`);
    $("messages").replaceChildren(...renderChat(messages));
    if (out.stop !== "done") $("messages").append(el("p", { class: "small muted" }, `(stopped: ${out.stop})`));
  } catch (err) {
    pending.textContent = "Error: " + err.message;
  } finally {
    state.busy = false; $("send").disabled = false; setBusyAgents(false);
    scrollDown();
    await Promise.all([renderChats(), refreshPanels()]);
  }
});

$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("send-form").requestSubmit(); }
});

function scrollDown() { $("messages").scrollTop = $("messages").scrollHeight; }
function setBusyAgents(on) { for (const li of $("my-agents").children) li.classList.toggle("busy", on); }

// --- side panels ---

async function refreshPanels() {
  const [agents, flagged, status] = await Promise.all([
    api("GET", "/api/agents"), api("GET", "/api/flagged"), api("GET", "/api/status")]);
  // accuracy = (claims - mistakes + 4) / (claims + 4); facts the user told it never count
  const row = (a) => el("li", { class: "agent-row" }, el("span", {}, `${a.name}`),
    el("span", { class: "trust" + (a.accuracy < 0.75 ? " low" : ""),
      title: `accuracy: ${a.mistakes} wrong of ${a.claims} claim(s) it made itself` },
      (a.accuracy < 0.75 ? "⚠ " : "") + Math.round(a.accuracy * 100) + "%"));
  const ours = agents.agents.filter((a) => mine(a.id));
  const others = agents.agents.filter((a) => !mine(a.id));
  $("my-agents").replaceChildren(...ours.map(row));
  $("other-agents").replaceChildren(...(others.length ? others.map((a) => {
    const li = row(a); li.firstChild.textContent = `${(a.owner || "").replace("user:", "")} · ${a.name}`; return li;
  }) : [el("li", {}, "Nobody else yet")]));

  const items = Object.entries(flagged.flagged).flatMap(([role, list]) => list.map((i) => ({ ...i, role })));
  $("flagged").replaceChildren(...items.map((i) => el("li", {
    class: i.wrong.length ? "wrong" : "changed",
    title: i.id,
  }, `${i.role}: ${short(i.text, 70)} — ${i.wrong.length ? "built on a wrong fact: redo" : "a source changed: review"}`)));
  $("flag-count").textContent = items.length;
  $("flag-count").classList.toggle("hidden", !items.length);
  $("status").textContent = `${status.mode} · $${status.spend.toFixed(4)} / $${status.cap.toFixed(2)}`;
  await drawGraph();
}

// --- graph ---

const COLORS = { active: "#2f9e44", flagged: "#e03131", retracted: "#868e96", superseded: "#ced4da" };
const SHAPES = { Fact: "ellipse", Decision: "diamond", Procedure: "hexagon", Entity: "round-rectangle",
  Agent: "triangle", Message: "tag" };

async function drawGraph() {
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

// --- inspector: buttons are memory tools run as the user's own Researcher ---

async function tool(name, args) {
  return api("POST", `/api/tools/${name}`, { role: "researcher", chat: state.chat, args });
}

function inspect(node) {
  const box = $("inspector");
  box.classList.remove("muted");
  const lines = [el("h4", {}, `${node.label} ${node.id}`), el("div", {}, node.text || "")];
  if (node.status) lines.push(el("div", { class: "small muted" },
    `status: ${node.status} · ${node.visibility} · by ${node.author}${mine(node.author) ? " (yours)" : ""}`));
  if (node.label === "Agent") lines.push(el("div", { class: "small muted" }, `owner ${node.owner} · accuracy ${Math.round((node.accuracy ?? 1) * 100)}% (${node.mistakes ?? 0} wrong of ${node.claims ?? 0} claims)`));
  const actions = el("div", { class: "actions" });
  const item = ["Fact", "Decision", "Procedure"].includes(node.label);
  if (item) {
    actions.append(
      el("button", { class: "ghost small", onclick: () => explain(node) }, "Explain"),
      el("button", { class: "ghost small", onclick: () => blast(node) }, "Blast radius"));
  }
  if (item && mine(node.author) && node.visibility === "private" && node.status === "active")
    actions.append(el("button", { class: "small", onclick: () => act("publish", { item_id: node.id }) }, "Publish"));
  if (node.label === "Fact" && mine(node.author) && ["active", "flagged"].includes(node.status)) {
    actions.append(
      el("button", { class: "small", onclick: () => correct(node) }, "Correct…"),
      el("button", { class: "danger small", onclick: () => retract(node) }, "Retract"));
  }
  if (item && mine(node.author) && node.status === "flagged")
    actions.append(el("button", { class: "ghost small", onclick: () => act("confirm", { item_id: node.id }) }, "Confirm (still fine)"));
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

async function retract(node) {
  const reason = prompt(`Retract "${short(node.text, 80)}"? It was WRONG and has no replacement.\nReason:`);
  if (reason === null) return;
  await act("retract", { fact_id: node.id, reason });
}

async function correct(node) {
  const text = prompt(`The fact was WRONG. What is right?\n(was: ${node.text})`);
  if (!text) return;
  const reason = prompt("Why was it wrong?") || "";
  await act("correct", { old_id: node.id, text, reason });
}

async function act(name, args) {
  try {
    const result = await tool(name, args);
    await refreshPanels();
    const flagged = result.flagged || [];          // ids of the downstream items this user may see
    const hidden = result.hidden_flagged ? ` (+${result.hidden_flagged} private to other users)` : "";
    out(el("p", { class: "small" }, `${name} done.` + (flagged.length || hidden ? ` Flagged ${flagged.length} item(s) downstream${hidden}.` : "")));
    if (flagged.length) highlight(flagged);
  } catch (err) { out(el("p", { class: "call err" }, err.message)); }
}

// --- start ---

state.user = recallLocal("user");
state.org = recallLocal("org");
(state.user && state.org ? enter().catch(showEntry) : showEntry());
