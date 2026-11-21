// The Help page: the handbook (docs/guide, read here), the report to paste into Discord, and where to get
// help, report a bug or suggest an idea. NitroLearn is supported in the Vanta Discord server; its button
// opens the invite in the web browser.
const SUPPORT_DISCORD = "https://discord.gg/bYP3kYtryg";

// ---------- a small Markdown reader for the handbook's pages ----------
// Headings, paragraphs, lists, tables, code blocks, `code`, **bold**, *italics* and links. Every text is
// escaped first; links become buttons: a handbook page opens here, a web page in the browser.
function mdSlug(text) { return String(text).toLowerCase().replace(/[`*]/g, "").replace(/[^a-z0-9 _-]/g, "").trim().replace(/ /g, "-"); }
function mdInline(text) {
  const codes = [];
  let html = esc(text).replace(/`([^`]+)`/g, (_m, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  html = html.replace(/&lt;(https:\/\/[^\s&]+)&gt;/g, (_m, url) => mdLink(url, url));
  html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_m, words, href) => mdLink(words, href));
  html = html.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*\w])\*([^*\s][^*]*)\*(?!\w)/g, "$1<i>$2</i>");
  return html.replace(/\u0000(\d+)\u0000/g, (_m, i) => `<code class="mono">${codes[Number(i)]}</code>`);
}
function mdLink(words, href) {
  const plain = String(href).replace(/&amp;/g, "&");
  return `<button type="button" class="btn link" data-act="guide-link" data-href="${esc(plain)}" title="${esc(plain)}">${words}</button>`;
}
function mdHTML(text) {
  const lines = String(text).replace(/\r/g, "").split("\n"), out = [];
  let i = 0;
  const cells = row => row.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const body = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) body.push(lines[i]);
      i++;
      out.push(`<pre class="md-code">${esc(body.join("\n"))}</pre>`);
    } else if (/^#{1,4} /.test(line)) {
      const level = line.match(/^#+/)[0].length, words = line.slice(level + 1);
      out.push(`<h${level + 1} class="md-h" id="md-${mdSlug(words)}">${mdInline(words)}</h${level + 1}>`);
      i++;
    } else if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s|:-]+\|\s*$/.test(lines[i + 1])) {
      const head = cells(line);
      const rows = [];
      for (i += 2; i < lines.length && /^\s*\|/.test(lines[i]); i++) rows.push(cells(lines[i]));
      out.push(`<div class="md-table"><table><thead><tr>${head.map(c => `<th>${mdInline(c)}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${mdInline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
    } else if (/^\s*([-*]|\d+\.) /.test(line)) {
      const ordered = /^\s*\d+\./.test(line), items = [];
      while (i < lines.length && lines[i].trim() !== "" && !/^```/.test(lines[i]) && !/^#{1,4} /.test(lines[i])) {
        if (/^\s*([-*]|\d+\.) /.test(lines[i]) && !/^\s{4,}/.test(lines[i])) items.push(lines[i].replace(/^\s*([-*]|\d+\.) /, ""));
        else items[items.length - 1] += " " + lines[i].trim();
        i++;
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>${items.map(it => `<li>${mdInline(it)}</li>`).join("")}</${tag}>`);
    } else if (line.trim() === "") {
      i++;
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() !== "" && !/^```/.test(lines[i]) && !/^#{1,4} /.test(lines[i]) && !/^\s*([-*]|\d+\.) /.test(lines[i]) && !/^\s*\|/.test(lines[i])) para.push(lines[i++].trim());
      out.push(`<p>${mdInline(para.join(" "))}</p>`);
    }
  }
  return out.join("");
}

// ---------- the page ----------
S.help = { page: "README.md", pages: null, text: "", error: "", anchor: "", report: null };
async function openGuide(page, anchor = "") {
  const out = await call("guide_page", page);
  S.help.page = page; S.help.anchor = anchor;
  S.help.text = out.error ? "" : out.text; S.help.error = out.error || "";
  if (S.view !== "help") { S.view = "help"; S.bot = null; S.data = null; }
  render();
  const target = anchor && document.getElementById("md-" + anchor);
  if (target) target.scrollIntoView({ block: "start" });
  else { const reader = $("#guide-reader"); if (reader) reader.scrollTop = 0; }
}
ENTER.help = () => {
  call("guide_pages").then(out => { S.help.pages = out.error ? [] : out.pages; if (S.view === "help") renderHelp(); });
  if (!S.help.text) openGuide(S.help.page);
};
function renderHelp() {
  const h = S.help;
  const reason = (title, text) => `<div class="setting"><div class="what"><div class="name">${title}</div></div><div class="help full">${text}</div></div>`;
  const list = (h.pages || []).map(p => `<button type="button" class="guide-nav${p.page === h.page ? " on" : ""}" data-act="guide-open" data-page="${esc(p.page)}">${esc(p.title)}</button>`).join("");
  const bots = (S.home && S.home.bots) || [];
  const chosen = h.reportBot != null ? h.reportBot : (bots[0] ? bots[0].name : "");
  const reader = h.error ? `<div class="banner warn">${esc(h.error)}</div>` : h.text ? mdHTML(h.text) : `<div class="muted">Opening…</div>`;
  $("#main").innerHTML = `<div class="heading"><h1 class="display">Help</h1><div class="row"><button type="button" class="btn small primary" data-act="help-discord">Join the Vanta Discord server</button></div></div>
    <div class="help-split">
      <nav class="guide-list" aria-label="The handbook"><button type="button" class="guide-nav${h.page === "README.md" ? " on" : ""}" data-act="guide-open" data-page="README.md">The handbook</button>${list}</nav>
      <article class="card md" id="guide-reader">${reader}</article>
    </div>
    <div class="heading"><h2 class="display">Ask for help</h2></div>
    <div class="muted" style="font-size:13px;max-width:90ch;">NitroLearn is supported in the <b>Vanta Discord server</b>. Join it for any of these, and say which one in your message:</div>
    <div class="set-rows">
      ${reason("Get help", "Something will not install, a bot will not train, or you are not sure what a setting does. Say what you were doing and what you saw, and paste the report below.")}
      ${reason("Report a bug", "Something went wrong or looks broken. Say what you did, what you expected and what happened instead. A screenshot and the report below help most; for an install problem, add install-log.txt from NitroLearn's folder (look at it before you share: it can show your folder names).")}
      ${reason("Suggest an idea", "A feature, a reward, a change to the dashboard, anything that would make NitroLearn better for you.")}
    </div>
    <div class="card" style="padding:14px 16px;margin-top:14px;display:flex;flex-direction:column;gap:10px;">
      <div><b>Copy a report for Discord</b><div class="muted" style="font-size:13px;max-width:90ch;">NitroLearn's version, what Check this PC found, the bot's settings and the last lines of its console, ready to paste. Your user name and folder names are taken out. Run <b>Check this PC</b> first so the report holds its results.</div></div>
      <div class="row" style="gap:10px;flex-wrap:wrap;align-items:center;">
        <label class="muted" style="font-size:13px;">Bot <select class="field" data-help-bot="1"><option value="">No bot</option>${bots.map(b => `<option value="${esc(b.name)}"${b.name === chosen ? " selected" : ""}>${esc(b.name)}</option>`).join("")}</select></label>
        <button type="button" class="btn small primary" data-act="help-report">Copy a report for Discord</button>
        <button type="button" class="btn small ghost" data-act="help-report-show">Show it first</button>
      </div>
      ${h.report ? `<pre class="md-code" style="max-height:260px;overflow:auto;">${esc(h.report)}</pre>` : ""}
    </div>
    <div class="muted" style="font-size:13px;">The invite: <span class="mono">${esc(SUPPORT_DISCORD)}</span> · Before you ask, <b>Check this PC</b> often names the problem and how to fix it, and <button type="button" class="btn link" data-act="guide-open" data-page="troubleshooting.md">Troubleshooting</button> lists what to do by what you see.</div>`;
}
VIEWS.help = () => renderHelp();
ACTIONS["help-discord"] = () => call("open_url", SUPPORT_DISCORD).then(out => { if (out && out.error) toast(out.error, true); });
ACTIONS["guide-open"] = t => openGuide(t.dataset.page);
// A link's page as guide_page names it: the handbook's own by file name, the docs folder's as "../name.md"
// (a link on one of those pages is relative to the docs folder). Anything else is not a page.
function guidePath(path) {
  const inDocs = S.help.page.startsWith("../");
  let full = (inDocs ? "docs/" : "docs/guide/") + path;
  const parts = [];
  for (const part of full.split("/")) { if (part === "..") parts.pop(); else if (part !== ".") parts.push(part); }
  full = parts.join("/");
  if (/^docs\/guide\/[A-Za-z0-9_-]+\.md$/.test(full)) return full.slice("docs/guide/".length);
  if (/^docs\/[A-Za-z0-9_-]+\.md$/.test(full)) return "../" + full.slice("docs/".length);
  return "";
}
ACTIONS["guide-link"] = t => {
  const href = t.dataset.href || "";
  if (/^https:\/\//.test(href)) return call("open_url", href).then(out => { if (out && out.error) toast(out.error, true); });
  const [path, anchor] = href.split("#");
  if (!path && anchor) { const el = document.getElementById("md-" + anchor); if (el) el.scrollIntoView({ block: "start" }); return; }
  const page = guidePath(path);
  if (page) return openGuide(page, anchor || "");
  return toast(`${href} is a file in NitroLearn's folder.`);
};
on("change", el => {
  if (!el.dataset.helpBot) return SKIP;
  S.help.reportBot = el.value;
});
function reportBot() {
  const bots = (S.home && S.home.bots) || [];
  return S.help.reportBot != null ? S.help.reportBot : (bots[0] ? bots[0].name : "");
}
ACTIONS["help-report"] = async () => {
  const out = await call("discord_report", reportBot());
  if (out.error) return toast(out.error, true);
  const copied = await copyText(out.text);
  if (!copied) { S.help.report = out.text; renderHelp(); return toast("Copying is not allowed here: select the report below and press Ctrl+C.", true); }
  const notes = [out.checked ? "" : " Check this PC has not run yet, so its results are not in it.", out.long ? " It is longer than one Discord message: paste it into a text file and attach that." : ""].join("");
  return toast("Copied the report: paste it into your Discord message." + notes);
};
ACTIONS["help-report-show"] = async () => {
  const out = await call("discord_report", reportBot());
  if (out.error) return toast(out.error, true);
  S.help.report = out.text; renderHelp();
};
