// The dialogs of a bot's actions: New bot from this one, and adding a packed bot. Making an RLBot bot and
// packing a bot open their tool (37_tools.js); watching is in 75_viewer.js.
function copyModal() {
  const b = S.data, newest = b.newest_save;
  const from = newest != null
    ? `Start from ${esc(b.name)}'s trained networks (transfer learning): its newest save, at ${newest.toLocaleString("en-US")} steps, taking everything it learned, since the new bot keeps its rewards.`
    : `Start from ${esc(b.name)}'s trained networks (transfer learning): not possible yet, it has no save.`;
  modal(`<h2 class="display">New bot from ${esc(b.name)}</h2>
    <label class="f">Name<input id="cp-name" class="field" type="text" autocomplete="off" value="${esc(b.name)}-2"></label>
    <label class="f check"><input id="cp-transfer" type="checkbox" ${newest != null ? "checked" : "disabled"}><span>${from} Unticked, it starts from scratch with the same settings.</span></label>
    <div class="muted" style="font-size:13px;">The new bot is made but not started. Until its first start everything can still change in its Settings tab, network sizes and transfer learning included (on or off, another save, or the policy only). New rewards switch it to the policy only.</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Cancel</button><button type="button" class="btn primary" data-act="copy-go">Make it</button></div>`);
}

// ---------- renaming and deleting a bot ----------
function renameModal() {
  const b = S.data;
  modal(`<h2 class="display">Rename ${esc(b.name)}</h2>
    <label class="f">New name<span class="h">Up to 64 letters, digits, -, _ and . (starting with a letter or digit).</span><input id="rn-name" class="field" type="text" autocomplete="off" spellcheck="false" value="${esc(b.name)}"></label>
    <div id="rn-error"></div>
    <div class="muted" style="font-size:13px;">Its saves, settings, notes and charts stay as they are. ${S.tools && S.tools.rlbot ? "Its RLBot bot keeps the folder and name it was made with: make it again to give it the new name." : ""}</div>
    <div class="foot"><button type="button" class="btn" data-close="1">Cancel</button><button type="button" class="btn primary" data-act="rename-go">Rename</button></div>`);
  const field = $("#rn-name"); field.focus(); field.select();
}
ACTIONS["rename-go"] = async () => {
  const name = S.bot, out = await call("rename_bot", name, $("#rn-name").value.trim());
  if (out.error) return put($("#rn-error"), `<span class="err" role="alert">${esc(out.error[0].toUpperCase() + out.error.slice(1))}.</span>`);
  closeModal(); await loadHome();
  toast(out.renamed === name ? "The name stays " + name + "." : name + " is now called " + out.renamed + ".");
  return openBot(out.renamed, "tools");
};
function deleteModal() {
  const b = S.data, runs = String((S.home && S.home.runs) || "runs"), sep = runs.includes("\\") ? "\\" : "/";
  const made = S.tools && S.tools.rlbot;
  modal(`<h2 class="display">Delete ${esc(b.name)}?</h2>
    <div>This deletes <b>${esc(b.name)}</b> from this PC for good: its folder <span class="mono path">${esc(runs + sep + b.name)}</span>, with its settings, ${b.saves} ${b.saves === 1 ? "save" : "saves"}, notes, charts and recorded games. No other bot is touched. This cannot be undone.</div>
    ${made ? `<label class="check"><input id="del-rlbot" type="checkbox"><span>Also delete its RLBot bot, <span class="mono">${esc(made.folder)}</span></span></label>` : ""}
    <label class="f">Type the bot's name to confirm<input id="del-name" class="field" type="text" autocomplete="off" spellcheck="false" placeholder="${esc(b.name)}"></label>
    <div id="del-error"></div>
    <div class="foot"><button type="button" class="btn" data-close="1">Keep it</button><button type="button" class="btn danger" id="del-go" data-act="delete-go" disabled>Delete ${esc(b.name)}</button></div>`);
  $("#del-name").focus();
}
on("input", el => { if (el.id === "del-name" && S.data) $("#del-go").disabled = el.value.trim() !== S.data.name; });
ACTIONS["delete-go"] = async () => {
  const name = S.bot;
  if ($("#del-name").value.trim() !== name) return;
  const out = await call("delete_bot", name, !!($("#del-rlbot") && $("#del-rlbot").checked));
  if (out.error) return put($("#del-error"), `<span class="err" role="alert">${esc(out.error[0].toUpperCase() + out.error.slice(1))}.</span>`);
  closeModal(); toast("Deleted " + name + ".");
  S.view = "home"; S.bot = null; S.data = null;
  await loadHome(); return render();
};

ACTIONS.export = () => openTool("export");
ACTIONS.copy = () => copyModal();
ACTIONS["copy-go"] = async () => { const out = await call("copy_bot", S.bot, $("#cp-name").value.trim(), $("#cp-transfer").checked); if (out.error) return toast(out.error, true); closeModal(); await loadHome(); toast("Made " + out.made + "."); (out.notes || []).forEach(n => toast(n)); return openBot(out.made, "settings"); };
ACTIONS.pack = () => openTool("pack");
// Adds a packed bot: `how` "replace" takes the pack chosen before in place of the bot of its name, "rename"
// adds it under the name typed in the dialog.
async function unpackBot(how) {
  const out = how === "replace" ? await call("unpack_bot", S.unpacking, true)
    : how === "rename" ? await call("unpack_bot", S.unpacking, false, $("#up-name").value.trim())
    : await call("unpack_bot");
  if (out.cancelled) return;
  if (out.error) return how === "rename" ? put($("#up-error"), `<span class="err" role="alert">${esc(out.error[0].toUpperCase() + out.error.slice(1))}.</span>`) : toast(out.error, true);
  if (out.exists) {
    S.unpacking = out.path;
    return modal(`<h2 class="display">There is already a bot called ${esc(out.exists)}</h2>
      <div style="font-size:13.5px;">Add the packed bot under another name, or replace the bot here with it.</div>
      <label class="f">New name<input id="up-name" class="field" type="text" autocomplete="off" spellcheck="false" value="${esc(out.free || "")}"></label>
      <div id="up-error"></div>
      <div class="muted" style="font-size:13px;">Replacing does not delete the bot here: it is kept, as it is, in the runs folder's .replaced folder.</div>
      <div class="foot"><button type="button" class="btn" data-close="1">Cancel</button><button type="button" class="btn" data-act="unpack-replace">Replace it</button><button type="button" class="btn primary" data-act="unpack-rename">Add under the new name</button></div>`);
  }
  closeModal(); await loadHome();
  toast("Added " + out.unpacked + "." + ((out.rust || []).length ? " It uses Rust rewards of its own (" + out.rust.join(", ") + "): press Build on the Resources page before you start it." : "") + ((out.notes || []).length ? " " + out.notes.join(" ") : ""));
  return openBot(out.unpacked);
}
ACTIONS.unpack = () => unpackBot("");
ACTIONS["unpack-replace"] = () => unpackBot("replace");
ACTIONS["unpack-rename"] = () => unpackBot("rename");
