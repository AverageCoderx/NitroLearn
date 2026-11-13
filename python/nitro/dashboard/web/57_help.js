// The Help page: where to get help, report a bug or suggest an idea. NitroLearn is supported in the Vanta
// Discord server; the button opens its invite in the web browser.
const SUPPORT_DISCORD = "https://discord.gg/bYP3kYtryg";
function renderHelp() {
  const reason = (title, text) => `<div class="setting"><div class="what"><div class="name">${title}</div></div><div class="help full">${text}</div></div>`;
  $("#main").innerHTML = `<div class="heading"><h1 class="display">Help and ideas</h1><div class="row"><button type="button" class="btn small primary" data-act="help-discord">Join the Vanta Discord server</button></div></div>
    <div class="muted" style="font-size:13px;max-width:90ch;">NitroLearn is supported in the <b>Vanta Discord server</b>. Join it for any of these, and say which one in your message:</div>
    <div class="set-rows">
      ${reason("Get help", "Something will not install, a bot will not train, or you are not sure what a setting does. Say what you were doing and what you saw.")}
      ${reason("Report a bug", "Something went wrong or looks broken. Say what you did, what you expected and what happened instead. If you can, add a screenshot and the end of the bot's Console tab, or install-log.txt from NitroLearn's folder for an install problem. These can show your folder names, so look at them before you share.")}
      ${reason("Suggest an idea", "A feature, a reward, a change to the dashboard, anything that would make NitroLearn better for you.")}
    </div>
    <div class="muted" style="font-size:13px;">The invite: <span class="mono">${esc(SUPPORT_DISCORD)}</span> · Before you ask, <b>Check this PC</b> often names the problem and how to fix it.</div>`;
}
VIEWS.help = () => renderHelp();
ACTIONS["help-discord"] = () => call("open_url", SUPPORT_DISCORD).then(out => { if (out && out.error) toast(out.error, true); });
