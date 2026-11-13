// The page's guard, run before any other script of the page: when the 3D view cannot start (no WebGL, an
// error while the page's scripts load or start, or no picture drawn in time) the page says so in plain words
// instead of staying blank. It stands alone, with its own style, so it works when nothing else on the page
// does. The page's start-up script reports to it through `NV_GUARD`.
window.NV_GUARD = (() => {
  "use strict";
  // The first picture should be drawn within this much time while the page is in view (ms).
  const LIMIT = 8000;
  const ADVICE = "Try another browser, or turn on hardware acceleration (Chrome: Settings > System).";
  const page = location.href.split("#")[0];
  let box = null;
  // Whether the message shown is only that the first picture is late (it goes if the picture comes).
  let late = false;
  let drawn = false;
  let settled = false;
  let timer = 0;

  // Shows the message (in place of one that only said the picture is late); false when there is nothing to
  // show it for, or a message is already shown.
  function show(reason, detail) {
    if (drawn || settled || (box && !late)) return false;
    if (box) box.remove();
    late = false;
    box = document.createElement("div");
    box.className = "nv-fail";
    box.setAttribute("role", "alert");
    box.style.cssText =
      "position:fixed;inset:0;z-index:2147483647;display:flex;flex-direction:column;align-items:center;" +
      "justify-content:center;gap:14px;padding:0 10%;margin:0;background:#141413;color:#F3F1EC;" +
      "font:18px/1.45 Archivo,system-ui,sans-serif;text-align:center;user-select:text;-webkit-user-select:text";
    const said = document.createElement("p");
    said.style.cssText = "margin:0;max-width:44em";
    said.textContent = `The 3D view could not start: ${reason}. ${ADVICE}`;
    const why = document.createElement("p");
    why.className = "nv-fail-detail";
    why.style.cssText =
      "margin:0;max-width:60em;color:#ACA9A1;font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere";
    why.textContent = detail || "";
    box.append(said, why);
    const place = () => (document.body || document.documentElement).append(box);
    if (document.body) place();
    else document.addEventListener("DOMContentLoaded", place, { once: true });
    return true;
  }

  // What went wrong, from an error the start-up met: a plain reason and the technical detail.
  function explain(error) {
    const text = error && error.message ? `${error.name || "Error"}: ${error.message}` : String(error);
    if (error && error.name === "NoGraphicsError") {
      return ["this browser cannot draw 3D graphics here (WebGL 2 is turned off or not available)", text];
    }
    return ["the page stopped with an error", text];
  }

  function fail(error) {
    const [reason, detail] = explain(error);
    show(reason, detail);
  }

  // Errors in the page's own scripts before the first picture: a script that does not parse or stops while
  // the page starts. An error in a script file beside the page (the arena's) is not the page's: the view
  // starts without what that file holds.
  addEventListener("error", (event) => {
    if (!event.filename || event.filename.split("#")[0] !== page) return;
    const where = event.lineno ? ` (line ${event.lineno})` : "";
    show("the page stopped with an error", `${event.message}${where}`);
  });
  addEventListener("unhandledrejection", (event) => fail(event.reason));

  // Counts the time the page is in view (a page opened behind other tabs draws nothing until shown) and
  // speaks when the first picture has not come within LIMIT; the message goes if the picture comes later.
  function arm() {
    if (timer || drawn || settled) return;
    let waited = 0;
    let last = performance.now();
    timer = setInterval(() => {
      const now = performance.now();
      if (document.visibilityState === "visible") waited += now - last;
      last = now;
      if (waited >= LIMIT) {
        clearInterval(timer);
        late = show(`the first picture was not drawn within ${LIMIT / 1000} seconds`, "No frame was drawn.");
      }
    }, 250);
  }

  // The first picture is drawn: the view works.
  function started() {
    drawn = true;
    clearInterval(timer);
    if (late) {
      box.remove();
      box = null;
    }
  }

  // The page shows a message of its own instead of a game (a recording it cannot read): nothing to watch for.
  function settle() {
    settled = true;
    clearInterval(timer);
  }

  return { fail, arm, started, settle };
})();
