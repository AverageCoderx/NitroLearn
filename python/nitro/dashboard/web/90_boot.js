// Starts the page: draws it from the first data, then keeps it fresh.
render();
(async () => {
  await ready();
  await loadHome(); render();
  askUnfinishedUpdate();
  // While something fills the window over the page (the match viewer sets S.resting), the page rests: it asks
  // for nothing and redraws nothing.
  const every = (ms, run) => setInterval(() => { if (!S.resting) run(); }, ms);
  every(5000, async () => {
    if (S.view === "home") { await loadHome(); if ($("#modal").hidden) renderHome(true); }
    else if (S.view === "prefs") { await loadHome(); }
    else { await loadHome(); await refreshBot(); }
  });
  every(1500, pollConsole);
  every(1500, pollBuild);
  every(1500, pollUpdate);
  every(1500, pollRLBotInstall);
  every(2000, pollConverting);
})();
