// The controls over the picture: scoreboard, timeline, play and speed, cameras, a card per car with what
// its bot is thinking, name tags, attention lines, the minimap, the view options, the camera settings, the
// help panel and the keyboard. The page's elements come from markup.html (found by their data-nv names); the
// cars' cards, the name tags and the lists of buttons are made here.

NV.ui = (() => {
  const SPEEDS = [0.25, 0.5, 1, 2, 4];
  const QUALITIES = [["auto", "Auto"], ["low", "Low"], ["medium", "Medium"], ["high", "High"], ["ultra", "Ultra"]];
  // One letter per mechanics mark, in the order of the mark bits.
  const MECHANICS = [["A", "Aerial touch"], ["D", "Dribble"], ["F", "Flick"], ["DT", "Double tap"],
    ["R", "Flip reset"], ["S", "Shot on goal"], ["SV", "Save"], ["X", "Demolition"]];
  const DEFAULTS = { tags: true, tagSpeed: false, thinking: true, attention: true, hitboxes: false, minimap: false };
  const KMH = 0.036;
  const IDLE_SECONDS = 3.5;
  const MARK_SECONDS = 1.6;
  const LIVE_WINDOW = 120; // seconds of a live game the timeline shows
  const EDGE = 8; // pixels the timeline keeps free at each end, so the handle is never cut off
  const TARGET_NAMES = { ball: "Ball", itself: "Itself", pads: "Boost pads", nothing: "Nothing" };

  // The colours of what is drawn on a canvas or in the scene: the style sheet's, read at start-up.
  const paint = { team: ["#5B97FF", "#FF8A2B"], chalk: "#F3F1EC", dim: "#ACA9A1", yellow: "#FFD60A" };

  const el = {};
  const options = { ...DEFAULTS };
  const cards = [], tags = [], names = [];
  const size = { w: 0, h: 0, compact: false };
  const timeline = { from: 0, to: 1, w: 0, dirty: true, drag: false, resume: false, frames: -1, wait: 0 };
  const values = { seen: 0, lo: Infinity, hi: -Infinity };
  const lines = { group: null, looks: [], talk: [] };
  const marks = { lists: [], read: 0 };
  const shown = { playing: null, speed: null, score: [-1, -1], clock: "", live: null, now: "", total: "", quality: "" };
  const v = new THREE.Vector3(), v2 = new THREE.Vector3(), side = new THREE.Vector3(), look = new THREE.Vector3();
  const ends = [new THREE.Vector3(), new THREE.Vector3()];
  let host, rec, player;
  let shownFrame = -1;
  let activeAt = 0, overControl = false, slow = 1, hintTimer = 0, showTimer = 0, quality = "", savedQuality = "";
  const tip = { target: null, timer: 0 };

  const clockText = (seconds) => {
    const whole = Math.max(0, Math.floor(seconds));
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  };
  const setText = (node, text) => {
    if (node.textContent !== text) node.textContent = text;
  };
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const teamCss = (team) => paint.team[team ? 1 : 0];
  const targetName = (target) => (typeof target === "number" ? names[target] : TARGET_NAMES[target] || "Nothing");
  const targetCss = (target) => (typeof target === "number" ? teamCss(rec.teams[target])
    : target === "ball" ? paint.chalk : paint.dim);

  // ---------- playing ----------

  function togglePlay() {
    // A live game keeps running at its newest moment: pausing steps out of it.
    if (player.atLive) {
      player.atLive = false;
      player.playing = true;
    }
    player.toggle();
  }

  function setPlaying(on) {
    if (player.playing !== on || (player.atLive && !on)) togglePlay();
  }

  function seekBy(seconds) {
    player.seek(player.t + seconds);
  }

  function stepFrames(frames) {
    setPlaying(false);
    player.seek((Math.round(player.t * rec.fps) + frames) / rec.fps);
  }

  function changeSpeed(step) {
    let at = SPEEDS.indexOf(player.speed);
    if (at < 0) at = SPEEDS.indexOf(1);
    player.setSpeed(SPEEDS[NV.clamp(at + step, 0, SPEEDS.length - 1)]);
  }

  function goLive() {
    if (!rec.live) return;
    player.goLive();
    NV.cams.cut();
  }

  function watchAgain() {
    player.seek(0);
    setPlaying(true);
  }

  function fullScreen() {
    try {
      const done = document.fullscreenElement ? document.exitFullscreen() : host.requestFullscreen();
      if (done && done.catch) done.catch(() => {});
    } catch {
      // Not allowed here (a frame without the permission): the viewer stays as it is.
    }
  }

  // ---------- the top: score, clock, title, live ----------

  function buildTop() {
    setText(el.title, rec.title);
    setText(el.subtitle, rec.subtitle);
    el.subtitle.hidden = !rec.subtitle;
    el.titleRow.hidden = !rec.title && !rec.subtitle && !rec.live;
    el.liveBadge.hidden = !rec.live;
    el.goLive.addEventListener("click", goLive, { signal: NV.ending.signal });
    NV.on("title", ({ title }) => {
      setText(el.title, title || "");
      el.titleRow.hidden = !title && !rec.subtitle && el.liveBadge.hidden;
    });
    NV.on("live-ended", liveEnded);
    if (NV.live && NV.live.ended) liveEnded();
    el.close.hidden = !canClose();
    el.close.addEventListener("click", close, { signal: NV.ending.signal });
    NV.on("goal", ({ team }) => {
      const node = team ? el.scoreOrange : el.scoreBlue;
      node.classList.remove("nv-pulse");
      void node.offsetWidth;
      node.classList.add("nv-pulse");
    });
  }

  // A live game that is over: the badge says so, and the end of the game offers to watch it again.
  function liveEnded() {
    setText(el.liveBadge, "Ended");
    el.liveBadge.classList.remove("nv-live-now");
    el.liveBadge.hidden = false;
    el.titleRow.hidden = false;
    el.goLive.hidden = true;
    setText(el.endText, "The live game has ended");
    timeline.dirty = true;
    if (player.t >= player.duration - 1) showEnd();
  }

  // The host page may give a way to close the viewer (the dashboard's window does).
  function canClose() {
    return !!NV.options && typeof NV.options.onClose === "function";
  }

  function close() {
    if (canClose()) NV.options.onClose();
  }

  function updateTop(world) {
    for (let team = 0; team < 2; team++) {
      if (shown.score[team] === world.score[team]) continue;
      shown.score[team] = world.score[team];
      setText(team ? el.scoreOrange : el.scoreBlue, String(world.score[team]));
    }
    const clock = clockText(world.clock);
    if (clock !== shown.clock) {
      shown.clock = clock;
      setText(el.clock, clock);
    }
    if (rec.live && shown.live !== player.atLive) {
      shown.live = player.atLive;
      el.liveBadge.classList.toggle("nv-live-now", player.atLive);
      el.goLive.hidden = player.atLive;
    }
  }

  // ---------- the bottom bar: play, speed, cameras ----------

  function buildBar() {
    el.play.addEventListener("click", togglePlay, { signal: NV.ending.signal });
    el.back.addEventListener("click", () => seekBy(-5), { signal: NV.ending.signal });
    el.forward.addEventListener("click", () => seekBy(5), { signal: NV.ending.signal });
    for (const speed of SPEEDS) {
      const button = make("button", "", `${speed}\u00d7`);
      button.type = "button";
      button.dataset.speed = speed;
      button.setAttribute("aria-label", `Speed ${speed} times`);
      button.dataset.tip = speed === 1 ? "Normal speed" : speed < 1 ? "Slow motion" : "Fast forward";
      button.dataset.key = speed < 1 ? "-" : speed > 1 ? "+" : "";
      button.addEventListener("click", () => player.setSpeed(speed), { signal: NV.ending.signal });
      el.speeds.appendChild(button);
    }
    for (const button of el.cameras.querySelectorAll("button[data-camera]")) {
      button.setAttribute("aria-label", `${button.textContent.trim()} camera`);
      button.addEventListener("click", () => NV.cams.set(button.dataset.camera), { signal: NV.ending.signal });
    }
    el.again.addEventListener("click", watchAgain, { signal: NV.ending.signal });
    NV.on("end", showEnd);
    const hideEnd = () => {
      if (el.end.hidden) return;
      el.end.hidden = true;
      if (host.contains(document.activeElement)) host.focus({ preventScroll: true });
    };
    NV.on("seek", hideEnd);
    NV.on("play", ({ playing }) => playing && hideEnd());
    NV.on("camera", syncCamera);
  }

  function showEnd() {
    el.end.hidden = false;
    if (hasKeyboard()) el.again.focus({ preventScroll: true });
  }

  function syncCamera() {
    for (const button of el.cameras.querySelectorAll("button[data-camera]")) {
      button.setAttribute("aria-pressed", String(button.dataset.camera === NV.cams.mode));
    }
    // In the free camera the side arrows turn the view instead of moving through time.
    const free = NV.cams.mode === "free";
    el.back.dataset.key = free ? "" : "Left arrow";
    el.forward.dataset.key = free ? "" : "Right arrow";
    placeThinking();
    timeline.dirty = true;
  }

  function updateBar() {
    const playing = player.playing || player.atLive;
    if (playing !== shown.playing) {
      shown.playing = playing;
      el.play.classList.toggle("nv-playing", playing);
      el.play.setAttribute("aria-label", playing ? "Pause" : "Play");
    }
    if (player.speed !== shown.speed) {
      shown.speed = player.speed;
      for (const button of el.speeds.children) {
        button.setAttribute("aria-pressed", String(Number(button.dataset.speed) === player.speed));
      }
    }
  }

  // ---------- the timeline ----------

  const timelineX = (frame) => EDGE + ((frame - timeline.from) / Math.max(1, timeline.to - timeline.from))
    * (timeline.w - 2 * EDGE);

  function timeAt(clientX) {
    const box = el.timeline.getBoundingClientRect();
    const k = NV.clamp((clientX - box.left - EDGE) / Math.max(1, box.width - 2 * EDGE), 0, 1);
    return (timeline.from + k * (timeline.to - timeline.from)) / rec.fps;
  }

  function buildTimeline() {
    const bar = el.timeline;
    bar.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      bar.setPointerCapture(event.pointerId);
      timeline.drag = true;
      timeline.resume = player.playing && !player.atLive;
      if (timeline.resume) setPlaying(false);
      player.seek(timeAt(event.clientX));
    }, { signal: NV.ending.signal });
    bar.addEventListener("pointermove", (event) => {
      if (timeline.drag) player.seek(timeAt(event.clientX));
      peek(event.clientX);
    }, { signal: NV.ending.signal });
    const release = () => {
      if (!timeline.drag) return;
      timeline.drag = false;
      if (timeline.resume && player.t < player.duration - 0.05) setPlaying(true);
    };
    bar.addEventListener("pointerup", release, { signal: NV.ending.signal });
    bar.addEventListener("pointercancel", release, { signal: NV.ending.signal });
    bar.addEventListener("pointerleave", () => {
      el.peek.hidden = true;
    }, { signal: NV.ending.signal });
    const watch = new ResizeObserver(() => {
      timeline.dirty = true;
    });
    watch.observe(bar);
    NV.atStop(() => watch.disconnect());
  }

  // The small bubble over the timeline: the time under the pointer, and the goal or kick-off there.
  function peek(clientX) {
    const box = el.timeline.getBoundingClientRect();
    const x = NV.clamp(clientX - box.left, EDGE, box.width - EDGE);
    const frame = timeAt(clientX) * rec.fps;
    const near = (6 * (timeline.to - timeline.from)) / Math.max(1, box.width - 2 * EDGE);
    let what = "";
    for (const mark of rec.restarts) if (Math.abs(mark - frame) <= near) what = "Kick-off";
    for (const goal of rec.goals) if (Math.abs(goal.frame - frame) <= near) what = `${NV.TEAMS[goal.team].name} goal`;
    setText(el.peek, what ? `${what} \u00b7 ${clockText(frame / rec.fps)}` : clockText(frame / rec.fps));
    el.peek.hidden = false;
    const half = el.peek.offsetWidth / 2;
    el.peek.style.left = `${NV.clamp(x, half, Math.max(half, box.width - half))}px`;
  }

  // The lowest and highest value estimate of the recording so far (a live game adds to them).
  function readValues() {
    const list = rec.decisions;
    for (; values.seen < list.length; values.seen++) {
      const decision = list[values.seen];
      for (let k = 1; k < decision.length; k++) {
        if (!decision[k]) continue;
        values.lo = Math.min(values.lo, decision[k][0]);
        values.hi = Math.max(values.hi, decision[k][0]);
      }
    }
  }

  // What does not move with the play: the track, the marks and the value curves.
  function drawTimeline() {
    const canvas = el.timelineCanvas;
    const w = el.timeline.clientWidth, h = el.timeline.clientHeight;
    if (w < 2 * EDGE + 4 || h < 12) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const last = Math.max(1, rec.count - 1);
    timeline.from = rec.live ? Math.max(0, last - Math.round(LIVE_WINDOW * rec.fps)) : 0;
    timeline.to = last;
    timeline.w = w;
    timeline.frames = rec.count;
    const track = h - 7, inRange = (frame) => frame >= timeline.from && frame <= timeline.to;

    ctx.fillStyle = "rgba(243,241,236,0.2)";
    ctx.fillRect(EDGE, track - 2, w - 2 * EDGE, 4);

    ctx.strokeStyle = "rgba(243,241,236,0.34)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    for (const frame of rec.restarts) {
      if (!inRange(frame)) continue;
      const x = Math.round(timelineX(frame)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, track - 4);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Each car's value estimate over time; the followed car's curve stands out.
    const list = rec.decisions;
    readValues();
    if (values.hi >= values.lo) {
      const lo = values.lo, range = Math.max(1e-6, values.hi - values.lo);
      const top = 5, bottom = track - 12;
      const y = (value) => bottom - ((value - lo) / range) * (bottom - top);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      const order = rec.teams.map((_, car) => car).sort((p, q) => (p === NV.cams.car) - (q === NV.cams.car));
      for (const car of order) {
        const followed = car === NV.cams.car;
        ctx.strokeStyle = teamCss(rec.teams[car]);
        ctx.globalAlpha = followed ? 0.95 : 0.38;
        ctx.lineWidth = followed ? 1.8 : 1.1;
        ctx.beginPath();
        let started = false;
        for (const decision of list) {
          if (!decision[car + 1] || !inRange(decision[0])) continue;
          const px = timelineX(decision[0]), py = y(decision[car + 1][0]);
          if (started) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          started = true;
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    for (const touch of rec.touches) {
      if (!inRange(touch.frame)) continue;
      ctx.fillStyle = teamCss(rec.teams[touch.car]);
      ctx.fillRect(Math.round(timelineX(touch.frame)) - 1, track - 10, 2, 6);
    }
    for (const goal of rec.goals) {
      if (!inRange(goal.frame)) continue;
      const x = Math.round(timelineX(goal.frame));
      ctx.fillStyle = teamCss(goal.team);
      ctx.fillRect(x - 2, 1, 4, h - 2);
      // A flag leaning forward, as the buttons do.
      ctx.beginPath();
      ctx.moveTo(x + 2, 1);
      ctx.lineTo(x + 14, 1);
      ctx.lineTo(x + 12, 10);
      ctx.lineTo(x + 2, 10);
      ctx.fill();
    }
  }

  function updateTimeline(dt) {
    timeline.wait -= dt;
    if (rec.live && rec.count !== timeline.frames && timeline.wait <= 0) timeline.dirty = true;
    if (timeline.dirty) {
      timeline.dirty = false;
      timeline.wait = 0.5;
      drawTimeline();
    }
    if (!timeline.w) return;
    const x = NV.clamp(timelineX(player.t * rec.fps), EDGE, timeline.w - EDGE);
    el.head.style.transform = `translateX(${x.toFixed(1)}px)`;
    el.played.style.width = `${Math.max(0, x - EDGE).toFixed(1)}px`;
    const total = clockText(player.duration);
    if (total !== shown.total) {
      shown.total = total;
      setText(el.timeTotal, total);
      el.timeline.setAttribute("aria-valuemax", String(Math.round(player.duration)));
    }
    const now = clockText(player.t);
    if (now !== shown.now) {
      shown.now = now;
      setText(el.timeNow, now);
      el.timeline.setAttribute("aria-valuenow", String(Math.round(player.t)));
      el.timeline.setAttribute("aria-valuetext", `${now} of ${total}`);
    }
  }

  // ---------- the cars' cards ----------

  function buildCars() {
    const counts = [0, 0];
    for (let car = 0; car < rec.cars; car++) {
      const team = rec.teams[car] ? 1 : 0;
      const number = ++counts[team];
      names.push(`${NV.TEAMS[team].name} ${number}`);
      const root = make("div", "nv-card");
      root.style.setProperty("--nv-team", teamCss(team));
      const head = make("button", "nv-card-head");
      head.type = "button";
      head.setAttribute("aria-label", `Follow ${names[car]}`);
      head.dataset.tip = `Follow ${names[car]} and see what its bot is thinking`;
      const first = make("span", "nv-card-row");
      const speed = make("span", "nv-card-speed");
      first.append(make("span", "nv-plate", String(number)), make("b", "nv-card-name", names[car]), speed);
      const second = make("span", "nv-card-row");
      const boost = make("span", "nv-boost");
      const fill = make("i");
      boost.appendChild(fill);
      const boostText = make("b", "nv-boost-number");
      second.append(make("span", "nv-boost-label", "Boost"), boost, boostText);
      const state = make("span", "nv-card-state");
      const where = make("span", "nv-state");
      const fast = make("span", "nv-state nv-state-fast", "Supersonic");
      const marks = make("span", "nv-card-marks");
      state.append(where, fast, marks);
      head.append(first, second, state);
      head.addEventListener("click", () => NV.cams.follow(car), { signal: NV.ending.signal });
      root.appendChild(head);
      if (team) el.teamOrange.insertBefore(root, el.minimap);
      else el.teamBlue.appendChild(root);
      cards.push({ root, head, marks, speed, fill, boostText, where, fast, letters: "" });

      const tag = make("div", "nv-tag");
      tag.style.setProperty("--nv-team", teamCss(team));
      const tagBoost = make("span", "nv-tag-boost"), tagSpeed = make("span", "nv-tag-speed");
      const tagFill = make("i");
      const tagBar = make("span", "nv-tag-bar");
      tagBar.appendChild(tagFill);
      tag.append(make("span", "nv-tag-name", names[car]), tagBoost, tagSpeed, tagBar);
      el.tags.appendChild(tag);
      tags.push({ root: tag, boost: tagBoost, speed: tagSpeed, fill: tagFill, on: false });
    }
    setText(el.helpLetters, MECHANICS.map(([letter, name]) => `${letter} ${name.toLowerCase()}`).join(", "));
  }

  // The mechanics marks of every car, read once from the recording (and again as a live game grows):
  // per car a list of [frame, mark] in the order they happened.
  function readMarks() {
    const column = rec.L.flags, shift = NV.data.FLAG.marksShift;
    while (marks.lists.length < rec.cars) marks.lists.push([]);
    for (let frame = marks.read; frame < rec.count; frame++) {
      for (let car = 0; car < rec.cars; car++) {
        const bits = (rec.c(frame, car, column) | 0) >> shift;
        for (let k = 0; bits >> k; k++) if (bits & (1 << k) && MECHANICS[k]) marks.lists[car].push([frame, k]);
      }
    }
    marks.read = rec.count;
  }

  // The marks of a car in the moments before `frame`: their numbers, newest last, each once.
  function recentMarks(car, frame) {
    const list = marks.lists[car], oldest = frame - MARK_SECONDS * rec.fps;
    let lo = 0, hi = list.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (list[mid][0] < oldest) lo = mid + 1;
      else hi = mid;
    }
    let letters = "";
    for (let at = lo; at < list.length && list[at][0] <= frame; at++) {
      if (!letters.includes(`${list[at][1]},`)) letters += `${list[at][1]},`;
    }
    return letters;
  }

  function updateCars(world) {
    if (marks.read < rec.count) readMarks();
    for (const car of world.cars) {
      const card = cards[car.index], tag = tags[car.index];
      const boost = String(Math.round(car.boost)), speed = `${Math.round(car.speed * KMH)} km/h`;
      card.fill.style.width = `${NV.clamp(car.boost, 0, 100)}%`;
      setText(card.boostText, boost);
      setText(card.speed, car.demoed ? "" : speed);
      setText(card.where, car.demoed ? "Demolished" : car.onGround ? "On the ground" : "In the air");
      card.where.classList.toggle("nv-state-air", !car.onGround && !car.demoed);
      card.where.classList.toggle("nv-state-out", car.demoed);
      card.fast.hidden = !car.supersonic || car.demoed;
      card.root.classList.toggle("nv-demolished", car.demoed);
      // A mark lasts one step of the game: its letter stays on the card for a moment.
      const letters = recentMarks(car.index, world.frame);
      if (letters !== card.letters) {
        card.letters = letters;
        card.marks.replaceChildren(...letters.split(",").filter(Boolean).map((k) => {
          const badge = make("span", "nv-mark", MECHANICS[k][0]);
          badge.dataset.tip = MECHANICS[k][1];
          return badge;
        }));
      }
      setText(tag.boost, boost);
      setText(tag.speed, options.tagSpeed ? speed : "");
      tag.fill.style.width = `${NV.clamp(car.boost, 0, 100)}%`;
    }
  }

  // ---------- what the bot is thinking ----------

  // The panel sits inside the followed car's card, or under the strip of cars when the viewer is narrow.
  function placeThinking() {
    const car = NV.cams.car;
    cards.forEach((card, index) => {
      card.root.classList.toggle("nv-followed", index === car);
      card.head.setAttribute("aria-pressed", String(index === car));
    });
    tags.forEach((tag, index) => tag.root.classList.toggle("nv-followed", index === car));
    const show = options.thinking && car >= 0 && car < cards.length;
    el.think.hidden = !show;
    if (!show) return;
    const parent = size.compact ? el.dock : cards[car].root;
    if (el.think.parentNode !== parent) parent.appendChild(el.think);
  }

  function shareRow(list, index) {
    while (list.children.length <= index) {
      const row = make("div", "nv-share");
      const bar = make("span", "nv-share-bar");
      bar.appendChild(make("i"));
      row.append(make("span", "nv-share-name"), bar, make("b"));
      list.appendChild(row);
    }
    return list.children[index];
  }

  function fillShares(list, rows) {
    rows.forEach(([name, share, css], index) => {
      const row = shareRow(list, index);
      row.hidden = false;
      setText(row.children[0], name);
      row.children[1].firstChild.style.width = `${Math.round(NV.clamp(share, 0, 1) * 100)}%`;
      row.children[1].firstChild.style.background = css;
      setText(row.children[2], `${Math.round(share * 100)}%`);
    });
    for (let index = rows.length; index < list.children.length; index++) list.children[index].hidden = true;
  }

  function updateThinking(world) {
    const index = NV.cams.car;
    if (el.think.hidden || index < 0) return;
    const car = world.cars[index];
    const decision = rec.decisionAt(world.frame);
    const figures = decision && decision[index + 1];
    el.think.classList.toggle("nv-think-empty", !figures);
    el.thinkNone.hidden = !!figures;
    if (figures) {
      const [value, sure, random] = figures;
      readValues();
      const reach = Math.max(Math.abs(values.lo), Math.abs(values.hi), 1e-6);
      const part = NV.clamp(Math.abs(value) / reach, 0, 1) * 50;
      el.valueBar.style.width = `${part}%`;
      el.valueBar.style.left = `${value < 0 ? 50 - part : 50}%`;
      el.valueBar.classList.toggle("nv-bad", value < 0);
      setText(el.valueText, `${value >= 0 ? "+" : "\u2212"}${Math.abs(value).toFixed(2)}`);
      el.sureBar.style.width = `${NV.clamp(sure, 0, 1) * 100}%`;
      setText(el.sureText, `${Math.round(sure * 100)}%`);
      el.randomBar.style.width = `${NV.clamp(random, 0, 1) * 100}%`;
      setText(el.randomText, `${Math.round(random * 100)}%`);
    }
    // The stick: steering on the ground, turning and tilting in the air.
    const air = !car.onGround, c = car.controls;
    el.stick.setAttribute("cx", (28 + NV.clamp(air ? c[3] : c[1], -1, 1) * 17).toFixed(1));
    el.stick.setAttribute("cy", (28 + NV.clamp(air ? c[2] : 0, -1, 1) * 17).toFixed(1));
    const push = NV.clamp(c[0], -1, 1) * 22;
    el.throttle.setAttribute("y", (push > 0 ? 28 - push : 28).toFixed(1));
    el.throttle.setAttribute("height", Math.abs(push).toFixed(1));
    el.lampJump.classList.toggle("nv-on", c[5] > 0.5);
    el.lampBoost.classList.toggle("nv-on", car.boosting || c[6] > 0.5);
    el.lampBrake.classList.toggle("nv-on", car.handbrake > 0.5);
    el.lampRoll.classList.toggle("nv-on", air && Math.abs(c[4]) > 0.1);

    const attention = rec.attention, targets = attention ? attention.targets[index] : null;
    const looks = figures && targets ? figures[4] : null;
    el.looksBlock.hidden = !looks;
    if (looks) {
      const rows = looks.map((share, place) => [targets[place], share])
        .filter(([target, share]) => target !== null && target !== undefined && share >= 0.02)
        .sort((p, q) => q[1] - p[1]).slice(0, 5)
        .map(([target, share]) => [targetName(target), share, targetCss(target)]);
      fillShares(el.looks, rows);
    }
    const heard = figures && targets && attention.talk ? figures[5] : null;
    el.heardBlock.hidden = !heard;
    if (heard) {
      const rows = heard.map(([source, share], place) => [targets[place + 1], targets[source], share])
        .filter(([listener, source, share]) => listener != null && source != null && share >= 0.1)
        .sort((p, q) => (q[0] === "itself") - (p[0] === "itself") || q[2] - p[2]).slice(0, 4)
        .map(([listener, source, share]) => [`${targetName(listener)} listens to ${targetName(source)}`, share, paint.chalk]);
      fillShares(el.heard, rows);
    }
  }

  // ---------- on the picture: name tags and attention lines ----------

  function updateTags(world) {
    if (!options.tags) return;
    const camera = NV.camera;
    for (const car of world.cars) {
      const tag = tags[car.index];
      let on = !car.demoed;
      if (on) {
        v.copy(car.pos);
        v.y += 0.6;
        v2.copy(v).applyMatrix4(camera.matrixWorldInverse);
        on = v2.z < -camera.near;
        if (on) {
          v.project(camera);
          const x = (v.x * 0.5 + 0.5) * size.w, y = (0.5 - v.y * 0.5) * size.h;
          on = x > -60 && x < size.w + 60 && y > 0 && y < size.h + 40;
          if (on) tag.root.style.transform = `translate(${Math.round(x)}px, ${Math.round(y) - 8}px) translate(-50%, -100%)`;
        }
      }
      if (on !== tag.on) {
        tag.on = on;
        tag.root.style.visibility = on ? "visible" : "hidden";
      }
    }
  }

  function buildLines() {
    lines.group = new THREE.Group();
    lines.group.name = "attention";
    NV.scene.add(lines.group);
  }

  // A flat strip that always faces the camera; its four corners are placed every frame.
  function line(pool, index, color) {
    while (pool.length <= index) {
      const geometry = new THREE.BufferGeometry();
      const corners = new THREE.BufferAttribute(new Float32Array(12), 3);
      corners.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute("position", corners);
      geometry.setIndex([0, 1, 2, 2, 1, 3]);
      const material = new THREE.MeshBasicMaterial({ color, transparent: true, depthWrite: false, fog: false,
        toneMapped: false, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = 5;
      mesh.frustumCulled = false;
      lines.group.add(mesh);
      pool.push(mesh);
    }
    return pool[index];
  }

  // Where a thing the bot can look at is now: a car (unless demolished), the ball, or the bot's own car.
  function targetPos(world, target, me) {
    if (target === "ball") return world.ball.pos;
    if (target === "itself") return me.pos;
    if (typeof target === "number" && world.cars[target] && !world.cars[target].demoed) return world.cars[target].pos;
    return null;
  }

  // Lays a strip from one point to another, `height` above both and `pixels` wide on the screen at any
  // distance (a width in metres would swell beside the camera and vanish from the top view).
  function stretch(mesh, from, to, height, pixels, opacity) {
    const camera = NV.camera, corners = mesh.geometry.attributes.position;
    camera.getWorldDirection(look);
    const closest = camera.near + 0.3;
    const depthFrom = v.copy(from).sub(camera.position).dot(look) + height * look.y;
    const depthTo = v.copy(to).sub(camera.position).dot(look) + height * look.y;
    if (depthFrom < closest && depthTo < closest) return false;
    // An end behind the camera is pulled forward along the strip to just in front of it.
    ends[0].copy(from);
    ends[1].copy(to);
    if (depthFrom < closest) ends[0].lerp(to, (closest - depthFrom) / (depthTo - depthFrom));
    if (depthTo < closest) ends[1].lerp(from, (closest - depthTo) / (depthFrom - depthTo));
    v.copy(ends[1]).sub(ends[0]);
    if (v.lengthSq() < 0.01) return false;
    // Across the strip and flat to the screen: the width then stays the same wherever the strip is.
    side.crossVectors(look, v);
    if (side.lengthSq() < 1e-8) return false;
    side.normalize();
    const metresPerPixel = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / Math.max(1, size.h);
    for (let end = 0; end < 2; end++) {
      const p = ends[end];
      p.y += height;
      const half = (pixels * metresPerPixel * Math.max(closest, v.copy(p).sub(camera.position).dot(look))) / 2;
      corners.setXYZ(end * 2, p.x + side.x * half, p.y + side.y * half, p.z + side.z * half);
      corners.setXYZ(end * 2 + 1, p.x - side.x * half, p.y - side.y * half, p.z - side.z * half);
    }
    corners.needsUpdate = true;
    mesh.material.opacity = opacity;
    mesh.visible = true;
    return true;
  }

  function updateLines(world) {
    const index = NV.cams.car, attention = rec.attention;
    let looksUsed = 0, talkUsed = 0;
    const me = index >= 0 ? world.cars[index] : null;
    if (attention && options.attention && me && !me.demoed) {
      const decision = rec.decisionAt(world.frame);
      const figures = decision && decision[index + 1];
      const targets = attention.targets[index];
      const looks = figures ? figures[4] : null;
      if (looks) {
        looks.forEach((share, place) => {
          const to = targets[place] === "itself" ? null : targetPos(world, targets[place], me);
          if (!to || share < 0.03) return;
          if (stretch(line(lines.looks, looksUsed, paint.yellow), me.pos, to, 0.6, 1.5 + 11 * share, 0.3 + 0.7 * share)) {
            looksUsed++;
          }
        });
      }
      const heard = figures && attention.talk ? figures[5] : null;
      if (heard) {
        heard.forEach(([source, share], place) => {
          const from = targetPos(world, targets[place + 1], me), to = targetPos(world, targets[source], me);
          if (!from || !to || share < 0.25) return;
          if (stretch(line(lines.talk, talkUsed, paint.chalk), from, to, 1.3, 1.5 + 6 * share, 0.2 + 0.6 * share)) {
            talkUsed++;
          }
        });
      }
    }
    for (let k = looksUsed; k < lines.looks.length; k++) lines.looks[k].visible = false;
    for (let k = talkUsed; k < lines.talk.length; k++) lines.talk[k].visible = false;
  }

  // ---------- the minimap ----------

  // The field from above, turned like the top camera: Blue's goal on the left.
  function drawMinimap(world) {
    const canvas = el.minimapCanvas;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
      canvas.width = Math.round(w * ratio);
      canvas.height = Math.round(h * ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const F = NV.FIELD, S = NV.S;
    const halfLength = F.halfY * S, halfWidth = F.halfX * S, depth = F.goalDepth * S, mouth = F.goalHalfWidth * S;
    const scale = Math.min((w - 8) / (2 * (halfLength + depth)), (h - 8) / (2 * halfWidth));
    const X = (z) => w / 2 + z * scale, Y = (x) => h / 2 - x * scale;
    ctx.lineWidth = 1;
    for (let team = 0; team < 2; team++) {
      const dir = team ? 1 : -1;
      ctx.fillStyle = teamCss(team);
      ctx.globalAlpha = 0.16;
      ctx.fillRect(Math.min(X(0), X(dir * halfLength)), Y(halfWidth), halfLength * scale, 2 * halfWidth * scale);
      ctx.globalAlpha = 0.75;
      ctx.fillRect(Math.min(X(dir * halfLength), X(dir * (halfLength + depth))), Y(mouth), depth * scale, 2 * mouth * scale);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "rgba(243,241,236,0.5)";
    ctx.strokeRect(X(-halfLength), Y(halfWidth), 2 * halfLength * scale, 2 * halfWidth * scale);
    ctx.beginPath();
    ctx.moveTo(X(0), Y(halfWidth));
    ctx.lineTo(X(0), Y(-halfWidth));
    ctx.moveTo(X(0) + 9 * scale, Y(0));
    ctx.arc(X(0), Y(0), 9 * scale, 0, 2 * Math.PI);
    ctx.stroke();
    for (const car of world.cars) {
      if (car.demoed) continue;
      v.set(1, 0, 0).applyQuaternion(car.quat);
      const angle = Math.atan2(-v.x, v.z);
      ctx.save();
      ctx.translate(X(car.pos.z), Y(car.pos.x));
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.moveTo(6, 0);
      ctx.lineTo(-4.5, 4);
      ctx.lineTo(-2.5, 0);
      ctx.lineTo(-4.5, -4);
      ctx.closePath();
      ctx.fillStyle = teamCss(car.team);
      ctx.fill();
      ctx.strokeStyle = car.index === NV.cams.car ? paint.yellow : "rgba(0,0,0,0.55)";
      ctx.lineWidth = car.index === NV.cams.car ? 1.6 : 1;
      ctx.stroke();
      ctx.restore();
    }
    const ball = world.ball.pos;
    ctx.beginPath();
    ctx.arc(X(ball.z), Y(ball.x), 3 + NV.clamp(ball.y / 20, 0, 1) * 2.5, 0, 2 * Math.PI);
    ctx.fillStyle = paint.chalk;
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.6)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // ---------- view options ----------

  function setOption(key, on) {
    if (!(key in DEFAULTS)) return;
    options[key] = !!on;
    NV.setPref(key, options[key]);
    applyOptions();
  }

  function applyOptions() {
    for (const button of el.menu.querySelectorAll("[data-option]")) {
      button.setAttribute("aria-checked", String(options[button.dataset.option]));
    }
    el.minimap.hidden = !options.minimap;
    el.tags.hidden = !options.tags;
    if (NV.cars && NV.cars.showHitboxes) NV.cars.showHitboxes(options.hitboxes);
    placeThinking();
    for (const tag of tags) tag.on = null;
  }

  function buildMenu() {
    for (const key of Object.keys(DEFAULTS)) {
      if (typeof NV.prefs[key] === "boolean") options[key] = NV.prefs[key];
    }
    for (const button of el.menu.querySelectorAll("[data-option]")) {
      button.addEventListener("click", () => setOption(button.dataset.option, !options[button.dataset.option]), { signal: NV.ending.signal });
    }
    el.menu.querySelector('[data-option="attention"]').hidden = !rec.attention;
    for (const [name, label] of QUALITIES) {
      const button = make("button", "", label);
      button.type = "button";
      button.dataset.quality = name;
      button.setAttribute("aria-label", `Image quality: ${label}`);
      button.addEventListener("click", () => setQuality(name), { signal: NV.ending.signal });
      el.qualities.appendChild(button);
    }
    savedQuality = QUALITIES.some(([name]) => name === NV.prefs.quality) ? NV.prefs.quality : "";
    const info = NV.post.info ? NV.post.info() : null;
    quality = savedQuality || (info ? (info.auto ? "auto" : info.tier) : NV.q.tier);
    el.menuButton.addEventListener("click", () => toggleMenu(), { signal: NV.ending.signal });
    el.hide.addEventListener("click", () => setHidden(true), { signal: NV.ending.signal });
    el.show.addEventListener("click", () => setHidden(false), { signal: NV.ending.signal });
    applyOptions();
    updateQuality();
  }

  function setQuality(name) {
    quality = name;
    NV.post.setQuality(name);
    NV.setPref("quality", name);
    updateQuality();
  }

  function updateQuality() {
    for (const button of el.qualities.children) {
      button.setAttribute("aria-pressed", String(button.dataset.quality === quality));
    }
    const info = NV.post.info ? NV.post.info() : null;
    const tier = String((info && info.tier) || NV.q.tier || "");
    const parts = [`Now: ${tier.charAt(0).toUpperCase()}${tier.slice(1)}`];
    const fps = info && info.fps ? info.fps : NV.frameMs ? 1000 / NV.frameMs : 0;
    if (fps) parts.push(`${Math.round(fps)} frames a second`);
    if (info && info.scale && Math.abs(info.scale - 1) > 0.01) parts.push(`${Math.round(info.scale * 100)}% sharpness`);
    const text = parts.join(" \u00b7 ");
    if (text !== shown.quality) {
      shown.quality = text;
      setText(el.qualityInfo, text);
      el.qualityInfo.dataset.tip = info && info.renderer ? `Drawn by: ${info.renderer}` : "";
    }
  }

  function toggleMenu(open) {
    const next = open === undefined ? el.menu.hidden : open;
    if (next === !el.menu.hidden) return;
    el.menu.hidden = !next;
    el.menuButton.setAttribute("aria-expanded", String(next));
    if (next) {
      el.menu.querySelector('[data-option="hitboxes"]').hidden = !(NV.cars && NV.cars.showHitboxes);
      shown.quality = "";
      updateQuality();
    }
  }

  // Hides everything but the score; moving the mouse offers a button that brings the controls back.
  function setHidden(on) {
    host.classList.toggle("nv-hidden", on);
    if (on) {
      toggleMenu(false);
      closeCamSettings();
      showTimer = 4;
      el.show.hidden = false;
    } else el.show.hidden = true;
    measure();
  }

  // ---------- camera settings ----------

  function buildCamSettings() {
    const spec = NV.cams.SETTINGS;
    for (const input of el.camPanel.querySelectorAll("input[data-cam]")) {
      const range = spec[input.dataset.cam];
      input.min = String(range.min);
      input.max = String(range.max);
      input.step = String(range.step);
      input.addEventListener("input", () => NV.cams.setSetting(input.dataset.cam, Number(input.value)), { signal: NV.ending.signal });
    }
    const lock = el.camPanel.querySelector("[data-cam-switch]");
    lock.addEventListener("click", () => NV.cams.setSetting("ballLock", !NV.cams.settings.ballLock), { signal: NV.ending.signal });
    el.camReset.addEventListener("click", () => NV.cams.resetSettings(), { signal: NV.ending.signal });
    el.camGear.addEventListener("click", () => (el.camPanel.hidden ? openCamSettings() : closeCamSettings()), { signal: NV.ending.signal });
    el.camMenu.addEventListener("click", openCamSettings, { signal: NV.ending.signal });
    el.camClose.addEventListener("click", closeCamSettings, { signal: NV.ending.signal });
    NV.on("chase", showCamSettings);
    showCamSettings();
  }

  // The sliders, their numbers (with as many decimals as the step has) and the switch, as the settings are.
  function showCamSettings() {
    const settings = NV.cams.settings, spec = NV.cams.SETTINGS;
    for (const input of el.camPanel.querySelectorAll("input[data-cam]")) {
      const name = input.dataset.cam, range = spec[name], value = settings[name];
      if (Number(input.value) !== value) input.value = String(value);
      input.style.setProperty("--nv-fill", `${(((value - range.min) / (range.max - range.min)) * 100).toFixed(1)}%`);
      const decimals = (String(range.step).split(".")[1] || "").length;
      setText(el.camPanel.querySelector(`[data-cam-value="${name}"]`), value.toFixed(decimals));
    }
    el.camPanel.querySelector("[data-cam-switch]").setAttribute("aria-checked", String(settings.ballLock));
  }

  function openCamSettings() {
    toggleMenu(false);
    showCamSettings();
    el.camPanel.hidden = false;
    el.camGear.setAttribute("aria-expanded", "true");
    el.camClose.focus({ preventScroll: true });
  }

  function closeCamSettings() {
    if (el.camPanel.hidden) return;
    const inside = el.camPanel.contains(document.activeElement);
    el.camPanel.hidden = true;
    el.camGear.setAttribute("aria-expanded", "false");
    if (inside) host.focus({ preventScroll: true });
  }

  // ---------- help ----------

  function openHelp() {
    toggleMenu(false);
    dismissHint();
    el.help.hidden = false;
    el.helpClose.focus({ preventScroll: true });
  }

  function closeHelp() {
    if (el.help.hidden) return;
    el.help.hidden = true;
    host.focus({ preventScroll: true });
  }

  function dismissHint() {
    if (!el.hint.hidden) el.hint.hidden = true;
    if (!NV.prefs.helpSeen) NV.setPref("helpSeen", true);
  }

  function buildHelp() {
    el.helpButton.addEventListener("click", openHelp, { signal: NV.ending.signal });
    el.helpClose.addEventListener("click", closeHelp, { signal: NV.ending.signal });
    el.help.addEventListener("pointerdown", (event) => {
      if (event.target === el.help) closeHelp();
    }, { signal: NV.ending.signal });
    el.fullButton.addEventListener("click", fullScreen, { signal: NV.ending.signal });
    const fullChanged = () => {
      const full = document.fullscreenElement === host;
      el.fullButton.setAttribute("aria-label", full ? "Leave full screen" : "Full screen");
      el.fullButton.dataset.tip = full ? "Back to the normal size" : "Fill the whole screen";
    };
    document.addEventListener("fullscreenchange", fullChanged, { signal: NV.ending.signal });
    el.hintOk.addEventListener("click", dismissHint, { signal: NV.ending.signal });
    el.hintHelp.addEventListener("click", openHelp, { signal: NV.ending.signal });
    if (!NV.prefs.helpSeen && !(NV.options && NV.options.embedded)) {
      el.hint.hidden = false;
      hintTimer = 20;
    }
  }

  // ---------- tooltips ----------

  function showTip(target) {
    const text = (target.dataset.tip || "").replace(/\s+/g, " ");
    if (!text || !host.contains(target)) return;
    const key = target.dataset.key;
    el.tip.replaceChildren(make("span", "", text));
    if (key) el.tip.appendChild(make("kbd", "", key));
    el.tip.hidden = false;
    const frame = host.getBoundingClientRect(), box = target.getBoundingClientRect();
    const w = el.tip.offsetWidth, h = el.tip.offsetHeight;
    let x = box.left - frame.left + box.width / 2 - w / 2, y = box.top - frame.top - h - 8;
    if (target.closest(".nv-menu, .nv-camset")) {
      x = box.left - frame.left - w - 10;
      y = box.top - frame.top + box.height / 2 - h / 2;
      if (x < 6) {
        x = box.left - frame.left + box.width / 2 - w / 2;
        y = box.top - frame.top - h - 8;
      }
    }
    if (y < 6) y = box.bottom - frame.top + 8;
    el.tip.style.left = `${Math.round(NV.clamp(x, 6, Math.max(6, frame.width - w - 6)))}px`;
    el.tip.style.top = `${Math.round(NV.clamp(y, 6, Math.max(6, frame.height - h - 6)))}px`;
  }

  function hideTip() {
    clearTimeout(tip.timer);
    tip.target = null;
    el.tip.hidden = true;
  }

  function buildTips() {
    host.addEventListener("pointerover", (event) => {
      const target = event.pointerType === "touch" ? null : event.target.closest("[data-tip]");
      if (target === tip.target) return;
      hideTip();
      if (!target) return;
      tip.target = target;
      tip.timer = setTimeout(() => showTip(target), 450);
    }, { signal: NV.ending.signal });
    host.addEventListener("pointerleave", hideTip, { signal: NV.ending.signal });
    host.addEventListener("pointerdown", hideTip, { signal: NV.ending.signal });
    host.addEventListener("focusin", (event) => {
      const target = event.target.closest("[data-tip]");
      if (!target || !event.target.matches(":focus-visible")) return;
      hideTip();
      tip.target = target;
      showTip(target);
    }, { signal: NV.ending.signal });
    host.addEventListener("focusout", hideTip, { signal: NV.ending.signal });
    NV.atStop(hideTip);
  }

  // ---------- keyboard, focus and fading ----------

  // The keys are the viewer's while it runs and has the focus, or when it is the only thing on the page.
  function hasKeyboard() {
    const active = document.activeElement;
    if (NV.stopped || !host.isConnected) return false;
    return host.contains(active) || (document.body.classList.contains("nv-page") && (!active || active === document.body));
  }

  function onKey(event) {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || !hasKeyboard()) return;
    const active = document.activeElement;
    const inside = host.contains(active);
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    // A field keeps its keys, but Escape still closes the viewer's own panel it sits in.
    if (active && /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName) && !(key === "Escape" && inside)) return;
    // A control reached with the keyboard keeps Space, Enter and Tab for itself.
    const onControl = inside && active !== host && active.matches(":focus-visible");
    activeAt = 0;
    if (key === "Escape") {
      if (!el.help.hidden) closeHelp();
      else if (!el.camPanel.hidden) closeCamSettings();
      else if (!el.menu.hidden) toggleMenu(false);
      else if (canClose()) close();
      else if (host.classList.contains("nv-hidden")) setHidden(false);
      else return;
    } else if (key === "?" || key === "F1") {
      if (el.help.hidden) openHelp();
      else closeHelp();
    } else if (!el.help.hidden) return;
    else if (key === " ") {
      if (onControl) return;
      togglePlay();
    } else if (key.startsWith("Arrow") && NV.cams.mode === "free") NV.cams.key(event.code, true);
    else if (key === "ArrowLeft") seekBy(-5);
    else if (key === "ArrowRight") seekBy(5);
    else if (key === ",") stepFrames(-1);
    else if (key === ".") stepFrames(1);
    else if (key >= "1" && key <= "5") NV.cams.set(NV.cams.MODES[Number(key) - 1]);
    else if (key === "Tab") {
      if (event.shiftKey || onControl) return;
      NV.cams.next(1);
    } else if (key === "]") NV.cams.next(1);
    else if (key === "[") NV.cams.next(-1);
    else if (key === "+" || key === "=") changeSpeed(1);
    else if (key === "-" || key === "_") changeSpeed(-1);
    else if (key === "h") setHidden(!host.classList.contains("nv-hidden"));
    else if (key === "f") fullScreen();
    else if (key === "l") goLive();
    else if (!NV.cams.key(event.code, true)) return;
    event.preventDefault();
  }

  function buildKeys() {
    if (!host.hasAttribute("tabindex")) host.tabIndex = -1;
    // Inside another page the keys are heard on the viewer's own element (they are the viewer's only while
    // it has the focus), so that page can keep them from its own key handlers.
    const keys = NV.options && NV.options.embedded ? host : document;
    keys.addEventListener("keydown", onKey, { signal: NV.ending.signal });
    document.addEventListener("keyup", (event) => NV.cams.key(event.code, false), { signal: NV.ending.signal });
    // A click leaves the keyboard with the viewer, not with the button that was clicked.
    host.addEventListener("pointerdown", () => {
      activeAt = 0;
      if (!host.contains(document.activeElement)) host.focus({ preventScroll: true });
    }, { signal: NV.ending.signal });
    host.addEventListener("click", (event) => {
      if (event.detail > 0 && event.target.closest("button") && el.help.hidden) host.focus({ preventScroll: true });
      if (!el.menu.hidden && !event.target.closest(".nv-menu") && !event.target.closest('[data-nv="menuButton"]')) {
        toggleMenu(false);
      }
    }, { signal: NV.ending.signal });
    host.addEventListener("pointermove", (event) => {
      activeAt = 0;
      overControl = !!event.target.closest(".nv-bar, .nv-tools, .nv-menu, .nv-camset, .nv-card, .nv-hint, .nv-show");
      if (host.classList.contains("nv-hidden")) {
        showTimer = 4;
        el.show.hidden = false;
      }
    }, { signal: NV.ending.signal });
    host.addEventListener("pointerleave", () => {
      overControl = false;
    }, { signal: NV.ending.signal });
  }

  // Time on the wall clock, not the sum of frame times: a slow frame must not keep the controls up longer.
  function updateFading(dt) {
    const active = document.activeElement, now = performance.now();
    const busy = !(player.playing || player.atLive) || overControl || timeline.drag || !el.menu.hidden || !el.help.hidden
      || !el.camPanel.hidden || !el.hint.hidden
      || (host.contains(active) && active !== host && active.matches(":focus-visible"));
    if (busy || !activeAt) activeAt = now;
    host.classList.toggle("nv-idle", now - activeAt > IDLE_SECONDS * 1000);
    if (hintTimer > 0 && !el.hint.hidden && (hintTimer -= dt) <= 0) dismissHint();
    if (showTimer > 0 && !overControl && (showTimer -= dt) <= 0) el.show.hidden = true;
  }

  // ---------- layout ----------

  // The space the controls take at each edge: the cameras keep the play clear of it.
  function measure() {
    const hidden = host.classList.contains("nv-hidden"), safe = NV.cams.safe;
    const columns = !hidden && !size.compact;
    safe.top = el.top.offsetHeight + (size.compact && !hidden ? el.teamBlue.offsetHeight + el.dock.offsetHeight : 0);
    safe.bottom = hidden ? 0 : el.bar.offsetHeight;
    safe.left = columns ? el.teamBlue.offsetWidth : 0;
    safe.right = columns ? el.teamOrange.offsetWidth : 0;
    host.style.setProperty("--nv-bar", `${safe.bottom}px`);
  }

  function layout({ w, h }) {
    size.w = w;
    size.h = h;
    // The remembered image quality, set once the picture has its first size.
    if (savedQuality) NV.post.setQuality(savedQuality);
    savedQuality = "";
    size.compact = w < 620;
    host.classList.toggle("nv-mid", w < 1080);
    host.classList.toggle("nv-nologo", w < 960);
    host.classList.toggle("nv-narrow", w < 800);
    host.classList.toggle("nv-compact", size.compact);
    host.classList.toggle("nv-low", h < 520);
    placeThinking();
    measure();
    timeline.dirty = true;
  }

  // ---------- start-up and the frame ----------

  function build() {
    host = NV.host;
    rec = NV.rec;
    player = NV.player;
    for (const node of host.querySelectorAll("[data-nv]")) el[node.dataset.nv] = node;
    const style = getComputedStyle(host);
    const colour = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    paint.team = [colour("--nv-blue", paint.team[0]), colour("--nv-orange", paint.team[1])];
    for (const name of ["chalk", "dim", "yellow"]) paint[name] = colour(`--nv-${name}`, paint[name]);
    buildTop();
    buildCars();
    buildBar();
    buildTimeline();
    buildMenu();
    buildCamSettings();
    buildHelp();
    buildTips();
    buildKeys();
    buildLines();
    syncCamera();
    NV.on("resize", layout);
    const watch = new ResizeObserver(measure);
    for (const node of [el.bar, el.top, el.dock, el.teamBlue]) watch.observe(node);
    NV.atStop(() => watch.disconnect());
  }

  function update(world, dt) {
    updateTop(world);
    updateBar();
    updateTimeline(dt);
    // Numbers that change every frame are unreadable: the cards refresh a few times a second.
    slow += dt;
    if (slow >= 0.07 || NV.player.cursor !== shownFrame) {
      shownFrame = NV.player.cursor;
      slow = 0;
      updateCars(world);
      updateThinking(world);
      if (!el.menu.hidden) updateQuality();
    }
    updateTags(world);
    updateLines(world);
    if (options.minimap && !host.classList.contains("nv-hidden")) drawMinimap(world);
    updateFading(dt);
  }

  return {
    build,
    update,
    options,
    names,
    setOption,
    setQuality,
    setHidden,
    toggleMenu,
    openCamSettings,
    closeCamSettings,
    openHelp,
    closeHelp,
    dismissHint,
  };
})();
