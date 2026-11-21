// Practice starts: a moment of this game (a recording made with `nitro watch --moments` keeps each car's
// exact state at every decision) or a start placed by hand, saved as a moments file for `nitro practice`,
// which adds it to the bot's training starts. A host page can take the file itself (`options.practice`).
// The editor is a view from above, as Blue sees the field (Blue attacks up): drag the car, its nose (to
// turn it) and the ball; heights, speeds and boost are typed in.

NV.practice = (() => {
  // The columns of a moment (nitro_core::snapshots::COLUMNS) and where its parts sit.
  const COLUMNS = ("cx,cy,cz,cvx,cvy,cvz,cwx,cwy,cwz,fx,fy,fz,rx,ry,rz,ux,uy,uz,onGround,hasJumped," +
    "hasDoubleJumped,hasFlipped,isJumping,jumpTime,isFlipping,flipTime,ftx,fty,ftz,airTime,airTimeSinceJump," +
    "boost,bx,by,bz,bvx,bvy,bvz,bwx,bwy,bwz").split(",");
  const M = 41, CAR_REST = 17, BALL_REST = 93.15;
  const FIELD = { x: 4096, y: 5120, goal: 893, depth: 880 };
  const el = {};
  const edit = { car: [0, -2000], yaw: Math.PI / 2, ball: [0, 0], drag: "" };
  let rec, moments = null, run = "", last = "";

  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const on = (node, type, fn) => node.addEventListener(type, fn, { signal: NV.ending.signal });

  async function floats(base64) {
    const text = atob(base64);
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Float32Array(await new Response(stream).arrayBuffer());
  }

  // A 32-bit number's text that reads back as the same number.
  const num = (v) => (Object.is(v, -0) || v === 0 ? "0" : String(Math.fround(v)));
  const csv = (rows) => [["group", ...COLUMNS].join(","), ...rows.map(([group, v]) =>
    [group, ...Array.from(v, num)].join(","))].join("\n") + "\n";

  // The moment of car `car` at the decision nearest time `t` (seconds), or null without moments.
  function momentAt(t, car) {
    if (!moments) return null;
    const list = rec.decisions;
    let best = 0;
    for (let i = 1; i < list.length; i++) if (Math.abs(list[i][0] / rec.fps - t) < Math.abs(list[best][0] / rec.fps - t)) best = i;
    const at = (best * rec.cars + car) * M;
    return { values: moments.subarray(at, at + M), frame: list[best][0] };
  }

  function build() {
    rec = NV.rec;
    const data = NV.recData || {};
    run = data.run || "";
    const tools = NV.host && NV.host.querySelector("[data-nv=top] .nv-tools");
    if (!tools) return;
    el.open = make("button", "nv-tool nv-practice-open", "Practice");
    el.open.type = "button";
    el.open.dataset.tip = "Add a moment of this game, or a start you place, to the bot's training starts";
    tools.prepend(el.open);
    el.panel = make("div", "nv-practice");
    el.panel.dataset.nv = "practice";
    el.panel.setAttribute("role", "dialog");
    el.panel.setAttribute("aria-label", "Practice starts");
    el.panel.hidden = true;
    const title = make("div", "nv-menu-title", "Practice starts");
    el.close = make("button", "nv-menu-action", "Close");
    el.close.type = "button";
    const head = make("div", "nv-practice-head");
    head.append(title, el.close);
    el.intro = make("p", "nv-practice-text");
    el.moment = make("button", "nv-big", "Practice this moment");
    el.moment.type = "button";
    el.moment.dataset.tip = "The followed car and the ball, exactly as they are now";
    el.editorOpen = make("button", "nv-menu-action", "Place a start by hand");
    el.editorOpen.type = "button";
    el.editor = make("div", "nv-practice-editor");
    el.editor.hidden = true;
    el.field = make("canvas", "nv-practice-field");
    el.field.width = 240;
    el.field.height = 330;
    const inputs = make("div", "nv-practice-inputs");
    const field = (key, label, value, step) => {
      const row = make("label", "nv-practice-input");
      const input = make("input");
      input.type = "number";
      input.step = String(step);
      input.value = String(value);
      input.dataset.field = key;
      row.append(make("span", "", label), input);
      inputs.append(row);
      el[key] = input;
      on(input, "input", drawField);
    };
    field("carZ", "Car height", CAR_REST, 10);
    field("carSpeed", "Car speed (forward)", 0, 50);
    field("boost", "Boost (0-100)", 33.3, 1);
    field("ballZ", "Ball height", BALL_REST, 10);
    field("ballVx", "Ball speed x", 0, 50);
    field("ballVy", "Ball speed y", 0, 50);
    field("ballVz", "Ball speed up", 0, 50);
    el.saveStart = make("button", "nv-big", "Save this start");
    el.saveStart.type = "button";
    el.editor.append(el.field, inputs, el.saveStart);
    el.result = make("div", "nv-practice-result");
    el.result.hidden = true;
    el.command = make("code", "nv-practice-command");
    el.copy = make("button", "nv-menu-action", "Copy command");
    el.copy.type = "button";
    el.result.append(make("p", "nv-practice-text"), el.command, el.copy);
    el.panel.append(head, el.intro, el.moment, el.editorOpen, el.editor, el.result);
    NV.host.append(el.panel);
    on(el.open, "click", () => toggle(el.panel.hidden));
    on(el.close, "click", () => toggle(false));
    on(el.moment, "click", practiseMoment);
    on(el.editorOpen, "click", () => openEditor(!el.editor.hidden ? false : true));
    on(el.saveStart, "click", saveEdited);
    on(el.copy, "click", () => navigator.clipboard && navigator.clipboard.writeText(el.command.textContent));
    fieldEvents();
    if (data.moments && data.moments.values) {
      floats(data.moments.values).then((values) => {
        moments = values;
        intro();
      });
    }
    intro();
  }

  function intro() {
    el.moment.disabled = !moments;
    el.intro.textContent = moments
      ? "Pause where the bot should do better, follow its car, and press Practice this moment: its games can " +
        "start from there. Or place the car and the ball yourself."
      : "This recording keeps no exact moments (record it with nitro watch --moments to practise moments " +
        "of the game). You can still place a start by hand.";
  }

  function toggle(open) {
    el.panel.hidden = !open;
    if (open) NV.player.playing = false;
  }

  function followed() {
    const car = NV.cams && Number.isInteger(NV.cams.car) ? NV.cams.car : 0;
    return Math.min(Math.max(car, 0), rec.cars - 1);
  }

  // Hands the file over: to the host page when it takes it, else as a download with the command to add it.
  function hand(text, what) {
    last = text;
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    const name = `practice-${stamp}.csv`;
    el.result.hidden = false;
    const said = el.result.querySelector("p");
    if (NV.options && typeof NV.options.practice === "function") {
      el.command.textContent = "";
      el.copy.hidden = true;
      said.textContent = "Adding\u2026";
      Promise.resolve()
        .then(() => NV.options.practice(text, { run, name }))
        .then((message) => { said.textContent = message || `${what} added to the bot's practice starts.`; })
        .catch((err) => { said.textContent = `Not added: ${(err && err.message) || err}`; });
      return;
    }
    const link = make("a");
    link.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10_000);
    said.textContent = `${what} saved as ${name} (in your downloads). To add it to the bot's training starts, ` +
      "run this in NitroLearn's folder (it is used from the bot's next start):";
    const nitro = /Windows/.test(navigator.userAgent) ? ".\\nitro" : "./nitro.sh";
    el.command.textContent = `${nitro} practice ${run || "<bot>"} "${name}" --weight 0.2`;
    el.copy.hidden = false;
  }

  function practiseMoment() {
    const car = followed();
    const moment = momentAt(NV.player.t, car);
    if (!moment) return;
    const names = (NV.ui && NV.ui.names) || [];
    hand(csv([["practice", moment.values]]), `The moment of ${names[car] || `car ${car + 1}`}`);
  }

  // The editor starts from the followed car and the ball as shown now (Blue's view of the field).
  function openEditor(open) {
    el.editor.hidden = !open;
    if (!open) return;
    const car = followed();
    const moment = momentAt(NV.player.t, car);
    if (moment) {
      const v = moment.values;
      edit.car = [v[0], v[1]];
      edit.yaw = Math.atan2(v[10], v[9]);
      edit.ball = [v[32], v[33]];
      el.carZ.value = v[2].toFixed(1);
      el.carSpeed.value = (v[3] * v[9] + v[4] * v[10] + v[5] * v[11]).toFixed(0);
      el.boost.value = v[31].toFixed(1);
      el.ballZ.value = v[34].toFixed(1);
      el.ballVx.value = v[35].toFixed(0);
      el.ballVy.value = v[36].toFixed(0);
      el.ballVz.value = v[37].toFixed(0);
    }
    drawField();
  }

  // Numbers typed in, held to what the arena allows.
  const read = (input, lo, hi, fallback) => {
    const v = Number(input.value);
    return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
  };

  // The start as a moment: the car level on its wheels at rest height (or in the air above it, with its
  // jump still to use), moving forward at its speed; the ball still turning by nothing.
  function editedMoment() {
    const v = new Float32Array(M);
    const z = read(el.carZ, CAR_REST, 1900, CAR_REST);
    const speed = read(el.carSpeed, -2300, 2300, 0);
    const [c, s] = [Math.cos(edit.yaw), Math.sin(edit.yaw)];
    v.set([edit.car[0], edit.car[1], z, speed * c, speed * s, 0, 0, 0, 0, c, s, 0, -s, c, 0, 0, 0, 1], 0);
    v[18] = z <= CAR_REST + 0.5 ? 1 : 0;
    v[31] = read(el.boost, 0, 100, 33.3);
    v.set([edit.ball[0], edit.ball[1], read(el.ballZ, BALL_REST, 1950, BALL_REST),
      read(el.ballVx, -6000, 6000, 0), read(el.ballVy, -6000, 6000, 0), read(el.ballVz, -6000, 6000, 0)], 32);
    return v;
  }

  function saveEdited() {
    hand(csv([["placed", editedMoment()]]), "The start");
  }

  // ---------- the field from above ----------

  const SCALE = 240 / (2 * FIELD.x + 400);
  const toCanvas = ([x, y]) => [120 + x * SCALE, 165 - y * SCALE];
  const fromCanvas = (px, py) => [(px - 120) / SCALE, (165 - py) / SCALE];
  const nose = () => [edit.car[0] + Math.cos(edit.yaw) * 260, edit.car[1] + Math.sin(edit.yaw) * 260];

  function drawField() {
    if (el.editor.hidden) return;
    const ctx = el.field.getContext("2d");
    ctx.clearRect(0, 0, el.field.width, el.field.height);
    ctx.strokeStyle = "rgba(243,241,236,0.4)";
    ctx.lineWidth = 1;
    const [x0, y0] = toCanvas([-FIELD.x, FIELD.y]), [x1, y1] = toCanvas([FIELD.x, -FIELD.y]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.beginPath();
    ctx.moveTo(x0, 165);
    ctx.lineTo(x1, 165);
    ctx.stroke();
    for (const [team, sign] of [["#FF8A2B", 1], ["#5B97FF", -1]]) {
      const [gx, gy] = toCanvas([-FIELD.goal, sign * FIELD.y]);
      ctx.fillStyle = team;
      ctx.fillRect(gx, sign > 0 ? gy - FIELD.depth * SCALE : gy, 2 * FIELD.goal * SCALE, FIELD.depth * SCALE);
    }
    const [bx, by] = toCanvas(edit.ball);
    ctx.fillStyle = "#F3F1EC";
    ctx.beginPath();
    ctx.arc(bx, by, Math.max(4, 93 * SCALE * 2.5), 0, 2 * Math.PI);
    ctx.fill();
    const [cx, cy] = toCanvas(edit.car), [nx, ny] = toCanvas(nose());
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-edit.yaw);
    ctx.fillStyle = "#5B97FF";
    ctx.fillRect(-118 * SCALE * 2.5, -42 * SCALE * 2.5, 236 * SCALE * 2.5, 84 * SCALE * 2.5);
    ctx.restore();
    ctx.strokeStyle = "#FFD60A";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(nx, ny);
    ctx.stroke();
    ctx.fillStyle = "#FFD60A";
    ctx.beginPath();
    ctx.arc(nx, ny, 4, 0, 2 * Math.PI);
    ctx.fill();
  }

  function fieldEvents() {
    const where = (event) => {
      const box = el.field.getBoundingClientRect();
      return [((event.clientX - box.left) * el.field.width) / box.width,
        ((event.clientY - box.top) * el.field.height) / box.height];
    };
    const near = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) < 10;
    on(el.field, "pointerdown", (event) => {
      const p = where(event);
      edit.drag = near(p, toCanvas(nose())) ? "nose" : near(p, toCanvas(edit.ball)) ? "ball"
        : near(p, toCanvas(edit.car)) ? "car" : "";
      if (edit.drag) el.field.setPointerCapture(event.pointerId);
    });
    on(el.field, "pointermove", (event) => {
      if (!edit.drag) return;
      const [x, y] = fromCanvas(...where(event));
      const inside = [NV.clamp(x, -FIELD.x + 100, FIELD.x - 100), NV.clamp(y, -FIELD.y + 100, FIELD.y - 100)];
      if (edit.drag === "nose") edit.yaw = Math.atan2(y - edit.car[1], x - edit.car[0]);
      else if (edit.drag === "ball") edit.ball = inside;
      else edit.car = inside;
      drawField();
    });
    const stop = () => { edit.drag = ""; };
    on(el.field, "pointerup", stop);
    on(el.field, "pointercancel", stop);
  }

  // For tests and hosts: a start placed with these numbers, as the editor would save it.
  function place({ car, yaw, ball } = {}) {
    if (car) edit.car = car;
    if (yaw !== undefined) edit.yaw = yaw;
    if (ball) edit.ball = ball;
    drawField();
  }

  return {
    build,
    toggle,
    openEditor,
    place,
    momentAt,
    editedMoment,
    csv,
    get last() { return last; },
    get ready() { return !!moments; },
  };
})();
