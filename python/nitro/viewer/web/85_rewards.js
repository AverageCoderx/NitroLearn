// The reward chart under the game (recordings made with `nitro watch --rewards`): what each reward paid the
// followed car in every step, stacked (pay above the line, charges below), in step with the timeline. A
// click jumps to the biggest pay or charge near the pointer; the legend hides and shows rewards.

NV.rewards = (() => {
  // Seven rewards get their own colour (the largest by total size); the rest are drawn together as "Other".
  const HUES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9"];
  const OTHER = "#8a877f";
  const HEIGHT = 74;
  const SNAP = 6; // pixels a click looks to each side for the biggest step
  const state = { on: false, ready: false, car: -1, dirty: true, hidden: new Set(), top: [], hover: -1 };
  const el = {};
  let rec, data, pay, steps, rewards, cars, series, scale = 1, columns = [];

  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
  const signed = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v));

  async function floats(base64) {
    const text = atob(base64);
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Float32Array(await new Response(stream).arrayBuffer());
  }

  // pay of reward r to car c in step s
  const at = (s, r, c) => pay[(s * rewards + r) * cars + c];

  function car() {
    const followed = NV.cams && Number.isInteger(NV.cams.car) ? NV.cams.car : 0;
    return Math.min(Math.max(followed, 0), cars - 1);
  }

  // The series drawn for `c`: the seven largest rewards by total size, then the rest as one.
  function seriesFor(c) {
    const size = new Float64Array(rewards);
    for (let s = 0; s < steps; s++) for (let r = 0; r < rewards; r++) size[r] += Math.abs(at(s, r, c));
    const order = [...Array(rewards).keys()].filter((r) => size[r] > 0).sort((a, b) => size[b] - size[a]);
    const own = rewards <= HUES.length + 1 ? order : order.slice(0, HUES.length);
    const rest = order.filter((r) => !own.includes(r));
    const list = own.map((r, k) => ({ name: data.labels[r], members: [r], colour: HUES[k], size: size[r] }));
    if (rest.length) {
      list.push({ name: "Other", members: rest, colour: OTHER, size: rest.reduce((sum, r) => sum + size[r], 0) });
    }
    return list;
  }

  function value(s, item, c) {
    let sum = 0;
    for (const r of item.members) sum += at(s, r, c);
    return sum;
  }

  function stepTotal(s, c) {
    let sum = 0;
    for (const item of series) if (!state.hidden.has(item.name)) sum += value(s, item, c);
    return sum;
  }

  function build() {
    rec = NV.rec;
    data = NV.recData && NV.recData.rewards;
    if (!data || !data.pay || !NV.host) return;
    const bar = NV.host.querySelector("[data-nv=bar]");
    if (!bar) return;
    el.panel = make("div", "nv-rewards");
    el.panel.dataset.nv = "rewards";
    const head = make("div", "nv-rewards-head");
    el.title = make("span", "nv-rewards-title", "Rewards");
    el.toggle = make("button", "nv-menu-action nv-rewards-toggle", "Hide");
    el.toggle.type = "button";
    el.toggle.dataset.tip = "Show or hide the reward chart";
    el.toggle.dataset.key = "R";
    el.legend = make("div", "nv-rewards-legend");
    head.append(el.title, el.legend, el.toggle);
    el.plot = make("div", "nv-rewards-plot");
    el.canvas = make("canvas");
    el.head = make("div", "nv-rewards-head-line");
    el.peek = make("div", "nv-timeline-peek nv-rewards-peek");
    el.peek.hidden = true;
    el.plot.append(el.canvas, el.head, el.peek);
    el.panel.append(head, el.plot);
    bar.prepend(el.panel);
    el.toggle.addEventListener("click", () => show(!state.on), { signal: NV.ending.signal });
    el.plot.addEventListener("pointermove", (event) => hover(event.clientX), { signal: NV.ending.signal });
    el.plot.addEventListener("pointerleave", () => {
      state.hover = -1;
      el.peek.hidden = true;
    }, { signal: NV.ending.signal });
    el.plot.addEventListener("click", (event) => jump(event.clientX), { signal: NV.ending.signal });
    document.addEventListener("keydown", (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.defaultPrevented) return;
      const active = document.activeElement;
      if (active && /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName)) return;
      const ours = NV.host.contains(active) || (document.body.classList.contains("nv-page") &&
        (!active || active === document.body));
      if (!ours) return;
      if (event.key === "r" || event.key === "R") show(!state.on);
    }, { signal: NV.ending.signal });
    const watch = new ResizeObserver(() => { state.dirty = true; });
    watch.observe(el.plot);
    NV.atStop(() => watch.disconnect());
    show(true);
    floats(data.pay).then((values) => {
      pay = values;
      rewards = data.labels.length;
      cars = rec.cars;
      steps = Math.min(data.frames.length, Math.floor(pay.length / Math.max(1, rewards * cars)));
      state.ready = true;
      state.dirty = true;
    });
  }

  function show(on) {
    state.on = on;
    el.plot.hidden = !on;
    el.legend.hidden = !on;
    el.toggle.textContent = on ? "Hide" : "Rewards";
    state.dirty = true;
  }

  // The plot's frame range follows the timeline's own (from the first to the last frame).
  const lastFrame = () => Math.max(1, rec.count - 1);
  function x(frame, w) {
    return 8 + (frame / lastFrame()) * Math.max(1, w - 16);
  }
  function frameAt(clientX) {
    const box = el.plot.getBoundingClientRect();
    return NV.clamp((clientX - box.left - 8) / Math.max(1, box.width - 16), 0, 1) * lastFrame();
  }

  // The step drawn in each pixel column: the biggest by size, so a spike is never averaged away.
  function draw() {
    const c = car();
    if (c !== state.car) {
      state.car = c;
      series = seriesFor(c);
      legend();
    }
    const canvas = el.canvas, w = el.plot.clientWidth, h = el.plot.clientHeight || HEIGHT;
    if (w < 24) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    columns = new Array(Math.ceil(w)).fill(-1);
    const best = new Float32Array(columns.length);
    let up = 0, down = 0;
    const sides = [];
    for (let s = 0; s < steps; s++) {
      let pos = 0, neg = 0;
      for (const item of series) {
        if (state.hidden.has(item.name)) continue;
        const v = value(s, item, c);
        if (v > 0) pos += v;
        else neg -= v;
      }
      sides.push(Math.max(pos, neg));
      const col = Math.round(x(data.frames[s], w));
      if (col >= 0 && col < columns.length && pos + neg >= best[col] && pos + neg > 0) {
        best[col] = pos + neg;
        columns[col] = s;
      }
      up = Math.max(up, pos);
      down = Math.max(down, neg);
    }
    // One goal can pay a hundred times a step's usual pay: the scale stops at the 98th percentile and taller
    // steps are cut, with a mark on top.
    const sorted = sides.filter((v) => v > 0).sort((a, b) => a - b);
    const cap = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))] : 1;
    const middle = Math.round(down > 0 && up > 0 ? h * (up / (up + down)) : up > 0 ? h - 6 : 6);
    const room = Math.max(4, Math.max(middle - 4, h - 4 - middle));
    scale = room / Math.max(cap, 1e-9);
    ctx.fillStyle = "rgba(243,241,236,0.16)";
    ctx.fillRect(8, middle, w - 16, 1);
    for (let col = 0; col < columns.length; col++) {
      const s = columns[col];
      if (s < 0) continue;
      let top = middle, bottom = middle + 1;
      for (const item of series) {
        if (state.hidden.has(item.name)) continue;
        const v = value(s, item, c);
        if (!v) continue;
        const size = Math.abs(v) * scale;
        ctx.fillStyle = item.colour;
        if (v > 0) {
          const from = Math.max(2, top - size);
          ctx.fillRect(col - 1, from, 2, top - from);
          top = from;
        } else {
          const to = Math.min(h - 2, bottom + size);
          ctx.fillRect(col - 1, bottom, 2, to - bottom);
          bottom = to;
        }
      }
      ctx.fillStyle = "#F3F1EC";
      if (top <= 2) ctx.fillRect(col - 2, 0, 4, 2);
      if (bottom >= h - 2) ctx.fillRect(col - 2, h - 2, 4, 2);
    }
  }

  function legend() {
    el.legend.replaceChildren(...series.map((item) => {
      const chip = make("button", "nv-rewards-chip");
      chip.type = "button";
      chip.dataset.reward = item.name;
      chip.setAttribute("aria-pressed", String(!state.hidden.has(item.name)));
      chip.dataset.tip = item.name === "Other"
        ? `Every other reward: ${item.members.map((r) => data.labels[r]).join(", ")}`
        : `Show or hide ${item.name}`;
      const swatch = make("i");
      swatch.style.background = item.colour;
      let total = 0;
      for (let s = 0; s < steps; s++) total += value(s, item, state.car);
      chip.append(swatch, make("span", "", item.name), make("b", "", signed(total)));
      chip.addEventListener("click", () => {
        if (state.hidden.has(item.name)) state.hidden.delete(item.name);
        else state.hidden.add(item.name);
        chip.setAttribute("aria-pressed", String(!state.hidden.has(item.name)));
        state.dirty = true;
      }, { signal: NV.ending.signal });
      return chip;
    }));
    const names = NV.ui && NV.ui.names;
    el.title.textContent = `Rewards · ${(names && names[state.car]) || `car ${state.car + 1}`}`;
  }

  // The step near the pointer with the biggest pay or charge.
  function nearest(clientX) {
    const box = el.plot.getBoundingClientRect();
    const col = Math.round(clientX - box.left);
    let found = -1, size = -1;
    for (let k = Math.max(0, col - SNAP); k <= Math.min(columns.length - 1, col + SNAP); k++) {
      const s = columns[k];
      if (s < 0) continue;
      const v = Math.abs(stepTotal(s, state.car));
      if (v > size) {
        size = v;
        found = s;
      }
    }
    if (found >= 0) return found;
    const frame = frameAt(clientX);
    let close = 0;
    for (let s = 0; s < steps; s++) if (Math.abs(data.frames[s] - frame) < Math.abs(data.frames[close] - frame)) close = s;
    return close;
  }

  function hover(clientX) {
    if (!state.ready) return;
    const s = nearest(clientX);
    state.hover = s;
    const parts = series
      .filter((item) => !state.hidden.has(item.name))
      .map((item) => [item.name, value(s, item, state.car)])
      .filter(([, v]) => v)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 4)
      .map(([name, v]) => `${name} ${signed(v)}`);
    const seconds = data.frames[s] / rec.fps;
    const clock = `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
    el.peek.textContent = `${clock} · ${parts.length ? parts.join(" · ") : "nothing paid"}`;
    el.peek.hidden = false;
    const box = el.plot.getBoundingClientRect();
    const half = el.peek.offsetWidth / 2;
    el.peek.style.left = `${NV.clamp(clientX - box.left, half, Math.max(half, box.width - half))}px`;
  }

  // A click goes to the step's first frame: the moment just before what it paid for.
  function jump(clientX) {
    if (!state.ready) return;
    const s = nearest(clientX);
    const start = s > 0 ? data.frames[s - 1] : 0;
    NV.player.seek(start / rec.fps);
  }

  function update() {
    if (!el.panel || !state.ready || !state.on) return;
    if (state.dirty || car() !== state.car) {
      state.dirty = false;
      draw();
    }
    const w = el.plot.clientWidth;
    const px = NV.clamp(x(NV.player.t * rec.fps, w), 8, w - 8);
    el.head.style.transform = `translateX(${px.toFixed(1)}px)`;
  }

  // For tests and hosts: the pay of every reward to car `c` in the step ending at or after `frame`.
  function payAt(frame, c = car()) {
    if (!state.ready) return null;
    let s = 0;
    while (s < steps - 1 && data.frames[s] < frame) s++;
    const out = {};
    for (let r = 0; r < rewards; r++) out[data.labels[r]] = at(s, r, c);
    return out;
  }

  return { build, update, show, payAt, state, get series() { return series; } };
})();
