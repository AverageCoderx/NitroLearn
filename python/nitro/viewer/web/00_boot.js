// Shared names of the match viewer. The files in this folder are joined in name order into one script and
// share the object NV. Each scene part (arena, cars, ball, fx, post, cams, ui) is one object on NV with a
// build() called once and an update(world, dt) called every drawn frame.
//
// Units and axes: the scene is in metres, 1 unit = 100 uu. Rocket League's axes (x, y, z up, left-handed)
// map to three.js as (x, z, y); the swap turns the left-handed frame into three's right-handed one, so a
// car's right side stays on its right. Blue defends -y (scene -z), Orange +y.
//
// The world every part reads (NV.world, filled by NV.data.sample):
//   t, frame            playback time in seconds, and the frame it falls in
//   ball                { pos, vel (m/s), quat, speed (uu/s) }
//   cars[i]             { index, team (0 Blue, 1 Orange), body (0..5, NV.BODIES), pos, vel (m/s), quat,
//                         speed (uu/s), boost (0..100), onGround, demoed, supersonic, boosting, jumped,
//                         doubleJumped, flipped, wheelContact[4] (front left, front right, back left, back
//                         right), roll (uu rolled by the wheels so far), steerAngle (rad, + = left),
//                         throttle, steer, handbrake, controls[8], marks (mechanics bits), respawn
//                         (seconds until a demolished car is back) }
//   pads[i]             { active, wait (seconds until it is back) }
//   score               [blue, orange];   clock: seconds since the recording began
// Events (NV.on(name, fn)), sent while playing forward: "goal" {team, pos}, "touch" {car, pos},
// "demo" {car, pos}, "restart" {}, and at any time: "seek" {}, "quality" {q}, "resize" {w, h},
// "camera" {mode, car}, "follow" {car}, "chase" {...settings} (a chase camera setting changed), "title"
// {title} and "live-ended" {} (a live game), and once while
// starting: "building" {} (the scene is about to be made) and "drawn" {} (its first picture is drawn).
//
// One NV shows one game: NV.start once, and NV.stop ends it for good. A page that shows several games in
// its life makes a viewer for each (`nitro.viewer.maker`).

const NV = {
  S: 0.01,
  listeners: {},
  on(name, fn) {
    (this.listeners[name] ||= []).push(fn);
  },
  emit(name, detail) {
    for (const fn of this.listeners[name] || []) fn(detail || {});
  },
  // A Rocket League point or direction (uu) as a scene vector (m).
  toScene(out, x, y, z) {
    return out.set(x * 0.01, z * 0.01, y * 0.01);
  },
  TEAMS: [
    { name: "Blue", color: 0x2f7dff, css: "#5b97ff", dark: 0x0b2f7a },
    { name: "Orange", color: 0xff7a1a, css: "#ff8a2b", dark: 0x7a3206 },
  ],
  // The field, in uu (RocketSim's soccar constants).
  FIELD: {
    halfX: 4096,
    halfY: 5120,
    ceiling: 2044,
    goalHalfWidth: 892.755,
    goalHeight: 642.775,
    goalDepth: 880,
    ballRadius: 91.25,
    bigPadRadius: 208,
    smallPadRadius: 144,
    bigPadWait: 10,
    smallPadWait: 4,
    supersonic: 2200,
    maxCarSpeed: 2300,
  },
  // The six car bodies: hitbox size and its offset from the car's origin, wheel radius and the wheel's
  // suspension anchor (x forward, y sideways, z up; mirrored for the other side) and rest length, in uu.
  BODIES: [
    { name: "Octane", size: [120.507, 86.6994, 38.6591], offset: [13.8757, 0, 20.755], front: { r: 12.5, at: [51.25, 25.9, 20.755], rest: 38.755 }, back: { r: 15, at: [-33.75, 29.5, 20.755], rest: 37.055 } },
    { name: "Dominus", size: [130.427, 85.7799, 33.8], offset: [9, 0, 15.75], front: { r: 12, at: [50.3, 31.1, 15.75], rest: 33.95 }, back: { r: 13.5, at: [-34.75, 33, 15.75], rest: 33.85 } },
    { name: "Plank", size: [131.32, 87.1704, 31.8944], offset: [9.00857, 0, 12.0942], front: { r: 12.5, at: [49.97, 27.8, 12.0942], rest: 31.9242 }, back: { r: 17, at: [-35.43, 20.28, 12.0942], rest: 27.9242 } },
    { name: "Breakout", size: [133.992, 83.021, 32.8], offset: [12.5, 0, 11.75], front: { r: 13.5, at: [51.5, 26.67, 11.75], rest: 29.7 }, back: { r: 15, at: [-35.75, 35, 11.75], rest: 29.666 } },
    { name: "Hybrid", size: [129.519, 84.6879, 36.6591], offset: [13.8757, 0, 20.755], front: { r: 12.5, at: [51.25, 25.9, 20.755], rest: 38.755 }, back: { r: 15, at: [-34, 29.5, 20.755], rest: 37.055 } },
    { name: "Merc", size: [123.22, 79.2103, 44.1591], offset: [11.3757, 0, 21.505], front: { r: 15, at: [51.25, 25.9, 21.505], rest: 39.505 }, back: { r: 15, at: [-33.75, 29.5, 21.505], rest: 39.105 } },
  ],
  // The height of a resting car's origin above the floor (uu), per body: the wheels sit this low.
  REST_HEIGHT: [17.0, 17.05, 18.65, 18.33, 17.0, 18.15],
  // Quality: what the current tier switches on (NV.post sets it; parts read it and listen to "quality").
  q: { tier: "high", pixelRatio: 1, shadows: 2, bloom: true, reflections: true, msaa: 4, particles: 1, crowd: true },
  world: null,
  rec: null,
  prefs: {},
  // What ends with the viewer when its host page closes it (NV.stop): every event listener takes
  // `NV.ending.signal`, and anything else to undo then is given to `NV.atStop`. Listeners on the viewer's own
  // elements count too: the browser may hold on to an element after it left the page (the last one under the
  // pointer or clicked), and a listener left on it would keep the whole closed viewer in memory.
  ending: new AbortController(),
  atStop(fn) {
    this.ending.signal.addEventListener("abort", fn, { once: true });
  },
  // Remembered choices. The host page may replace load/save (the dashboard keeps them in its own file).
  store: {
    load() {
      try {
        return JSON.parse(localStorage.getItem("nitro-viewer") || "{}");
      } catch {
        return {};
      }
    },
    save(prefs) {
      try {
        localStorage.setItem("nitro-viewer", JSON.stringify(prefs));
      } catch {
        // Nowhere to keep them (a private window): the choices last until the page closes.
      }
    },
  },
  setPref(key, value) {
    this.prefs[key] = value;
    this.store.save(this.prefs);
  },
  clamp: (v, lo, hi) => Math.min(hi, Math.max(lo, v)),
  lerp: (a, b, k) => a + (b - a) * k,
  smooth: (k) => k * k * (3 - 2 * k),
  // Moves `current` toward `target` at a rate that does not depend on the frame time.
  damp: (current, target, rate, dt) => current + (target - current) * (1 - Math.exp(-rate * dt)),
};
