// The recording: frames as 16-bit columns in the layout its `format` names, read into typed arrays once and
// sampled between frames while playing. Its events and each car's wheel roll are worked out once per frame.
// A recording in a format this viewer does not know is not opened (`refusal` says why).

NV.data = (() => {
  const MARK = { playing: 0, restart: 1, blueGoal: 2, orangeGoal: 3 };
  const FLAG = { ground: 1, demoed: 2, supersonic: 4, jumped: 8, doubleJumped: 16, flipped: 32, touched: 64, marksShift: 7 };
  // Column offsets of each format this viewer reads: head (per frame) and car (per car).
  const LAYOUT = {
    2: { ball: 1, ballVel: 4, ballFwd: 7, ballUp: 10, ballSpin: 13, score: 16, padWait: 18,
         team: 0, pos: 1, vel: 4, fwd: 7, up: 10, spin: 13, boost: 16, flags: 17, ctrl: 18, state: 26, handbrake: 27, body: 28,
         respawn: 29 },
  };
  const STATE = { wheel: 1, boosting: 16, jumping: 32, flipping: 64 };
  // RocketSim's steering angle (rad) by forward speed (uu/s), and with the handbrake held.
  const STEER = [[0, 0.53356], [500, 0.3193], [1000, 0.18203], [1500, 0.1057], [1750, 0.08507], [3000, 0.03454]];
  const SLIDE = [[0, 0.39235], [2500, 0.1261]];
  const curve = (points, x) => {
    if (x <= points[0][0]) return points[0][1];
    for (let i = 1; i < points.length; i++) {
      if (x <= points[i][0]) {
        const [x0, y0] = points[i - 1], [x1, y1] = points[i];
        return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
      }
    }
    return points[points.length - 1][1];
  };

  function bytes(base64) {
    const text = atob(base64);
    const out = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
    return out;
  }

  // A basis from a forward and an up direction (game axes) as a scene quaternion.
  const mx = new THREE.Matrix4(), vf = new THREE.Vector3(), vu = new THREE.Vector3(), vr = new THREE.Vector3();
  function basis(out, fx, fy, fz, ux, uy, uz) {
    vf.set(fx, fz, fy).normalize();
    vu.set(ux, uz, uy);
    vr.crossVectors(vu, vf).normalize(); // the car's left in the scene
    vu.crossVectors(vf, vr).normalize();
    // Model space: +x forward, +y up, +z to the car's right.
    mx.makeBasis(vf, vu, vr.negate());
    return out.setFromRotationMatrix(mx);
  }

  // The frames of a recording's data as 16-bit numbers, frame after frame. Packed frames are stored column
  // after column as differences from the frame before, deflated.
  async function unpack(data) {
    const stored = bytes(data.frames || "");
    if (data.packing !== "delta-deflate") return new Int16Array(stored.buffer);
    const stream = new Blob([stored]).stream().pipeThrough(new DecompressionStream("deflate"));
    const columns = new Int16Array(await new Response(stream).arrayBuffer());
    const width = data.head + data.cars * data.car, count = columns.length / width;
    const frames = new Int16Array(columns.length);
    for (let c = 0; c < width; c++) {
      let value = 0;
      for (let f = 0; f < count; f++) {
        value = ((value + columns[c * count + f]) << 16) >> 16;
        frames[f * width + c] = value;
      }
    }
    return frames;
  }

  // Why this viewer cannot play `data`, in a sentence for the user; "" when it can.
  function refusal(data) {
    if (LAYOUT[data.format]) return "";
    if (data.format > Math.max(...Object.keys(LAYOUT))) {
      return `This recording was made by a newer NitroLearn (format ${data.format}). Update NitroLearn to watch it.`;
    }
    return "This recording is in a format this NitroLearn does not read.";
  }

  async function open(data) {
    const why = refusal(data);
    if (why) throw new Error(why);
    return new Recording(data, await unpack(data));
  }

  class Recording {
    constructor(data, frames) {
      this.title = data.title || "";
      this.subtitle = data.subtitle || "";
      this.format = data.format;
      this.L = LAYOUT[this.format];
      this.fps = data.fps;
      this.cars = data.cars;
      this.head = data.head;
      this.carWidth = data.car;
      this.width = this.head + this.cars * this.carWidth;
      this.scales = Float32Array.from(data.scales);
      this.padSpots = data.pads || [];
      this.decisions = data.decisions || [];
      this.decisionsPerSecond = data.decisionsPerSecond || 15;
      this.attention = data.attention || null;
      this.live = !!data.live;
      this.count = 0;
      this.raw = new Int16Array(0);
      this.roll = new Float32Array(0);
      this.goals = [];
      this.touches = [];
      this.demos = [];
      this.restarts = [];
      this.teams = [];
      if (frames && frames.length) this.append(frames, []);
    }

    get duration() {
      return Math.max(0, (this.count - 1) / this.fps);
    }

    h(frame, col) {
      return this.raw[frame * this.width + col] / this.scales[col];
    }

    c(frame, car, col) {
      const at = this.head + car * this.carWidth + col;
      return this.raw[frame * this.width + at] / this.scales[at];
    }

    // New frames (and decisions) at the end: a live game grows this way.
    append(frames, decisions) {
      const add = Math.floor(frames.length / this.width);
      if (!add) return;
      const first = this.count, total = first + add;
      const grow = (old, per, Type) => {
        if (old.length >= total * per) return old;
        const next = new Type(Math.max(total, first * 2) * per);
        next.set(old.subarray(0, first * per));
        return next;
      };
      this.raw = grow(this.raw, this.width, Int16Array);
      this.raw.set(frames.subarray(0, add * this.width), first * this.width);
      this.roll = grow(this.roll, this.cars, Float32Array);
      this.count = total;
      for (const d of decisions) this.decisions.push(d);
      if (!this.teams.length) for (let k = 0; k < this.cars; k++) this.teams.push(this.c(0, k, this.L.team) | 0);
      this.derive(first, total);
    }

    // What is worked out once per frame: events and each car's wheel roll.
    derive(first, total) {
      const L = this.L, dt = 1 / this.fps;
      for (let f = first; f < total; f++) {
        const mark = this.h(f, 0), prev = Math.max(0, f - 1);
        if (mark === MARK.blueGoal || mark === MARK.orangeGoal) this.goals.push({ frame: f, team: mark === MARK.blueGoal ? 0 : 1 });
        if (mark === MARK.restart && f) this.restarts.push(f);
        for (let k = 0; k < this.cars; k++) {
          const flags = this.c(f, k, L.flags) | 0;
          if (flags & FLAG.touched) this.touches.push({ frame: f, car: k });
          if (flags & FLAG.demoed && f && !((this.c(prev, k, L.flags) | 0) & FLAG.demoed)) this.demos.push({ frame: f, car: k });
          // Wheel roll (uu): the forward speed while the wheels are down, spun by the throttle in the air.
          let roll = f && mark !== MARK.restart ? this.roll[prev * this.cars + k] : 0;
          const forward = this.c(f, k, L.vel) * this.c(f, k, L.fwd) + this.c(f, k, L.vel + 1) * this.c(f, k, L.fwd + 1) + this.c(f, k, L.vel + 2) * this.c(f, k, L.fwd + 2);
          roll += (flags & FLAG.ground ? forward : this.c(f, k, L.ctrl) * 900) * dt;
          this.roll[f * this.cars + k] = roll;
        }
      }
    }

    // Fills `world` with the game at `t` seconds, blended between the two frames around it.
    sample(t, world) {
      const L = this.L, last = this.count - 1;
      const at = NV.clamp(t * this.fps, 0, last);
      const f0 = Math.min(last, Math.floor(at)), f1 = Math.min(last, f0 + 1);
      // No blend into the first frame of a new episode: everything jumps there.
      const k = this.h(f1, 0) === MARK.restart ? 0 : at - f0;
      const mix = (frame, car, col) => (car < 0 ? this.h(frame, col) : this.c(frame, car, col));
      const lerp3 = (out, car, col) => {
        const a = mix(f0, car, col), b = mix(f0, car, col + 1), c = mix(f0, car, col + 2);
        return NV.toScene(out, a + (mix(f1, car, col) - a) * k, b + (mix(f1, car, col + 1) - b) * k, c + (mix(f1, car, col + 2) - c) * k);
      };
      world.t = t;
      world.frame = f0;
      world.blend = k;
      world.clock = f0 / this.fps;
      world.score[0] = this.h(f0, L.score);
      world.score[1] = this.h(f0, L.score + 1);
      const ball = world.ball;
      lerp3(ball.pos, -1, L.ball);
      lerp3(ball.vel, -1, L.ballVel);
      ball.speed = ball.vel.length() * 100;
      basis(qa, this.h(f0, L.ballFwd), this.h(f0, L.ballFwd + 1), this.h(f0, L.ballFwd + 2), this.h(f0, L.ballUp), this.h(f0, L.ballUp + 1), this.h(f0, L.ballUp + 2));
      basis(qb, this.h(f1, L.ballFwd), this.h(f1, L.ballFwd + 1), this.h(f1, L.ballFwd + 2), this.h(f1, L.ballUp), this.h(f1, L.ballUp + 1), this.h(f1, L.ballUp + 2));
      ball.quat.copy(qa).slerp(qb, k);
      for (let i = 0; i < this.cars; i++) {
        const car = world.cars[i];
        const flags = this.c(f0, i, L.flags) | 0;
        car.team = this.teams[i];
        car.body = this.c(f0, i, L.body) | 0;
        lerp3(car.pos, i, L.pos);
        lerp3(car.vel, i, L.vel);
        car.speed = car.vel.length() * 100;
        basis(qa, this.c(f0, i, L.fwd), this.c(f0, i, L.fwd + 1), this.c(f0, i, L.fwd + 2), this.c(f0, i, L.up), this.c(f0, i, L.up + 1), this.c(f0, i, L.up + 2));
        basis(qb, this.c(f1, i, L.fwd), this.c(f1, i, L.fwd + 1), this.c(f1, i, L.fwd + 2), this.c(f1, i, L.up), this.c(f1, i, L.up + 1), this.c(f1, i, L.up + 2));
        car.quat.copy(qa).slerp(qb, k);
        car.boost = NV.lerp(this.c(f0, i, L.boost), this.c(f1, i, L.boost), k);
        car.onGround = !!(flags & FLAG.ground);
        car.demoed = !!(flags & FLAG.demoed);
        car.supersonic = !!(flags & FLAG.supersonic);
        car.jumped = !!(flags & FLAG.jumped);
        car.doubleJumped = !!(flags & FLAG.doubleJumped);
        car.flipped = !!(flags & FLAG.flipped);
        car.marks = flags >> FLAG.marksShift;
        for (let j = 0; j < 8; j++) car.controls[j] = this.c(f0, i, L.ctrl + j);
        car.throttle = car.controls[0];
        car.steer = car.controls[1];
        const state = this.c(f0, i, L.state) | 0;
        for (let w = 0; w < 4; w++) car.wheelContact[w] = !!(state & (STATE.wheel << w));
        car.boosting = !!(state & STATE.boosting);
        car.jumping = !!(state & STATE.jumping);
        car.flipping = !!(state & STATE.flipping);
        car.handbrake = this.c(f0, i, L.handbrake);
        car.respawn = this.c(f0, i, L.respawn);
        car.roll = NV.lerp(this.roll[f0 * this.cars + i], this.roll[f1 * this.cars + i], k);
        // Steering turns the front wheels left for a positive angle; the game's steer is + to the right.
        const table = car.handbrake > 0.5 ? SLIDE : STEER;
        car.steerAngle = -car.steer * curve(table, Math.abs(car.speed));
      }
      for (let p = 0; p < world.pads.length; p++) {
        const pad = world.pads[p];
        pad.wait = this.h(f0, L.padWait + p);
        pad.active = pad.wait <= 0;
      }
      return world;
    }

    // The decision in force at `frame` (the newest one looking at a frame at or before it), or null.
    decisionAt(frame) {
      const list = this.decisions;
      let lo = 0, hi = list.length - 1, found = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (list[mid][0] <= frame) {
          found = mid;
          lo = mid + 1;
        } else hi = mid - 1;
      }
      return found < 0 ? null : list[found];
    }
  }
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();

  function newWorld(rec) {
    const v = () => new THREE.Vector3();
    return {
      t: 0, frame: 0, blend: 0, clock: 0, score: [0, 0],
      ball: { pos: v(), vel: v(), quat: new THREE.Quaternion(), speed: 0 },
      cars: Array.from({ length: rec.cars }, (_, index) => ({
        index, team: 0, body: 0, pos: v(), vel: v(), quat: new THREE.Quaternion(), speed: 0, boost: 0,
        onGround: true, demoed: false, supersonic: false, boosting: false, jumped: false, doubleJumped: false,
        flipped: false, jumping: false, flipping: false, wheelContact: [true, true, true, true], roll: 0,
        steerAngle: 0, throttle: 0, steer: 0, handbrake: 0, respawn: 0, controls: new Float32Array(8), marks: 0,
      })),
      pads: rec.padSpots.map((spot) => ({ x: spot[0], y: spot[1], z: spot[2], big: !!spot[3], active: true, wait: 0 })),
    };
  }

  return { MARK, FLAG, Recording, newWorld, bytes, open, refusal };
})();
