// Cameras: five ways to look at the game. Every mode works out the pose it wants each frame; a change of
// mode or of the followed car blends from the pose on screen to that one, and a jump in time cuts.
//   broadcast  from the side line, high up, panning with the play and widening when the cars spread out
//   ball       behind the followed car, turned so the ball stays in view
//   car        behind the followed car, looking where it points
//   top        straight down on the whole field (Blue's goal on the left, or at the bottom of a tall picture)
//   free       an orbit the viewer steers with the mouse and the keyboard
// The two chase cameras (ball and car) are set as a player sets them in the game; see SETTINGS.

NV.cams = (() => {
  const MODES = ["broadcast", "ball", "car", "top", "free"];
  const F = NV.FIELD, S = NV.S;
  // The chase camera settings, with the names, ranges and steps players of the game know:
  //   fov         horizontal field of view (degrees)
  //   distance    how far behind the car's origin the camera sits (uu)
  //   height      how far above the car's origin it sits (uu)
  //   angle       the camera's pitch (degrees, negative looks down)
  //   stiffness   how rigidly it follows the car: 0 lags and swings, 1 is locked to it
  //   swivel      how fast the free camera turns with the mouse and the arrow keys
  //   transition  how fast a change of camera blends
  //   ballLock    Ball cam faces the ball even when it is behind the car or right on it
  const SETTINGS = {
    fov: { min: 60, max: 110, step: 1, value: 110 },
    distance: { min: 100, max: 400, step: 10, value: 270 },
    height: { min: 40, max: 200, step: 10, value: 100 },
    angle: { min: -15, max: 0, step: 1, value: -3 },
    stiffness: { min: 0, max: 1, step: 0.05, value: 0.45 },
    swivel: { min: 1, max: 10, step: 0.1, value: 5 },
    transition: { min: 1, max: 2, step: 0.1, value: 1.2 },
    ballLock: { value: true },
  };
  const settings = {};
  // Seconds a blend between cameras takes at transition speed 1; faster speeds divide it.
  const EASE_SECONDS = 0.72;
  // Seconds the chase camera's place and turn trail the car at stiffness 0; none at stiffness 1.
  const PLACE_LAG = 0.06, TURN_LAG = 0.35;
  // Turning: never faster than this (radians a second), from stiffness 0 to just under 1.
  const TURN_LIMIT = [4.5, 12];
  // Ball cam stops facing the ball (with ballLock off) once the ball is this far round from the car's nose
  // (cosine), and faces it again under the second; and likewise when the ball is this close (metres).
  const BEHIND = [-0.34, -0.1], CLOSE = [2.5, 3];
  // Where the broadcast camera stands (metres): outside the wall on the -x side, which puts Blue's goal on
  // the left of the picture as on the scoreboard, and above the height of the ceiling.
  const SIDE_LINE = -66, SIDE_HEIGHT = 27;
  const FLOOR = 0.35;
  const WALL_X = F.halfX * S - 0.6, WALL_Z = F.halfY * S - 0.6, ROOF = F.ceiling * S - 0.5;
  const GOAL_Z = (F.halfY + F.goalDepth) * S - 0.6;
  // The corners are cut at 45 degrees: |x| + |z| stays under this (metres).
  const CORNER = 80.64 - 0.9;
  const UP = new THREE.Vector3(0, 1, 0);
  const V = () => new THREE.Vector3();

  const want = { pos: V(), quat: new THREE.Quaternion(), fov: 50, near: 1 };
  const from = { pos: V(), quat: new THREE.Quaternion(), fov: 50 };
  const lookMatrix = new THREE.Matrix4();
  const a = V(), b = V(), c = V(), d = V();
  const view = { w: 1, h: 1 };
  let ease = 1; // progress of the blend from `from` to the wanted pose; 1 when there is none
  let cut = true; // the next frame jumps: nothing is eased
  let fresh = true; // the mode's own smoothing starts anew
  let lastT = 0, wobble = 0;

  const side = { lead: V(), focus: V(), tan: 0.4, tilt: 0 };
  const RIGHT = new THREE.Vector3(1, 0, 0), turn = new THREE.Quaternion();
  const chase = { anchor: V(), dir: V().set(0, 0, 1), yaw: 0, onCar: false };
  const head = { yaw: 0, level: true }, h1 = V(), h2 = V(), h3 = V();
  const free = {
    target: V().set(0, 2, 0), yaw: -Math.PI / 2, pitch: 0.5, dist: 60,
    goal: { target: V().set(0, 2, 0), yaw: -Math.PI / 2, pitch: 0.5, dist: 60 },
    lock: null, // { ball: true } or { car: index }: the orbit's centre rides on it
    fov: 50,
    keys: new Set(),
    fly: V(),
  };

  const api = {
    MODES,
    mode: "broadcast",
    car: 0,
    // Pixels at each edge that the controls cover: the broadcast and top cameras keep the play clear of them.
    safe: { top: 0, right: 0, bottom: 0, left: 0 },
    build,
    update,
    set,
    follow: (car) => set(api.mode, car),
    next,
    cut: () => {
      cut = true;
    },
    key,
    pick,
    SETTINGS,
    settings,
    setSetting,
    resetSettings,
  };

  // Changes one chase camera setting (held to its range and step) and remembers it.
  function setSetting(name, value) {
    const spec = SETTINGS[name];
    if (!spec) return;
    if (name === "ballLock") {
      // What Ball cam faces can change at once, paused too: blended, not cut.
      if (settings.ballLock !== !!value && api.mode === "ball") restart();
      settings.ballLock = !!value;
    } else {
      const number = Number(value);
      if (!Number.isFinite(number)) return;
      const steps = Math.round((NV.clamp(number, spec.min, spec.max) - spec.min) / spec.step);
      settings[name] = Number(NV.clamp(spec.min + steps * spec.step, spec.min, spec.max).toFixed(4));
    }
    keepSettings();
  }

  function resetSettings() {
    if (!settings.ballLock && api.mode === "ball") restart();
    for (const [name, spec] of Object.entries(SETTINGS)) settings[name] = spec.value;
    keepSettings();
  }

  function keepSettings() {
    NV.setPref("chase", { ...settings });
    NV.emit("chase", { ...settings });
  }

  // The remembered settings; anything missing or not a setting's kind takes its starting value.
  function loadSettings(kept) {
    for (const [name, spec] of Object.entries(SETTINGS)) settings[name] = spec.value;
    if (!kept || typeof kept !== "object") return;
    for (const [name, spec] of Object.entries(SETTINGS)) {
      const value = kept[name];
      if (name === "ballLock") {
        if (typeof value === "boolean") settings.ballLock = value;
      } else if (Number.isFinite(value)) settings[name] = NV.clamp(value, spec.min, spec.max);
    }
  }

  function pose(eye, target, up) {
    want.pos.copy(eye);
    lookMatrix.lookAt(eye, target, up || UP);
    want.quat.setFromRotationMatrix(lookMatrix);
  }

  // Keeps a point inside the arena's walls, floor and ceiling (the goals count as inside).
  function keepInside(p) {
    p.y = NV.clamp(p.y, FLOOR, ROOF);
    p.x = NV.clamp(p.x, -WALL_X, WALL_X);
    const inGoal = Math.abs(p.x) < F.goalHalfWidth * S - 0.6 && p.y < F.goalHeight * S - 0.5;
    p.z = NV.clamp(p.z, inGoal ? -GOAL_Z : -WALL_Z, inGoal ? GOAL_Z : WALL_Z);
    const over = Math.abs(p.x) + Math.abs(p.z) - CORNER;
    if (over > 0 && !inGoal) {
      p.x -= Math.sign(p.x) * over * 0.5;
      p.z -= Math.sign(p.z) * over * 0.5;
    }
    return p;
  }

  function broadcast(world, dt) {
    const ball = world.ball.pos;
    // The centre of the play: mostly the ball, pulled a little toward the cars.
    let count = 0;
    b.set(0, 0, 0);
    for (const car of world.cars) {
      if (car.demoed) continue;
      b.add(car.pos);
      count++;
    }
    // A little ahead of the ball, as a camera operator leads a pass.
    a.copy(ball).addScaledVector(world.ball.vel, 0.18);
    a.x = NV.clamp(a.x, -WALL_X, WALL_X);
    a.z = NV.clamp(a.z, -GOAL_Z, GOAL_Z);
    if (count) a.lerp(b.multiplyScalar(1 / count), 0.3);
    a.y = NV.clamp(ball.y * 0.5 + 1, 1, 9);
    if (fresh) {
      side.lead.copy(a);
      side.focus.copy(a);
    } else {
      // Two eased steps in a row: the camera starts and stops gently instead of snapping after the ball.
      const k = 1 - Math.exp(-5 * dt);
      side.lead.lerp(a, k);
      side.focus.lerp(side.lead, k);
    }
    c.set(SIDE_LINE, SIDE_HEIGHT, NV.clamp(side.focus.z * 0.62, -40, 40));
    pose(c, side.focus);
    // The controls cover the top and the bottom of the picture: the play is aimed at the middle of what
    // stays clear, and has to fit between them. The cards at the sides cover only part of the height, so
    // most of their width is kept clear, not all of it.
    const clearUp = Math.max(0.3, 1 - (2 * api.safe.top) / view.h);
    const clearDown = Math.max(0.3, 1 - (2 * api.safe.bottom) / view.h);
    const clearSide = Math.max(0.5, 1 - (1.5 * Math.max(api.safe.left, api.safe.right)) / view.w);
    const tilt = Math.atan(((clearUp - clearDown) / 2) * side.tan);
    side.tilt = fresh ? tilt : NV.damp(side.tilt, tilt, 3, dt);
    want.quat.multiply(turn.setFromAxisAngle(RIGHT, -side.tilt));
    // How wide the picture must be to hold the ball and (nearly) every car.
    const right = a.set(1, 0, 0).applyQuaternion(want.quat), up = b.set(0, 1, 0).applyQuaternion(want.quat);
    const forward = d.set(0, 0, -1).applyQuaternion(want.quat);
    const aspect = NV.camera.aspect;
    let need = 0;
    const hold = (p, weight) => {
      const dx = p.x - c.x, dy = p.y - c.y, dz = p.z - c.z;
      const depth = Math.max(10, dx * forward.x + dy * forward.y + dz * forward.z);
      const across = Math.abs(dx * right.x + dy * right.y + dz * right.z) / depth / clearSide;
      const along = (dx * up.x + dy * up.y + dz * up.z) / depth;
      const upright = along > 0 ? along / clearUp : -along / clearDown;
      need = Math.max(need, (across * weight + 0.07) / aspect, upright * weight + 0.07);
    };
    hold(ball, 1);
    // Cars near the ball are kept in the picture; one far from the play does not pull the camera back.
    for (const car of world.cars) {
      if (car.demoed) continue;
      const far = Math.hypot(car.pos.x - ball.x, car.pos.z - ball.z);
      const near = 1 - NV.smooth(NV.clamp((far - 18) / 22, 0, 1));
      if (near > 0) hold(car.pos, 0.94 * near);
    }
    const tan = NV.clamp(need, 0.17, Math.max(0.5, 0.3 / aspect));
    side.tan = fresh ? tan : NV.damp(side.tan, tan, 1.5, dt);
    want.fov = THREE.MathUtils.radToDeg(2 * Math.atan(side.tan));
    want.near = 1;
  }

  // The fraction of the way to close this frame for something that trails by `seconds` (none: all of it).
  const catchUp = (seconds, dt) => (seconds > 0 ? 1 - Math.exp(-dt / seconds) : 1);

  // Where the followed car is, trailed as loosely as the stiffness asks; a demolished car stays where it
  // was last seen, and a car put somewhere else (a respawn) is not chased across the field.
  function anchor(world, dt) {
    const car = world.cars[api.car] || world.cars[0];
    if (fresh || (!car.demoed && chase.anchor.distanceToSquared(car.pos) > 225)) chase.anchor.copy(car.pos);
    else if (!car.demoed) chase.anchor.lerp(car.pos, catchUp(PLACE_LAG * (1 - settings.stiffness), dt));
    return car;
  }

  // The way the car heads on the field (head.yaw), or null when it cannot be told. A car that is the right
  // way up gives it with its nose. Upside down or standing on end (a flip, an aerial, a wall) the nose would
  // swing the camera around: the heading is then the way the car travels, or none when it hardly moves.
  function heading(car) {
    h1.set(1, 0, 0).applyQuaternion(car.quat).setY(0);
    h2.copy(car.vel).setY(0);
    const level = h1.length() > 0.35 && h3.set(0, 1, 0).applyQuaternion(car.quat).y > 0.15;
    const moving = h2.length() > 5;
    if (!level && moving) h1.copy(h2);
    if (!level && !moving && !(fresh && h1.length() > 0.05)) return null;
    head.yaw = Math.atan2(h1.x, h1.z);
    head.level = level;
    return head;
  }

  // Turns chase.yaw after the car's heading, the short way around and as stiffly as the settings ask. A
  // heading read from the car's travel (not level) is followed slowly: in the air it jumps about.
  function steer(car, dt) {
    if (fresh) chase.yaw = car.team ? Math.PI : 0;
    if (car.demoed) return;
    const found = heading(car);
    if (!found) return;
    if (fresh) {
      chase.yaw = found.yaw;
      return;
    }
    const turn = Math.atan2(Math.sin(found.yaw - chase.yaw), Math.cos(found.yaw - chase.yaw));
    const stiff = settings.stiffness;
    const lag = found.level ? TURN_LAG * (1 - stiff) : 0.65;
    let step = turn * catchUp(lag, dt);
    if (lag > 0) {
      const limit = NV.lerp(TURN_LIMIT[0], TURN_LIMIT[1], stiff) * dt;
      step = NV.clamp(step, -limit, limit);
    }
    chase.yaw += step;
  }

  // The chase pose: Distance behind and Height above the car's origin along `dir` (a unit vector), kept
  // inside the arena, looking along `dir` tilted by Angle.
  function chasePose(dir) {
    b.copy(chase.anchor).addScaledVector(dir, -settings.distance * S);
    b.y += settings.height * S;
    keepInside(b);
    const pitch = Math.asin(NV.clamp(dir.y, -1, 1)) + THREE.MathUtils.degToRad(settings.angle);
    const yaw = Math.atan2(dir.x, dir.z);
    c.set(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)).add(b);
    pose(b, c);
    // The setting is the width of the view; three.js takes its height. A tall picture keeps a sane height.
    const across = Math.tan(THREE.MathUtils.degToRad(settings.fov / 2));
    want.fov = Math.min(120, THREE.MathUtils.radToDeg(2 * Math.atan(across / NV.camera.aspect)));
    want.near = 0.3;
  }

  function ballCam(world, dt) {
    const car = anchor(world, dt);
    steer(car, dt);
    a.copy(world.ball.pos).sub(chase.anchor);
    a.y -= 0.6;
    const flat = Math.hypot(a.x, a.z);
    if (!settings.ballLock) {
      // Off: while the ball is behind the car or right on it, the camera looks where the car drives.
      const nose = flat > 1e-3 ? (a.x * Math.sin(chase.yaw) + a.z * Math.cos(chase.yaw)) / flat : 1;
      chase.onCar = chase.onCar ? nose < BEHIND[1] || flat < CLOSE[1] : nose < BEHIND[0] || flat < CLOSE[0];
    } else chase.onCar = false;
    if (chase.onCar) a.set(Math.sin(chase.yaw), 0, Math.cos(chase.yaw));
    else if (a.length() > 0.8) {
      // The ball overhead or under the car would flip the camera: the tilt has limits.
      a.y = NV.clamp(a.y, -flat * 0.45 - 0.2, flat * 1.4 + 0.5);
    } else a.copy(chase.dir);
    a.normalize();
    // Turned toward the ball a little more softly than after the car's heading, so a bounce does not jolt.
    if (fresh) chase.dir.copy(a);
    else chase.dir.lerp(a, catchUp(0.06 + TURN_LAG * (1 - settings.stiffness), dt)).normalize();
    chasePose(chase.dir);
  }

  function carCam(world, dt) {
    const car = anchor(world, dt);
    steer(car, dt);
    chase.dir.set(Math.sin(chase.yaw), 0, Math.cos(chase.yaw));
    chasePose(chase.dir);
  }

  function top() {
    const safe = api.safe;
    const w = Math.max(50, view.w - safe.left - safe.right), h = Math.max(50, view.h - safe.top - safe.bottom);
    const tall = w < h * 0.8;
    const length = 2 * (F.halfY + F.goalDepth) * S + 5, width = 2 * F.halfX * S + 5;
    const fov = 28, tan = Math.tan(THREE.MathUtils.degToRad(fov / 2));
    const metresPerPixel = tall ? Math.max(length / h, width / w) : Math.max(length / w, width / h);
    const height = (metresPerPixel * view.h) / (2 * tan);
    // The field's centre is drawn in the middle of the part of the picture the controls leave free.
    const shiftRight = -(safe.left + w / 2 - view.w / 2) * metresPerPixel;
    const shiftUp = (safe.top + h / 2 - view.h / 2) * metresPerPixel;
    if (tall) a.set(-shiftRight, height, shiftUp);
    else a.set(shiftUp, height, shiftRight);
    b.set(a.x, 0, a.z);
    pose(a, b, tall ? c.set(0, 0, 1) : c.set(1, 0, 0));
    want.fov = fov;
    want.near = 1;
  }

  // The orbit that shows the picture now on screen, so that switching to the free camera moves nothing.
  function adopt() {
    const camera = NV.camera;
    camera.getWorldDirection(a);
    const reach = a.y < -0.05 ? (camera.position.y - 1) / -a.y : 40;
    const dist = NV.clamp(reach, 4, 240);
    const goal = free.goal;
    free.fov = NV.clamp(camera.fov, 35, 70);
    goal.target.copy(camera.position).addScaledVector(a, dist);
    goal.dist = dist;
    goal.pitch = NV.clamp(Math.asin(NV.clamp(-a.y, -1, 1)), -0.2, 1.553);
    if (Math.abs(a.y) > 0.985) {
      b.set(0, 1, 0).applyQuaternion(camera.quaternion);
      goal.yaw = Math.atan2(-b.x, -b.z);
    } else goal.yaw = Math.atan2(-a.x, -a.z);
    free.target.copy(goal.target);
    free.yaw = goal.yaw;
    free.pitch = goal.pitch;
    free.dist = goal.dist;
    free.lock = null;
    free.fly.set(0, 0, 0);
  }

  function freeCam(world, dt) {
    const goal = free.goal, keys = free.keys;
    // The arrow keys turn the view, as fast as the swivel speed says.
    const swivel = settings.swivel * 0.45 * dt;
    goal.yaw += swivel * ((keys.has("ArrowLeft") ? 1 : 0) - (keys.has("ArrowRight") ? 1 : 0));
    const tilt = (keys.has("ArrowDown") ? 1 : 0) - (keys.has("ArrowUp") ? 1 : 0);
    if (tilt) goal.pitch = NV.clamp(goal.pitch + swivel * tilt, -0.2, 1.553);
    // Flying: W and S along the view, A and D sideways, Q and E down and up; faster from further away.
    const forward = a.set(-Math.sin(free.yaw), 0, -Math.cos(free.yaw));
    const right = b.set(Math.cos(free.yaw), 0, -Math.sin(free.yaw));
    c.set(0, 0, 0);
    c.addScaledVector(forward, (keys.has("KeyW") ? 1 : 0) - (keys.has("KeyS") ? 1 : 0));
    c.addScaledVector(right, (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0));
    c.y += (keys.has("KeyE") ? 1 : 0) - (keys.has("KeyQ") ? 1 : 0);
    if (c.lengthSq() > 0) {
      c.normalize().multiplyScalar(Math.max(12, free.dist * 0.9));
      free.lock = null;
    }
    free.fly.lerp(c, 1 - Math.exp(-8 * dt));
    goal.target.addScaledVector(free.fly, dt);
    if (free.lock) {
      const car = free.lock.car !== undefined ? world.cars[free.lock.car] : null;
      if (free.lock.ball) goal.target.copy(world.ball.pos);
      else if (car && !car.demoed) goal.target.copy(car.pos);
    }
    goal.target.x = NV.clamp(goal.target.x, -70, 70);
    goal.target.y = NV.clamp(goal.target.y, 0, 40);
    goal.target.z = NV.clamp(goal.target.z, -85, 85);
    const k = fresh ? 1 : 1 - Math.exp(-14 * dt);
    free.target.lerp(goal.target, k);
    free.yaw += (goal.yaw - free.yaw) * k;
    free.pitch += (goal.pitch - free.pitch) * k;
    free.dist += (goal.dist - free.dist) * k;
    const flat = Math.cos(free.pitch);
    c.set(flat * Math.sin(free.yaw), Math.sin(free.pitch), flat * Math.cos(free.yaw));
    d.copy(free.target).addScaledVector(c, free.dist);
    d.y = Math.max(FLOOR, d.y);
    pose(d, free.target);
    want.fov = free.fov;
    want.near = 0.3;
  }

  function update(world, dt) {
    const camera = NV.camera;
    // Time that jumps without an event (a live game catching up) is a cut as well.
    if (Math.abs(world.t - lastT) > 1) cut = true;
    // The chase cameras trail the car in the game's time: paused, they hold still with it.
    const played = NV.clamp(world.t - lastT, 0, 0.25);
    lastT = world.t;
    if (cut) {
      fresh = true;
      ease = 1;
    }
    if (api.mode === "broadcast") broadcast(world, dt);
    else if (api.mode === "ball") ballCam(world, played);
    else if (api.mode === "car") carCam(world, played);
    else if (api.mode === "top") top();
    else freeCam(world, dt);
    cut = false;
    fresh = false;

    let fov = want.fov;
    if (ease < 1) {
      ease = Math.min(1, ease + (dt * settings.transition) / EASE_SECONDS);
      const k = NV.smooth(ease);
      camera.position.lerpVectors(from.pos, want.pos, k);
      camera.quaternion.slerpQuaternions(from.quat, want.quat, k);
      fov = NV.lerp(from.fov, want.fov, k);
    } else {
      camera.position.copy(want.pos);
      camera.quaternion.copy(want.quat);
    }
    const shake = (NV.fx && NV.fx.shake) || 0;
    if (shake > 0) {
      wobble += dt;
      const size = 0.12 * Math.min(1, shake);
      camera.position.x += Math.sin(wobble * 53) * size;
      camera.position.y += Math.sin(wobble * 67 + 1) * size;
      camera.position.z += Math.sin(wobble * 59 + 2) * size;
      camera.rotateZ(Math.sin(wobble * 43) * 0.008 * Math.min(1, shake));
    }
    if (camera.position.y < FLOOR) camera.position.y = FLOOR;
    if (Math.abs(camera.fov - fov) > 0.01 || camera.near !== want.near) {
      camera.fov = fov;
      camera.near = want.near;
      camera.updateProjectionMatrix();
    }
    camera.updateMatrixWorld();
  }

  // The camera starts over from the picture on screen, blending into its new pose (unless a jump in time
  // already asked for a cut).
  function restart() {
    if (!cut && NV.camera) {
      from.pos.copy(NV.camera.position);
      from.quat.copy(NV.camera.quaternion);
      from.fov = NV.camera.fov;
      ease = 0;
    }
    fresh = true;
  }

  function set(mode, car) {
    const cars = NV.rec ? NV.rec.cars : 0;
    const nextMode = MODES.includes(mode) ? mode : api.mode;
    let nextCar = car === undefined || car === null ? api.car : car;
    if (!Number.isInteger(nextCar) || nextCar < 0 || nextCar >= cars) nextCar = -1;
    const rides = nextMode === "ball" || nextMode === "car";
    if (rides && !cars) return;
    if (rides && nextCar < 0) nextCar = 0;
    const modeChanged = nextMode !== api.mode, carChanged = nextCar !== api.car;
    if (!modeChanged && !carChanged) return;
    if (modeChanged || rides) {
      restart();
      if (modeChanged && nextMode === "free" && NV.camera) adopt();
    }
    api.mode = nextMode;
    api.car = nextCar;
    if (modeChanged) NV.setPref("camera", nextMode);
    if (carChanged) NV.setPref("follow", nextCar);
    NV.emit("camera", { mode: api.mode, car: api.car });
    if (carChanged) NV.emit("follow", { car: api.car });
  }

  // Follows the next car (step 1) or the one before (step -1), around the list: Blue's cars, then Orange's.
  function next(step) {
    const teams = NV.rec.teams;
    const order = teams.map((_, car) => car).sort((p, q) => teams[p] - teams[q] || p - q);
    if (!order.length) return;
    const at = order.indexOf(api.car);
    set(api.mode, at < 0 ? order[0] : order[(at + step + order.length) % order.length]);
  }

  // The keys of the free camera: W A S D Q E fly, the arrows turn. Returns true when the key was taken.
  function key(code, down) {
    if (!/^(Key[WASDQE]|Arrow(Up|Down|Left|Right))$/.test(code)) return false;
    if (!down) free.keys.delete(code);
    else if (api.mode === "free") free.keys.add(code);
    return api.mode === "free";
  }

  // What is drawn near a point of the picture (pixels from its top left): { car } or { ball: true }, or null.
  function pick(x, y) {
    const camera = NV.camera, world = NV.world;
    const pixelsPerMetre = view.h / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    let best = null, bestScore = 1;
    const test = (p, found) => {
      a.copy(p).applyMatrix4(camera.matrixWorldInverse);
      if (a.z > -camera.near) return;
      const reach = Math.max(26, (2.2 * pixelsPerMetre) / -a.z);
      a.applyMatrix4(camera.projectionMatrix);
      const score = Math.hypot((a.x * 0.5 + 0.5) * view.w - x, (0.5 - a.y * 0.5) * view.h - y) / reach;
      if (score < bestScore) {
        bestScore = score;
        best = found;
      }
    };
    test(world.ball.pos, { ball: true });
    for (const car of world.cars) if (!car.demoed) test(car.pos, { car: car.index });
    return best;
  }

  function build() {
    const prefs = NV.prefs || {};
    loadSettings(prefs.chase);
    api.mode = MODES.includes(prefs.camera) ? prefs.camera : "broadcast";
    api.car = Number.isInteger(prefs.follow) && prefs.follow >= 0 && prefs.follow < NV.rec.cars ? prefs.follow : 0;
    if (NV.rec.cars === 0) {
      api.car = -1;
      if (api.mode === "ball" || api.mode === "car") api.mode = "broadcast";
    }
    NV.camera.far = 3000;
    NV.on("seek", api.cut);
    NV.on("restart", api.cut);
    NV.on("resize", (size) => {
      view.w = size.w;
      view.h = size.h;
    });
    const canvas = NV.renderer.domElement;
    view.w = canvas.clientWidth || 1;
    view.h = canvas.clientHeight || 1;
    pointer(canvas);
  }

  // The mouse and touch on the picture: drag to orbit, right-drag or Shift-drag (or two fingers) to move
  // sideways, wheel or pinch to zoom, double-click to centre on a car or the ball.
  function pointer(canvas) {
    const down = new Map();
    let moved = false, pan = false, pinch = 0;
    const local = (event) => {
      const box = canvas.getBoundingClientRect();
      return [event.clientX - box.left, event.clientY - box.top];
    };
    const slide = (dx, dy) => {
      const goal = free.goal;
      const metres = (2 * Math.tan(THREE.MathUtils.degToRad(NV.camera.fov / 2)) * goal.dist) / view.h;
      a.set(1, 0, 0).applyQuaternion(NV.camera.quaternion);
      b.set(0, 1, 0).applyQuaternion(NV.camera.quaternion);
      goal.target.addScaledVector(a, -dx * metres).addScaledVector(b, dy * metres);
      free.lock = null;
    };
    canvas.style.touchAction = "none";
    canvas.addEventListener("pointerdown", (event) => {
      down.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (down.size === 1) {
        moved = false;
        pan = event.button !== 0 || event.shiftKey;
      }
      pinch = 0;
      canvas.setPointerCapture(event.pointerId);
    }, { signal: NV.ending.signal });
    canvas.addEventListener("pointermove", (event) => {
      const at = down.get(event.pointerId);
      if (!at) return;
      const dx = event.clientX - at.x, dy = event.clientY - at.y;
      if (!moved && Math.hypot(dx, dy) < 5) return;
      // Grabbing the picture in any other mode carries on from that view in the free camera.
      if (!moved && api.mode !== "free") set("free");
      moved = true;
      at.x = event.clientX;
      at.y = event.clientY;
      const goal = free.goal;
      if (down.size >= 2) {
        const [p, q] = [...down.values()];
        const gap = Math.hypot(p.x - q.x, p.y - q.y);
        if (pinch > 0 && gap > 0) goal.dist = NV.clamp((goal.dist * pinch) / gap, 2, 240);
        pinch = gap;
        slide(dx / 2, dy / 2);
      } else if (pan) slide(dx, dy);
      else {
        const turn = 0.0011 * settings.swivel;
        goal.yaw -= dx * turn;
        goal.pitch = NV.clamp(goal.pitch + dy * turn, -0.2, 1.553);
      }
    }, { signal: NV.ending.signal });
    const release = (event) => {
      down.delete(event.pointerId);
      pinch = 0;
    };
    canvas.addEventListener("pointerup", release, { signal: NV.ending.signal });
    canvas.addEventListener("pointercancel", release, { signal: NV.ending.signal });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault(), { signal: NV.ending.signal });
    canvas.addEventListener("wheel", (event) => {
      if (api.mode !== "free") return;
      event.preventDefault();
      const step = event.deltaMode === 1 ? event.deltaY * 30 : event.deltaY;
      free.goal.dist = NV.clamp(free.goal.dist * Math.exp(step * 0.0012), 2, 240);
    }, { passive: false, signal: NV.ending.signal });
    canvas.addEventListener("dblclick", (event) => {
      const found = pick(...local(event));
      if (!found) return;
      if (found.car !== undefined) set(api.mode, found.car);
      if (api.mode === "free") {
        free.lock = found;
        free.goal.dist = Math.min(free.goal.dist, 14);
      }
    }, { signal: NV.ending.signal });
    window.addEventListener("blur", () => free.keys.clear(), { signal: NV.ending.signal });
  }

  return api;
})();
