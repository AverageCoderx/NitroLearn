// Start-up and the frame loop: the renderer, the player (time, speed, seeking) and the order the parts run in.

NV.player = {
  t: 0,
  playing: true,
  speed: 1,
  atLive: false,
  get duration() {
    return NV.rec.duration;
  },
  seek(t) {
    this.t = NV.clamp(t, 0, this.duration);
    this.atLive = false;
    this.cursor = Math.floor(this.t * NV.rec.fps);
    NV.emit("seek");
  },
  toggle() {
    if (!this.playing && this.t >= this.duration - 1e-3 && !NV.rec.live) this.seek(0);
    this.playing = !this.playing;
    NV.emit("play", { playing: this.playing });
  },
  setSpeed(speed) {
    this.speed = speed;
    NV.setPref("speed", speed);
    NV.emit("speed", { speed });
  },
  // Follows the newest moment of a live game.
  goLive() {
    this.atLive = true;
    this.playing = true;
  },
  cursor: 0,
};

// The renderer on `canvas`. When the browser gives no WebGL 2 context (turned off, blocked, or no graphics it
// can use) this throws a NoGraphicsError carrying the browser's own reason when it gave one.
NV.graphics = function graphics(canvas) {
  let refused = "";
  const heard = (event) => (refused = event.statusMessage || "");
  canvas.addEventListener("webglcontextcreationerror", heard);
  try {
    return new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance", stencil: false });
  } catch (error) {
    const none = new Error(refused.trim() || error.message);
    none.name = "NoGraphicsError";
    throw none;
  } finally {
    canvas.removeEventListener("webglcontextcreationerror", heard);
  }
};

NV.start = async function start(data, options) {
  const host = options.host || document.body;
  NV.host = host;
  NV.options = options;
  NV.prefs = NV.store.load();
  const rec = await NV.data.open(data);
  if (NV.stopped) return;
  NV.rec = rec;
  // A page that follows a run loads itself again when its game ends: its file then holds the newest game.
  if (data.follow && !options.embedded) NV.on("end", () => setTimeout(() => location.reload(), 1500));
  // A live game starts with no frames: the scene is built once the first ones are here.
  if (options.live) await NV.live.start(options.live);
  if (NV.stopped) return;
  NV.world = NV.data.newWorld(rec);
  NV.emit("building");
  const renderer = NV.graphics(host.querySelector("canvas.nv-canvas"));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  NV.renderer = renderer;
  NV.scene = new THREE.Scene();
  NV.camera = new THREE.PerspectiveCamera(55, 1, 0.5, 2500);
  NV.post.init(renderer);
  NV.arena.build(NV.scene, { meshes: options.arena || null, pads: rec.padSpots });
  NV.ball.build(NV.scene);
  NV.cars.build(NV.scene, rec);
  NV.fx.build(NV.scene);
  NV.cams.build();
  NV.ui.build();
  const player = NV.player;
  if (NV.prefs.speed) player.speed = NV.prefs.speed;
  if (rec.live) player.goLive();

  const resize = () => {
    const w = Math.max(1, host.clientWidth), h = Math.max(1, host.clientHeight);
    NV.camera.aspect = w / h;
    NV.camera.updateProjectionMatrix();
    NV.post.resize(w, h);
    NV.emit("resize", { w, h });
  };
  const sized = new ResizeObserver(resize);
  sized.observe(host);
  NV.atStop(() => sized.disconnect());
  resize();

  // The events between two frames, sent while time moves forward by small steps.
  const lists = [["goals", "goal"], ["touches", "touch"], ["demos", "demo"]];
  const point = new THREE.Vector3();
  function sendEvents(from, to) {
    if (to <= from || to - from > rec.fps) return;
    for (const [list, name] of lists) {
      for (const event of rec[list]) {
        if (event.frame > from && event.frame <= to) {
          const L = rec.L;
          if (name === "goal") NV.toScene(point, rec.h(event.frame, L.ball), rec.h(event.frame, L.ball + 1), rec.h(event.frame, L.ball + 2));
          else NV.toScene(point, rec.c(event.frame, event.car, L.pos), rec.c(event.frame, event.car, L.pos + 1), rec.c(event.frame, event.car, L.pos + 2));
          NV.emit(name, { ...event, pos: point.clone() });
        }
      }
    }
    for (const frame of rec.restarts) if (frame > from && frame <= to) NV.emit("restart", { frame });
  }

  // The first frame's time stamp can lie before the scene was built: time only ever moves forward here.
  let before = null;
  NV.frameMs = 16;
  function frame(now) {
    if (NV.stopped) return;
    const real = before === null ? 0 : NV.clamp((now - before) / 1000, 0, 0.1);
    before = now;
    NV.frameMs = NV.lerp(NV.frameMs, real * 1000, 0.05);
    if (player.atLive) {
      // A little behind the newest frame, so there is always a frame to blend toward.
      const target = Math.max(0, player.duration - 0.15);
      player.t = target - player.t > 1 ? target : Math.min(target, player.t + real * (target - player.t > 0.3 ? 1.15 : 1));
    } else if (player.playing) {
      player.t += real * player.speed;
      if (player.t >= player.duration) {
        player.t = player.duration;
        if (!rec.live) {
          player.playing = false;
          NV.emit("play", { playing: false });
          NV.emit("end");
        }
      }
    }
    const world = rec.sample(player.t, NV.world);
    const dt = player.playing || player.atLive ? real * (player.atLive ? 1 : player.speed) : 0;
    sendEvents(player.cursor, world.frame);
    player.cursor = world.frame;
    NV.arena.update(world, dt);
    NV.ball.update(world, dt);
    NV.cars.update(world, dt);
    NV.fx.update(world, dt);
    NV.cams.update(world, real);
    NV.ui.update(world, real);
    NV.post.render(NV.scene, NV.camera, real);
    NV.frames = (NV.frames || 0) + 1;
    if (NV.frames === 1) NV.emit("drawn");
    if (!NV.stopped) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
};

// Ends the viewer for good (the dashboard closes it this way): the loop, what it listens to outside its own
// elements, and the graphics memory. The graphics context is given back at once: a page may hold only a few.
NV.stop = function stop() {
  if (NV.stopped) return;
  NV.stopped = true;
  NV.ending.abort();
  NV.post.dispose();
  const renderer = NV.renderer;
  if (!renderer) return;
  // three.js gives every renderer the same lookup texture for shiny surfaces, and that texture remembers each
  // renderer that drew with it. Disposing it makes them forget (one still drawing takes it up again);
  // without this a closed viewer, scene and all, would stay in memory for as long as the page lives.
  const shared = new Set();
  NV.scene.traverse((object) => {
    for (const material of [].concat(object.material || [])) {
      const uniforms = renderer.properties.get(material).uniforms;
      if (uniforms && uniforms.dfgLUT && uniforms.dfgLUT.value) shared.add(uniforms.dfgLUT.value);
    }
  });
  for (const texture of shared) texture.dispose();
  renderer.dispose();
  renderer.forceContextLoss();
};
