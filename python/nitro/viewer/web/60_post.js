// Drawing: the scene goes into an HDR target (multisampled when the level asks for it), bright parts are
// spread by a bloom chain, and one last pass writes the screen (tone mapping, anti-aliasing of edges when
// there is no multisampling, vignette, dithering, sRGB). Also here: the floor's mirror image, the quality
// levels and the automatic level, which watches the frame times.
//
// The scene parts write linear light, glowing things above 1. NV.post.mirror is { texture, textureMatrix }
// while reflections are on (textureMatrix * world position = projective uv), else null. Objects listed in
// NV.post.mirrorHidden are left out of the mirror image, as is NV.arena.floor.

NV.post = (() => {
  const ORDER = ["low", "medium", "high", "ultra"];
  const TIERS = {
    low: { pixelRatio: 1, shadows: 0, bloom: true, reflections: false, msaa: 0, particles: 0.35, crowd: false },
    medium: { pixelRatio: 1.25, shadows: 1, bloom: true, reflections: false, msaa: 0, particles: 0.6, crowd: true },
    high: { pixelRatio: 1.5, shadows: 2, bloom: true, reflections: true, msaa: 4, particles: 1, crowd: true },
    ultra: { pixelRatio: 2, shadows: 3, bloom: true, reflections: true, msaa: 4, particles: 1, crowd: true },
  };
  // What a level sets that only this file reads: bloom levels, the mirror's share of the resolution and
  // how far the edge search of the anti-aliasing walks.
  const DETAIL = {
    low: { bloomLevels: 4, mirror: 0.5, mirrorSamples: 0, edgeSteps: 3 },
    medium: { bloomLevels: 5, mirror: 0.5, mirrorSamples: 0, edgeSteps: 5 },
    high: { bloomLevels: 6, mirror: 0.5, mirrorSamples: 4, edgeSteps: 5 },
    ultra: { bloomLevels: 6, mirror: 0.75, mirrorSamples: 4, edgeSteps: 5 },
  };
  // The resolution scales the automatic level steps through before it drops a level.
  const SCALES = [1, 0.85, 0.7, 0.55];
  const TARGET_FPS = 55;

  let renderer = null, gl = null, canvas = null;
  let width = 1, height = 1, scale = 1, tier = "high", lost = false;
  let hdr = true, samples = 0, rendererName = "";
  let sceneTarget = null, mirrorTarget = null;
  const down = [], up = [];
  let triangle = null, screen = null, flatCamera = null;
  let prefilter = null, shrink = null, grow = null;
  const finals = {};

  const VERTEX = `
    in vec3 position;
    out vec2 vUv;
    void main() {
      vUv = position.xy * 0.5 + 0.5;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }`;

  // Halves the picture with 13 taps: five overlapping 2x2 boxes, the middle one weighted double (the
  // filter of Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare"). The first
  // step keeps only what is brighter than the threshold (with a soft knee) and weights each box by
  // 1 / (1 + brightness), so one very bright pixel cannot make the whole glow flicker.
  const SHRINK = `
    precision highp float;
    uniform sampler2D tSource;
    uniform vec2 uTexel;
    uniform vec3 uCurve;
    in vec2 vUv;
    out vec4 fragColor;
    vec3 tap(float x, float y) {
      return min(texture(tSource, vUv + vec2(x, y) * uTexel).rgb, vec3(64.0));
    }
    float weight(vec3 c) {
      return 1.0 / (1.0 + dot(c, vec3(0.2126, 0.7152, 0.0722)) * uCurve.z);
    }
    void main() {
      vec3 a = tap(-2.0, 2.0), b = tap(0.0, 2.0), c = tap(2.0, 2.0);
      vec3 d = tap(-2.0, 0.0), e = tap(0.0, 0.0), f = tap(2.0, 0.0);
      vec3 g = tap(-2.0, -2.0), h = tap(0.0, -2.0), i = tap(2.0, -2.0);
      vec3 j = tap(-1.0, 1.0), k = tap(1.0, 1.0), l = tap(-1.0, -1.0), m = tap(1.0, -1.0);
      #ifdef FIRST
        vec3 b0 = (j + k + l + m) * 0.25, b1 = (a + b + d + e) * 0.25, b2 = (b + c + e + f) * 0.25;
        vec3 b3 = (d + e + g + h) * 0.25, b4 = (e + f + h + i) * 0.25;
        float w0 = weight(b0) * 0.5, w1 = weight(b1) * 0.125, w2 = weight(b2) * 0.125;
        float w3 = weight(b3) * 0.125, w4 = weight(b4) * 0.125;
        vec3 color = (b0 * w0 + b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4) / (w0 + w1 + w2 + w3 + w4) * uCurve.z;
        float peak = max(color.r, max(color.g, color.b));
        float soft = clamp(peak - uCurve.x + uCurve.y, 0.0, 2.0 * uCurve.y);
        soft = soft * soft / (4.0 * uCurve.y + 1e-4);
        color *= max(soft, peak - uCurve.x) / max(peak, 1e-4);
      #else
        vec3 color = (j + k + l + m) * 0.125 + e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625;
      #endif
      fragColor = vec4(max(color, vec3(0.0)), 1.0);
    }`;

  // Doubles a level with a 3x3 tent and blends it with the sharper level of the same size. Every level
  // ends up with a share of the whole and the shares add up to 1, so the chain keeps the light it got.
  const GROW = `
    precision highp float;
    uniform sampler2D tSmall;
    uniform sampler2D tSharp;
    uniform vec2 uTexel;
    uniform float uSpread;
    in vec2 vUv;
    out vec4 fragColor;
    void main() {
      vec4 o = uTexel.xyxy * vec4(1.0, 1.0, -1.0, 0.0);
      vec3 wide = texture(tSmall, vUv).rgb * 4.0;
      wide += (texture(tSmall, vUv - o.xw).rgb + texture(tSmall, vUv + o.xw).rgb
        + texture(tSmall, vUv - o.wy).rgb + texture(tSmall, vUv + o.wy).rgb) * 2.0;
      wide += texture(tSmall, vUv - o.xy).rgb + texture(tSmall, vUv + o.xy).rgb
        + texture(tSmall, vUv - o.zy).rgb + texture(tSmall, vUv + o.zy).rgb;
      fragColor = vec4(mix(texture(tSharp, vUv).rgb, wide * 0.0625, uSpread), 1.0);
    }`;

  const FINAL = `
    precision highp float;
    uniform sampler2D tScene;
    uniform sampler2D tBloom;
    uniform vec2 uTexel;
    uniform vec3 uLook;
    uniform float uAspect;
    in vec2 vUv;
    out vec4 fragColor;

    // The ACES filmic curve: Stephen Hill's fit of the reference rendering and output transforms.
    vec3 toneMap(vec3 c) {
      const mat3 toCurve = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
      const mat3 toDisplay = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
      c = toCurve * (c / 0.6);
      c = (c * (c + 0.0245786) - 0.000090537) / (c * (0.983729 * c + 0.4329510) + 0.238081);
      return clamp(toDisplay * c, 0.0, 1.0);
    }
    float shade;
    vec3 display(vec3 scene, vec3 glow) {
      return toneMap((scene * uLook.x + glow * uLook.z) * shade);
    }
    #ifdef FXAA
      // Brightness as the eye weighs it, of a pixel that is still linear and unbounded.
      float luma(vec2 uv) {
        float l = dot(texture(tScene, uv).rgb, vec3(0.2126, 0.7152, 0.0722)) * uLook.x;
        return sqrt(l / (1.0 + l));
      }
    #endif
    void main() {
      vec2 fromCentre = (vUv - 0.5) * vec2(uAspect, 1.0);
      shade = 1.0 - uLook.y * smoothstep(0.35, 1.05, length(fromCentre));
      vec3 glow = vec3(0.0);
      #ifdef BLOOM
        glow = texture(tBloom, vUv).rgb;
      #endif
      vec3 here = texture(tScene, vUv).rgb;
      vec3 color;
      #ifdef FXAA
        // Edge anti-aliasing after Timothy Lottes' FXAA 3.11: find the edge through this pixel, walk
        // along it both ways to its ends, and blend with the pixel across it by how near an end is.
        float lM = luma(vUv);
        float lN = luma(vUv + vec2(0.0, uTexel.y)), lS = luma(vUv - vec2(0.0, uTexel.y));
        float lE = luma(vUv + vec2(uTexel.x, 0.0)), lW = luma(vUv - vec2(uTexel.x, 0.0));
        float lLow = min(lM, min(min(lN, lS), min(lE, lW))), lHigh = max(lM, max(max(lN, lS), max(lE, lW)));
        float range = lHigh - lLow;
        if (range < max(0.04, lHigh * 0.125)) {
          color = display(here, glow);
        } else {
          float lNE = luma(vUv + uTexel), lSW = luma(vUv - uTexel);
          float lNW = luma(vUv + vec2(-uTexel.x, uTexel.y)), lSE = luma(vUv + vec2(uTexel.x, -uTexel.y));
          float acrossRows = abs(lN + lS - 2.0 * lM) * 2.0 + abs(lNE + lSE - 2.0 * lE) + abs(lNW + lSW - 2.0 * lW);
          float acrossColumns = abs(lE + lW - 2.0 * lM) * 2.0 + abs(lNE + lNW - 2.0 * lN) + abs(lSE + lSW - 2.0 * lS);
          bool level = acrossRows >= acrossColumns;
          float lA = level ? lS : lW, lB = level ? lN : lE;
          float gA = abs(lA - lM), gB = abs(lB - lM);
          bool towardA = gA >= gB;
          vec2 across = level ? vec2(0.0, uTexel.y) : vec2(uTexel.x, 0.0);
          if (towardA) across = -across;
          vec2 along = level ? vec2(uTexel.x, 0.0) : vec2(0.0, uTexel.y);
          float lEdge = 0.5 * ((towardA ? lA : lB) + lM);
          float limit = 0.25 * max(gA, gB);
          vec2 centre = vUv + across * 0.5;
          vec2 back = centre - along, ahead = centre + along;
          float endBack = luma(back) - lEdge, endAhead = luma(ahead) - lEdge;
          bool doneBack = abs(endBack) >= limit, doneAhead = abs(endAhead) >= limit;
          for (int n = 0; n < EDGE_STEPS; n++) {
            if (doneBack && doneAhead) break;
            float stride = n == 0 ? 1.5 : (n < 3 ? 2.0 : (n == 3 ? 4.0 : 8.0));
            if (!doneBack) {
              back -= along * stride;
              endBack = luma(back) - lEdge;
              doneBack = abs(endBack) >= limit;
            }
            if (!doneAhead) {
              ahead += along * stride;
              endAhead = luma(ahead) - lEdge;
              doneAhead = abs(endAhead) >= limit;
            }
          }
          float toBack = level ? vUv.x - back.x : vUv.y - back.y;
          float toAhead = level ? ahead.x - vUv.x : ahead.y - vUv.y;
          float nearEnd = toBack < toAhead ? endBack : endAhead;
          // Only the side of the edge whose end turns away from this pixel is blended.
          float blend = (nearEnd < 0.0) != (lM < lEdge) ? 0.5 - min(toBack, toAhead) / (toBack + toAhead) : 0.0;
          float around = (2.0 * (lN + lS + lE + lW) + lNE + lNW + lSE + lSW) / 12.0;
          float lone = smoothstep(0.0, 1.0, clamp(abs(around - lM) / range, 0.0, 1.0));
          blend = max(blend, lone * lone * 0.6);
          // Blended after tone mapping, or a very bright pixel would swallow its neighbour.
          color = mix(display(here, glow), display(texture(tScene, vUv + across).rgb, glow), blend);
        }
      #else
        color = display(here, glow);
      #endif
      color = mix(color * 12.92, 1.055 * pow(max(color, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, color));
      // Noise of one display step with a triangular spread hides the bands of dark gradients.
      vec2 p = gl_FragCoord.xy;
      float n1 = fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
      float n2 = fract(52.9829189 * fract(dot(p + 31.7, vec2(0.00583715, 0.06711056))));
      fragColor = vec4(color + (n1 + n2 - 1.0) / 255.0, 1.0);
    }`;

  const look = { value: new THREE.Vector3() }, shown = new THREE.Vector2();
  const sceneTexel = { value: new THREE.Vector2() };

  function pass(fragmentShader, uniforms, defines) {
    return new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: VERTEX, fragmentShader, uniforms, defines: defines || {},
      depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    });
  }

  function draw(material, target) {
    screen.material = material;
    renderer.setRenderTarget(target);
    renderer.render(screen, flatCamera);
  }

  // The last pass, compiled for what is switched on.
  function finalPass(bloom) {
    const key = (bloom ? "b" : "-") + (samples ? "-" : "f") + DETAIL[tier].edgeSteps;
    if (!finals[key]) {
      const defines = { EDGE_STEPS: DETAIL[tier].edgeSteps };
      if (bloom) defines.BLOOM = 1;
      if (!samples) defines.FXAA = 1;
      finals[key] = pass(FINAL, {
        tScene: { value: null }, tBloom: { value: null }, uTexel: sceneTexel, uLook: look, uAspect: { value: 1 },
      }, defines);
    }
    return finals[key];
  }

  // The most samples per pixel the graphics card gives a target of this format, up to `wanted`.
  function supportedSamples(wanted, format) {
    if (wanted < 2) return 0;
    let best = 0;
    try {
      for (const count of gl.getInternalformatParameter(gl.RENDERBUFFER, format, gl.SAMPLES) || []) {
        if (count <= wanted && count > best) best = count;
      }
    } catch {
      best = 0;
    }
    return best;
  }

  function target(w, h, options) {
    return new THREE.WebGLRenderTarget(w, h, {
      type: hdr ? THREE.HalfFloatType : THREE.UnsignedByteType,
      // Without float targets the picture is kept in 8 bits with the sRGB curve: no light above 1, but
      // dark tones keep their steps.
      colorSpace: hdr ? THREE.LinearSRGBColorSpace : THREE.SRGBColorSpace,
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false, ...options,
    });
  }

  // Lets go of every target; `gone` when the graphics context that held them is lost (nothing to free).
  function free(gone) {
    if (!gone) for (const one of [sceneTarget, mirrorTarget, ...down, ...up]) if (one) one.dispose();
    sceneTarget = mirrorTarget = null;
    down.length = up.length = 0;
    api.mirror = null;
  }

  // Sizes the canvas and builds every target for the current level, resolution scale and window.
  function allocate() {
    if (!renderer || lost) return;
    const q = NV.q;
    const ratio = Math.min(window.devicePixelRatio || 1, q.pixelRatio);
    // The canvas is only touched when its size changes: setting it again would clear the picture.
    if (renderer.getPixelRatio() !== ratio || renderer.getSize(shown).x !== width || shown.y !== height) {
      renderer.setPixelRatio(ratio);
      renderer.setSize(width, height, false);
    }
    const w = Math.max(8, Math.round(width * ratio * scale)), h = Math.max(8, Math.round(height * ratio * scale));
    free();
    hdr = renderer.extensions.has("EXT_color_buffer_float") || renderer.extensions.has("EXT_color_buffer_half_float");
    const format = hdr ? gl.RGBA16F : gl.SRGB8_ALPHA8;
    samples = supportedSamples(q.msaa, format);
    sceneTarget = target(w, h, { samples, depthBuffer: true });
    sceneTexel.value.set(1 / w, 1 / h);
    if (q.bloom) {
      let bw = w >> 1, bh = h >> 1;
      for (let level = 0; level < DETAIL[tier].bloomLevels && Math.min(bw, bh) >= 4; level++) {
        down.push(target(bw, bh));
        bw >>= 1;
        bh >>= 1;
      }
      for (let level = 0; level < down.length - 1; level++) up.push(target(down[level].width, down[level].height));
    }
    if (q.reflections) {
      const share = DETAIL[tier].mirror;
      mirrorTarget = target(Math.max(4, Math.round(w * share)), Math.max(4, Math.round(h * share)), {
        samples: supportedSamples(Math.min(samples, DETAIL[tier].mirrorSamples), format), depthBuffer: true,
        generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter,
      });
      api.mirror = { texture: mirrorTarget.texture, textureMatrix };
    }
  }

  // ---- The floor's mirror image ----

  const mirrorCamera = new THREE.PerspectiveCamera();
  const textureMatrix = new THREE.Matrix4();
  const eye = new THREE.Vector3(), ahead = new THREE.Vector3(), spin = new THREE.Matrix4();
  const floorPlane = new THREE.Plane(), clip = new THREE.Vector4(), corner = new THREE.Vector4();
  const UP = new THREE.Vector3(0, 1, 0);
  const hiddenNow = [];

  // Draws the scene as seen from the camera's mirror image under the floor (scene y = 0). What that
  // camera sees of the real scene is what the floor shows of it, so nothing has to be flipped.
  function drawMirror(scene, camera) {
    camera.updateMatrixWorld();
    eye.setFromMatrixPosition(camera.matrixWorld);
    if (eye.y < 0.02) return;
    spin.extractRotation(camera.matrixWorld);
    ahead.set(0, 0, -1).applyMatrix4(spin);
    mirrorCamera.up.set(0, 1, 0).applyMatrix4(spin);
    mirrorCamera.up.y *= -1;
    mirrorCamera.position.set(eye.x, -eye.y, eye.z);
    ahead.set(eye.x + ahead.x, -eye.y - ahead.y, eye.z + ahead.z);
    mirrorCamera.lookAt(ahead);
    mirrorCamera.updateMatrixWorld();
    const projection = mirrorCamera.projectionMatrix.copy(camera.projectionMatrix);
    textureMatrix.set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);
    textureMatrix.multiply(projection).multiply(mirrorCamera.matrixWorldInverse);
    // The near plane is tilted onto the floor, so nothing under it is drawn (Eric Lengyel's oblique
    // near-plane clipping).
    floorPlane.set(UP, 0).applyMatrix4(mirrorCamera.matrixWorldInverse);
    clip.set(floorPlane.normal.x, floorPlane.normal.y, floorPlane.normal.z, floorPlane.constant);
    const e = projection.elements;
    corner.set((Math.sign(clip.x) + e[8]) / e[0], (Math.sign(clip.y) + e[9]) / e[5], -1, (1 + e[10]) / e[14]);
    clip.multiplyScalar(2 / clip.dot(corner));
    e[2] = clip.x;
    e[6] = clip.y;
    e[10] = clip.z + 1 - 0.003;
    e[14] = clip.w;
    mirrorCamera.projectionMatrixInverse.copy(projection).invert();

    hiddenNow.length = 0;
    const floor = NV.arena && NV.arena.floor;
    if (floor && floor.visible) hiddenNow.push(floor);
    for (const object of api.mirrorHidden) if (object.visible) hiddenNow.push(object);
    for (const object of hiddenNow) object.visible = false;
    renderer.setRenderTarget(mirrorTarget);
    renderer.render(scene, mirrorCamera);
    for (const object of hiddenNow) object.visible = true;
  }

  // ---- Frame times and the automatic level ----

  const auto = {
    on: false, step: 0, settle: 1.5, time: 0, frames: 0, worst: 0, slow: 0, fast: 0,
    raisedAt: -1e9, blocked: -1, blockedUntil: 0, backoff: 30, vsync: 16.7, cap: 1e9,
  };
  const timer = { ext: null, waiting: [], spare: [], running: null, ms: -1 };
  let last = 0, hiddenSince = false;

  // Every level and resolution scale in one order, lowest first.
  const rank = (name, step) => ORDER.indexOf(name) * SCALES.length + SCALES.length - 1 - step;

  function guess() {
    const name = rendererName.toLowerCase();
    if (/swiftshader|llvmpipe|softpipe|software|basic render|warp/.test(name)) return "low";
    if (/mali|adreno|powervr|videocore|vivante|tegra/.test(name)) return "low";
    if (/intel/.test(name)) {
      if (/\barc\b/.test(name)) return "high";
      return /iris|uhd|xe\b/.test(name) ? "medium" : "low";
    }
    if (/radeon/.test(name) && !/\brx\b|\bpro\b|firepro/.test(name)) {
      if (/radeon( ?\(tm\))? (hd|r[2-7])\b/.test(name)) return "low";
      if (/graphics|vega|\d{3}m\b/.test(name)) return "medium";
    }
    if (/geforce (gt |mx|\d{3}m\b)/.test(name)) return "medium";
    return "high";
  }

  function apply(name, step) {
    const changed = tier !== name || NV.q.tier !== name;
    tier = name;
    auto.step = step;
    scale = SCALES[step];
    Object.assign(NV.q, TIERS[name], { tier: name });
    allocate();
    auto.settle = 1.5;
    auto.time = auto.frames = auto.worst = auto.slow = auto.fast = 0;
    if (changed) NV.emit("quality", { q: NV.q });
  }

  function choose(name) {
    auto.on = name === "auto" || !TIERS[name];
    auto.blocked = -1;
    auto.backoff = 30;
    auto.cap = 1e9;
    // The automatic level starts where it settled last time on this graphics card, else from a guess.
    const kept = auto.on && NV.prefs.qualityAuto;
    if (kept && kept.renderer === rendererName && TIERS[kept.tier] && SCALES[kept.step]) apply(kept.tier, kept.step);
    else apply(auto.on ? guess() : name, 0);
  }

  function keep() {
    NV.setPref("qualityAuto", { tier, step: auto.step, renderer: rendererName });
  }

  function stepDown(far, now) {
    // A step up that cannot be held is not tried again for a while, and longer each time it fails.
    if (now - auto.raisedAt < 8000) {
      auto.blocked = rank(tier, auto.step);
      auto.blockedUntil = now + auto.backoff * 1000;
      auto.backoff = Math.min(auto.backoff * 3, 900);
    }
    const at = ORDER.indexOf(tier);
    const lastStep = at ? 2 : SCALES.length - 1;
    if (auto.step < lastStep && !(far && at)) apply(tier, auto.step + 1);
    else if (at) apply(ORDER[at - 1], 0);
    keep();
  }

  function stepUp(now) {
    const at = ORDER.indexOf(tier);
    if (!auto.step && at === ORDER.length - 1) return;
    const name = auto.step ? tier : ORDER[at + 1], step = Math.max(0, auto.step - 1);
    if (auto.blocked >= 0 && rank(name, step) >= auto.blocked && now < auto.blockedUntil) return;
    auto.raisedAt = now;
    apply(name, step);
    keep();
  }

  // Called once per drawn frame: keeps the window of frame times and moves the automatic level.
  function measure(now) {
    const ms = now - last;
    last = now;
    if (document.hidden || hiddenSince || ms > 250 || ms <= 0) {
      // A hidden tab, or a stall that says nothing about the scene: start the window again.
      hiddenSince = document.hidden;
      auto.settle = Math.max(auto.settle, 1);
      auto.time = auto.frames = auto.worst = 0;
      return;
    }
    if (!auto.on) return;
    if (auto.settle > 0) {
      auto.settle -= ms / 1000;
      return;
    }
    auto.time += ms;
    auto.frames++;
    auto.worst = Math.max(auto.worst, ms);
    if (auto.time < 1000) return;
    const mean = auto.time / auto.frames, fps = 1000 / mean, worst = auto.worst;
    auto.time = auto.frames = auto.worst = 0;
    auto.vsync = Math.min(auto.vsync, mean);
    if (fps > auto.cap * 1.1) auto.cap = 1e9;
    const target = Math.min(TARGET_FPS, auto.cap * 0.9);
    if (fps < target) {
      auto.fast = 0;
      if (++auto.slow < 2 && fps > target * 0.72) return;
      // Nothing left to lower and every frame as slow as the next: the display or the browser sets this
      // pace (a 50 Hz screen, a power saver), not the picture. That pace is the aim from now on.
      if (tier === ORDER[0] && auto.step === SCALES.length - 1 && worst < 1.3 * mean) auto.cap = fps;
      else stepDown(fps < target * 0.5, now);
      return;
    }
    auto.slow = 0;
    // Room to spare: every frame of the window on time and, where the graphics card reports its time,
    // under half of a frame's budget. Without that report a step up is a careful trial.
    const pace = Math.max(auto.vsync, 1000 / 144, 1000 / auto.cap);
    const steady = worst < 1.4 * pace && fps > Math.min(58, auto.cap * 0.96);
    const spare = !timer.ext || (timer.ms >= 0 && timer.ms < (auto.step ? 8.5 : 6.5));
    if (steady && spare) {
      if (++auto.fast >= (timer.ext ? 5 : 12)) {
        auto.fast = 0;
        stepUp(now);
      }
    } else auto.fast = 0;
  }

  // The time the graphics card spends on a frame, where the browser tells: a smoothed value, -1 before
  // the first answer. Only the automatic level asks.
  function timerBegin() {
    if (!timer.ext || !auto.on) return;
    const first = timer.waiting[0];
    if (first && gl.getQueryParameter(first, gl.QUERY_RESULT_AVAILABLE)) {
      const ms = gl.getQueryParameter(first, gl.QUERY_RESULT) / 1e6;
      if (!gl.getParameter(timer.ext.GPU_DISJOINT_EXT)) timer.ms = timer.ms < 0 ? ms : NV.lerp(timer.ms, ms, 0.1);
      timer.spare.push(timer.waiting.shift());
    }
    if (timer.waiting.length < 3) {
      timer.running = timer.spare.pop() || gl.createQuery();
      gl.beginQuery(timer.ext.TIME_ELAPSED_EXT, timer.running);
    }
  }

  function timerEnd() {
    if (!timer.running) return;
    gl.endQuery(timer.ext.TIME_ELAPSED_EXT);
    timer.waiting.push(timer.running);
    timer.running = null;
  }

  // Forgets the queries; `gone` when the graphics context that made them is lost.
  function timerReset(gone) {
    if (!gone) for (const query of [...timer.waiting, ...timer.spare]) gl.deleteQuery(query);
    timer.waiting.length = timer.spare.length = 0;
    timer.running = null;
    timer.ms = -1;
  }

  function onLost() {
    lost = true;
    timerReset(true);
  }

  function onRestored() {
    lost = false;
    timer.ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    free(true);
    allocate();
  }

  const api = {
    // Brightness of the picture before the tone curve; 1 keeps middle grey where the scene put it.
    exposure: 1,
    // Bloom: how much of the spread light is added, the brightness where it starts and the width of the
    // soft start below it, and how much each level leans on the wider one (0..1).
    bloomStrength: 0.55,
    bloomThreshold: 1,
    bloomKnee: 0.5,
    bloomSpread: 0.62,
    // Darkening toward the corners (0 = none).
    vignette: 0.22,
    mirror: null,
    mirrorHidden: [],

    init(webglRenderer) {
      renderer = webglRenderer;
      gl = renderer.getContext();
      canvas = renderer.domElement;
      renderer.toneMapping = THREE.NoToneMapping;
      const names = gl.getExtension("WEBGL_debug_renderer_info");
      rendererName = String((names && gl.getParameter(names.UNMASKED_RENDERER_WEBGL)) || gl.getParameter(gl.RENDERER) || "");
      timer.ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
      triangle = new THREE.BufferGeometry();
      triangle.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
      screen = new THREE.Mesh(triangle, null);
      screen.frustumCulled = false;
      flatCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      const curve = { value: new THREE.Vector3() };
      const source = () => ({ tSource: { value: null }, uTexel: { value: new THREE.Vector2() }, uCurve: curve });
      prefilter = pass(SHRINK, source(), { FIRST: 1 });
      shrink = pass(SHRINK, source());
      grow = pass(GROW, {
        tSmall: { value: null }, tSharp: { value: null }, uTexel: { value: new THREE.Vector2() }, uSpread: { value: 0.6 },
      });
      canvas.addEventListener("webglcontextlost", onLost);
      canvas.addEventListener("webglcontextrestored", onRestored);
      last = performance.now();
      choose(NV.prefs.quality || "auto");
    },

    // The size of the picture in CSS pixels.
    resize(w, h) {
      width = w;
      height = h;
      allocate();
    },

    setQuality(name) {
      const known = name === "auto" || TIERS[name] ? name : "auto";
      NV.setPref("quality", known);
      choose(known);
    },

    info() {
      return { tier, auto: auto.on, fps: Math.round(1000 / Math.max(1, NV.frameMs || 16.7)), scale, renderer: rendererName };
    },

    render(scene, camera) {
      if (!renderer || lost || !sceneTarget) return;
      measure(performance.now());
      timerBegin();
      const q = NV.q, shadowMap = renderer.shadowMap, shadowUpdates = shadowMap.autoUpdate;
      if (mirrorTarget) {
        drawMirror(scene, camera);
        // The shadows were drawn for the mirror image; the same ones serve the main picture.
        shadowMap.autoUpdate = false;
      }
      renderer.setRenderTarget(sceneTarget);
      renderer.render(scene, camera);
      shadowMap.autoUpdate = shadowUpdates;

      const bloom = q.bloom && down.length > 0;
      let glow = null;
      if (bloom) {
        const limit = hdr ? api.bloomThreshold : Math.min(api.bloomThreshold, 0.8);
        prefilter.uniforms.uCurve.value.set(limit, Math.max(1e-3, api.bloomKnee * limit), api.exposure);
        let source = sceneTarget;
        for (let level = 0; level < down.length; level++) {
          const material = level ? shrink : prefilter;
          material.uniforms.tSource.value = source.texture;
          material.uniforms.uTexel.value.set(1 / source.width, 1 / source.height);
          draw(material, down[level]);
          source = down[level];
        }
        grow.uniforms.uSpread.value = api.bloomSpread;
        for (let level = down.length - 2; level >= 0; level--) {
          grow.uniforms.tSmall.value = source.texture;
          grow.uniforms.tSharp.value = down[level].texture;
          grow.uniforms.uTexel.value.set(1 / source.width, 1 / source.height);
          draw(grow, up[level]);
          source = up[level];
        }
        glow = source.texture;
      }
      const final = finalPass(bloom);
      final.uniforms.tScene.value = sceneTarget.texture;
      final.uniforms.tBloom.value = glow;
      final.uniforms.uAspect.value = width / height;
      look.value.set(api.exposure, api.vignette, api.bloomStrength);
      draw(final, null);
      timerEnd();
    },

    dispose() {
      if (!renderer) return;
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
      timerReset(lost);
      free();
      for (const material of [prefilter, shrink, grow, ...Object.values(finals)]) material.dispose();
      for (const key of Object.keys(finals)) delete finals[key];
      triangle.dispose();
      renderer = null;
    },
  };
  return api;
})();
