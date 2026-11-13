// Effects: boost flames and their trails, supersonic wheel trails, the ball's trail, touches, demolitions
// and goal explosions; dust and scrape sparks. Everything is drawn from a few pooled, instanced meshes:
//   sprites   camera-facing or flat quads (soft discs, rings, sparks, smoke). A sprite is written once
//             when it is born and the graphics card moves, grows and fades it from the effect clock, so
//             a frame only costs what it adds.
//   ribbons   strips of points left behind an emitter, widened toward the camera.
//   flames    one cone of two shells per car, shaped and lit by a noise shader.
// All colours are linear light, far above 1 where something glows; the bloom does the rest. The effect
// clock is game time: it stands still while the game is paused.

NV.fx = (() => {
  const S = NV.S, FIELD = NV.FIELD;
  // Sprite shapes, and what can be added to them: FLAT lies in the plane whose normal is given as the
  // velocity, STEADY is rewritten every frame and does not age.
  const DISC = 0, RING = 1, SPARK = 2, SMOKE = 3, FLAT = 8, STEADY = 16;
  const STRIDE = 20;
  const WHITE = new THREE.Color(1, 1, 1), FIRE = new THREE.Color(1, 0.42, 0.1), UP = new THREE.Vector3(0, 1, 0);
  const rand = (lo, hi) => lo + Math.random() * (hi - lo);
  const v = () => new THREE.Vector3();

  const clock = { value: 0 }, since = { value: 0 }, pictureHeight = { value: 720 };
  let time = 0, lastReal = 0, warm = true, scene = null, noise = null;
  let particles = null, glows = null, ribbons = null, flames = null, debris = null, flash = null;
  const waves = [], bursts = [], cars = [], tints = [], owned = [];
  const ball = { seen: false, last: v() };
  let history = null;

  // The sprite being written: set the fields, then add it.
  const P = {
    x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, life: 1, r: 1, g: 1, b: 1, opacity: 0, size0: 1, size1: 1,
    drag: 0, gravity: 0, shape: DISC, turn: 0, curve: 1,
  };
  const at = (p) => {
    P.x = p.x;
    P.y = p.y;
    P.z = p.z;
  };
  const going = (x, y, z) => {
    P.vx = x;
    P.vy = y;
    P.vz = z;
  };
  const lit = (color, gain) => {
    P.r = color.r * gain;
    P.g = color.g * gain;
    P.b = color.b * gain;
  };
  // A random direction, in `out`; with `toward` only the half of the sphere on that side.
  function scatter(out, toward) {
    const y = rand(-1, 1), turn = rand(0, Math.PI * 2), flat = Math.sqrt(1 - y * y);
    out.set(Math.cos(turn) * flat, y, Math.sin(turn) * flat);
    if (toward && out.dot(toward) < 0) out.reflect(toward);
    return out;
  }

  // Smooth noise that repeats at its edges: three sizes of blobs in r, g, b and their mix in a.
  function noiseTexture() {
    const size = 64, data = new Uint8Array(size * size * 4);
    let seed = 7;
    const next = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const layers = [4, 8, 16].map((cells) => {
      const values = Float32Array.from({ length: cells * cells }, next);
      return (x, y) => {
        const fx = x * cells, fy = y * cells, x0 = Math.floor(fx), y0 = Math.floor(fy);
        const kx = NV.smooth(fx - x0), ky = NV.smooth(fy - y0);
        const value = (i, j) => values[(j % cells) * cells + (i % cells)];
        const low = NV.lerp(value(x0, y0), value(x0 + 1, y0), kx), high = NV.lerp(value(x0, y0 + 1), value(x0 + 1, y0 + 1), kx);
        return NV.lerp(low, high, ky);
      };
    });
    for (let j = 0; j < size; j++) {
      for (let i = 0; i < size; i++) {
        const o = (j * size + i) * 4, n = layers.map((layer) => layer(i / size, j / size));
        data[o] = n[0] * 255;
        data[o + 1] = n[1] * 255;
        data[o + 2] = n[2] * 255;
        data[o + 3] = (n[0] * 0.5 + n[1] * 0.3 + n[2] * 0.2) * 255;
      }
    }
    const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.minFilter = texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    return texture;
  }

  // ---- Sprites ----

  const SPRITE_VERTEX = `
    attribute vec4 aPlace;   // where it was born, and when
    attribute vec4 aMotion;  // velocity (or the plane's normal), life in seconds
    attribute vec4 aColor;   // light, and opacity (0 = only adds light)
    attribute vec4 aSize;    // radius at birth and at the end, drag, gravity
    attribute vec4 aKind;    // shape, seed, turn (or a spark's length per speed), fade power
    uniform float uTime;
    uniform float uSince;
    uniform float uHeight;
    varying vec2 vUv;
    varying vec4 vColor;
    varying vec4 vKind;
    void main() {
      float age = uTime - aPlace.w;
      float shape = aKind.x;
      float steady = step(15.5, shape);
      shape -= steady * 16.0;
      float flat_ = step(7.5, shape);
      shape -= flat_ * 8.0;
      float k = steady > 0.5 ? 0.0 : age / aMotion.w;
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      if (k < 0.0 || k >= 1.0 || aPlace.w < uSince) return;
      float size = mix(aSize.x, aSize.y, 1.0 - (1.0 - k) * (1.0 - k));
      vec3 centre = aPlace.xyz, right, up;
      float ground = 1.0, thin = 1.0;
      if (flat_ > 0.5) {
        vec3 n = normalize(aMotion.xyz);
        // Light lying on the floor is not part of the floor's own mirror image.
        if (cameraPosition.y < 0.0 && n.y > 0.9) return;
        right = normalize(cross(abs(n.y) > 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), n)) * size;
        up = cross(n, right);
      } else {
        float drag = aSize.z, slowed = exp(-drag * age);
        centre += aMotion.xyz * (drag > 0.0 ? (1.0 - slowed) / drag : age);
        centre.y -= 0.5 * aSize.w * age * age;
        // Nothing is drawn thinner than about a pixel: it is widened and dimmed instead, so far sparks
        // and embers do not flicker.
        float pixel = -2.0 * (viewMatrix * vec4(centre, 1.0)).z / (projectionMatrix[1][1] * uHeight);
        if (shape > 1.5 && shape < 2.5) {
          if (centre.y < 0.0) centre.y *= -0.3;
          vec3 velocity = aMotion.xyz * slowed - vec3(0.0, aSize.w * age, 0.0);
          float speed = length(velocity);
          vec3 along = velocity / max(speed, 1e-4);
          float wide = max(size, pixel);
          thin = size / wide;
          right = along * max(wide, speed * aKind.z);
          up = normalize(cross(along, cameraPosition - centre)) * wide;
          centre -= right;
        } else {
          float wide = max(size, 1.2 * pixel);
          thin = (size * size) / (wide * wide);
          size = wide;
          float angle = aKind.y * 6.2832 + k * aKind.z;
          vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
          vec3 camUp = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
          right = (camRight * cos(angle) + camUp * sin(angle)) * size;
          up = (camUp * cos(angle) - camRight * sin(angle)) * size;
          // Its height over the floor, in the lengths over which it fades out instead of being cut.
          ground = (centre.y + right.y * position.x + up.y * position.y) / min(0.5 * size, 1.5);
        }
      }
      float fade = steady > 0.5 ? 1.0 : pow(max(1.0 - k, 1e-4), aKind.w) * min(k * 30.0, 1.0);
      vUv = position.xy;
      // Light that only adds cools as it ages: it keeps its strongest colour and loses the rest.
      float peak = max(aColor.r, max(aColor.g, aColor.b)) + 1e-5;
      vec3 light = aColor.a > 0.0 ? aColor.rgb * aColor.a : aColor.rgb * pow(max(aColor.rgb / peak, vec3(1e-4)), vec3(1.6 * k));
      vColor = vec4(light, aColor.a) * fade * thin;
      vKind = vec4(shape, aKind.y, k, ground);
      gl_Position = projectionMatrix * viewMatrix * vec4(centre + right * position.x + up * position.y, 1.0);
    }`;

  const SPRITE_FRAGMENT = `
    uniform sampler2D tNoise;
    varying vec2 vUv;
    varying vec4 vColor;
    varying vec4 vKind;
    void main() {
      float r2 = dot(vUv, vUv), shape = vKind.x, k = vKind.z;
      float soft = max(1.0 - r2, 0.0), a;
      if (shape < 0.5) {
        a = soft * soft * soft;
      } else if (shape < 1.5) {
        float width = mix(0.2, 0.03, k);
        float band = clamp(1.0 - abs(sqrt(r2) - (1.0 - width)) / width, 0.0, 1.0);
        a = band * band;
      } else if (shape < 2.5) {
        float across = 1.0 - abs(vUv.y);
        a = across * across * smoothstep(-1.0, 0.7, vUv.x) * (1.0 - smoothstep(0.8, 1.0, vUv.x));
      } else {
        float n = texture2D(tNoise, vUv * 0.3 + vKind.y * 7.0).a;
        a = soft * soft * smoothstep(0.2, 0.7, n + soft * 0.45 - k * 0.25);
      }
      gl_FragColor = vColor * a * smoothstep(0.0, 1.0, vKind.w);
    }`;

  class Sprites {
    constructor(capacity, material) {
      this.capacity = capacity;
      this.data = new Float32Array(capacity * STRIDE);
      this.buffer = new THREE.InstancedInterleavedBuffer(this.data, STRIDE);
      this.buffer.setUsage(THREE.DynamicDrawUsage);
      const geometry = new THREE.InstancedBufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0], 3));
      geometry.setIndex([0, 1, 2, 2, 1, 3]);
      ["aPlace", "aMotion", "aColor", "aSize", "aKind"].forEach((name, i) => {
        geometry.setAttribute(name, new THREE.InterleavedBufferAttribute(this.buffer, 4, i * 4));
      });
      geometry.instanceCount = 0;
      this.mesh = new THREE.Mesh(geometry, material);
      this.mesh.frustumCulled = false;
      this.head = this.first = this.added = 0;
      owned.push(geometry);
    }

    // Writes P as the next sprite, born `late` seconds ago (negative: still to come).
    add(late) {
      const d = this.data, o = this.head * STRIDE;
      d[o] = P.x;
      d[o + 1] = P.y;
      d[o + 2] = P.z;
      d[o + 3] = time - (late || 0);
      d[o + 4] = P.vx;
      d[o + 5] = P.vy;
      d[o + 6] = P.vz;
      d[o + 7] = P.life;
      d[o + 8] = P.r;
      d[o + 9] = P.g;
      d[o + 10] = P.b;
      d[o + 11] = P.opacity;
      d[o + 12] = P.size0;
      d[o + 13] = P.size1;
      d[o + 14] = P.drag;
      d[o + 15] = P.gravity;
      d[o + 16] = P.shape;
      d[o + 17] = Math.random();
      d[o + 18] = P.turn;
      d[o + 19] = P.curve;
      this.head = (this.head + 1) % this.capacity;
      this.added++;
    }

    // Hands what was added since the last call to the graphics card. The oldest sprites are written
    // over when the pool is full.
    send() {
      if (!this.added) return;
      const geometry = this.mesh.geometry, end = this.first + this.added;
      geometry.instanceCount = Math.min(this.capacity, Math.max(geometry.instanceCount, end));
      if (end <= this.capacity) this.buffer.addUpdateRange(this.first * STRIDE, this.added * STRIDE);
      else if (this.added < this.capacity) {
        this.buffer.addUpdateRange(this.first * STRIDE, (this.capacity - this.first) * STRIDE);
        this.buffer.addUpdateRange(0, (end - this.capacity) * STRIDE);
      }
      this.buffer.needsUpdate = true;
      this.first = this.head;
      this.added = 0;
    }

    // For sprites rewritten every frame: draws exactly those added since the last call.
    sendAll() {
      this.mesh.geometry.instanceCount = this.added;
      if (this.added) {
        this.buffer.addUpdateRange(0, this.added * STRIDE);
        this.buffer.needsUpdate = true;
      }
      this.head = this.first = this.added = 0;
    }
  }

  // ---- Ribbons ----

  const RIBBON_VERTEX = `
    attribute vec4 aPoint;  // where, and when it was laid
    attribute vec4 aAim;    // the way the emitter was going, strength
    attribute vec4 aStyle;  // light, half width
    attribute vec4 aSide;   // -1 | 1, life in seconds, width at the end of it (1 = as laid), fade-in share of life
    uniform float uTime;
    uniform float uHeight;
    varying vec4 vLight;
    varying vec3 vWorld;
    void main() {
      float k = clamp((uTime - aPoint.w) / aSide.y, 0.0, 1.0);
      float fade = aAim.w * (1.0 - k) * min(k / aSide.w, 1.0);
      vec3 toEye = cameraPosition - aPoint.xyz;
      vec3 across = cross(aAim.xyz, toEye);
      float size = length(across);
      across = size > 1e-5 && fade > 0.0 ? across / size : vec3(0.0);
      // Seen from its end a ribbon has no width to show: it fades out instead of flapping.
      fade *= smoothstep(0.06, 0.3, size / length(toEye));
      // Never thinner than about a pixel: widened and dimmed instead, so a far ribbon is not a dotted line.
      float pixel = -2.0 * (viewMatrix * vec4(aPoint.xyz, 1.0)).z / (projectionMatrix[1][1] * uHeight);
      float half_ = aStyle.w * mix(1.0, aSide.z, k), wide = max(half_, 1.3 * pixel);
      vLight = vec4(aStyle.rgb * fade * fade * half_ / wide, aSide.x);
      vWorld = aPoint.xyz;
      gl_Position = projectionMatrix * viewMatrix * vec4(aPoint.xyz + across * aSide.x * wide, 1.0);
    }`;

  const RIBBON_FRAGMENT = `
    uniform sampler2D tNoise;
    varying vec4 vLight;
    varying vec3 vWorld;
    void main() {
      float edge = 1.0 - vLight.a * vLight.a;
      // Streaks that run along the ribbon.
      float n = texture2D(tNoise, vec2(vLight.a * 0.3, (vWorld.x + vWorld.y + vWorld.z) * 0.11)).g;
      gl_FragColor = vec4(vLight.rgb * edge * edge * (0.65 + 0.7 * n), 1.0);
    }`;

  class Ribbons {
    constructor(count, points) {
      this.points = points;
      const verts = count * points * 2, index = [];
      this.data = new Float32Array(verts * 8);
      this.buffer = new THREE.InterleavedBuffer(this.data, 8);
      this.buffer.setUsage(THREE.DynamicDrawUsage);
      this.side = new THREE.Float32BufferAttribute(verts * 4, 4);
      this.look = new THREE.Float32BufferAttribute(verts * 4, 4);
      for (let p = 0; p < count * points; p++) {
        this.side.setX(p * 2, -1);
        this.side.setX(p * 2 + 1, 1);
        if (p % points < points - 1) index.push(p * 2, p * 2 + 1, p * 2 + 2, p * 2 + 2, p * 2 + 1, p * 2 + 3);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("aPoint", new THREE.InterleavedBufferAttribute(this.buffer, 4, 0));
      geometry.setAttribute("aAim", new THREE.InterleavedBufferAttribute(this.buffer, 4, 4));
      geometry.setAttribute("aSide", this.side);
      geometry.setAttribute("aStyle", this.look);
      geometry.setIndex(index);
      this.mesh = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
        uniforms: { uTime: clock, uHeight: pictureHeight, tNoise: { value: noise } },
        vertexShader: RIBBON_VERTEX, fragmentShader: RIBBON_FRAGMENT,
        blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, side: THREE.DoubleSide,
      }));
      this.mesh.frustumCulled = false;
      this.open = new Uint8Array(count);
      this.laid = new Float32Array(count);
      this.changed = false;
      this.clear();
      owned.push(geometry, this.mesh.material);
    }

    // How a ribbon looks: its light, half its width, how long a point lasts, how wide it ends and the
    // share of its life it takes to appear.
    style(ribbon, color, gain, halfWidth, life, endWidth, fadeIn) {
      for (let p = ribbon * this.points * 2, end = p + this.points * 2; p < end; p++) {
        this.look.setXYZW(p, color.r * gain, color.g * gain, color.b * gain, halfWidth);
        this.side.setY(p, life);
        this.side.setZ(p, endWidth);
        this.side.setW(p, fadeIn);
      }
      this.look.needsUpdate = this.side.needsUpdate = true;
    }

    // Moves the points one place toward the old end; the newest place is then free.
    shift(ribbon) {
      const first = ribbon * this.points * 16;
      this.data.copyWithin(first, first + 16, first + this.points * 16);
    }

    // The newest point: at (x, y, z), aimed along the way from the point before it.
    write(ribbon, x, y, z, strength) {
      const d = this.data, o = (ribbon * this.points + this.points - 1) * 16, b = o - 16;
      let dx = x - d[b], dy = y - d[b + 1], dz = z - d[b + 2];
      const length = Math.hypot(dx, dy, dz);
      if (length > 1e-4) {
        dx /= length;
        dy /= length;
        dz /= length;
      } else {
        dx = d[b + 4];
        dy = d[b + 5];
        dz = d[b + 6];
      }
      for (let q = o; q < o + 16; q += 8) {
        d[q] = x;
        d[q + 1] = y;
        d[q + 2] = z;
        d[q + 3] = time;
        d[q + 4] = dx;
        d[q + 5] = dy;
        d[q + 6] = dz;
        d[q + 7] = strength;
      }
      this.changed = true;
    }

    // The emitter is at `p` now. The newest point follows it; a point is left behind 75 times a second.
    push(ribbon, p, strength) {
      if (!this.open[ribbon]) {
        // A dark point first: the stretch since the ribbon last ended stays empty.
        this.shift(ribbon);
        this.write(ribbon, p.x, p.y, p.z, 0);
        this.open[ribbon] = 1;
        this.laid[ribbon] = -1e9;
      }
      if (time - this.laid[ribbon] >= 1 / 75) {
        this.shift(ribbon);
        this.laid[ribbon] = time;
      }
      this.write(ribbon, p.x, p.y, p.z, strength);
    }

    end(ribbon) {
      if (!this.open[ribbon]) return;
      const d = this.data, o = (ribbon * this.points + this.points - 1) * 16;
      this.shift(ribbon);
      this.write(ribbon, d[o], d[o + 1], d[o + 2], 0);
      this.open[ribbon] = 0;
    }

    clear() {
      this.data.fill(0);
      for (let o = 3; o < this.data.length; o += 8) this.data[o] = -1e9;
      this.open.fill(0);
      this.changed = true;
    }

    send() {
      if (this.changed) this.buffer.needsUpdate = true;
      this.changed = false;
    }
  }

  // ---- Flames ----

  const FLAME_VERTEX = `
    attribute vec3 aShape;   // along the flame 0..1, angle around it, shell (0 outer, 1 inner)
    attribute vec4 aNozzle;  // where the flame starts, how strongly it burns 0..1
    attribute vec4 aTurn;    // the car's rotation
    attribute vec4 aTint;    // light, seed
    uniform float uTime;
    varying vec4 vShape;
    varying vec4 vTint;
    varying float vFacing;
    vec3 turned(vec3 p) {
      return p + 2.0 * cross(aTurn.xyz, cross(aTurn.xyz, p) + aTurn.w * p);
    }
    void main() {
      float t = aShape.x, inner = aShape.z, power = aNozzle.w, seed = aTint.w;
      float flicker = 0.9 + 0.07 * sin(uTime * 53.0 + seed * 40.0) + 0.05 * sin(uTime * 131.0 + seed * 17.0);
      float reach = mix(1.25, 0.62, inner) * (0.25 + 0.75 * power) * flicker;
      float radius = mix(0.105, 0.05, inner) * (0.6 + 0.4 * smoothstep(0.0, 0.15, t)) * pow(max(1.0 - t, 0.0), 0.7) * sqrt(power);
      vec2 sway = vec2(sin(uTime * 23.0 + t * 7.0 + seed * 50.0), cos(uTime * 19.0 + t * 9.0 + seed * 30.0)) * 0.055 * t * t;
      vec3 ring = vec3(0.0, cos(aShape.y), sin(aShape.y));
      vec3 world = aNozzle.xyz + turned(vec3(-t * reach, sway.x, sway.y) + ring * radius);
      vFacing = abs(dot(turned(ring), normalize(cameraPosition - world)));
      vShape = vec4(t, aShape.y / 6.2832, inner, power);
      vTint = aTint;
      gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
    }`;

  const FLAME_FRAGMENT = `
    uniform sampler2D tNoise;
    uniform float uTime;
    varying vec4 vShape;
    varying vec4 vTint;
    varying float vFacing;
    void main() {
      float t = vShape.x, inner = vShape.z, power = vShape.w;
      vec2 uv = vec2(vShape.y * 2.0 + vTint.w, t * 0.9 - uTime * 3.7);
      float n = texture2D(tNoise, uv).g * 0.6 + texture2D(tNoise, uv * vec2(2.0, 2.6) + 0.37).b * 0.4;
      // The tip is eaten away by the noise, the sides thin out toward the silhouette.
      float body = pow(max(1.0 - t, 0.0), 1.5) * smoothstep(0.0, 0.3, n + 0.55 - t * 1.3);
      float heat = body * mix(mix(0.25, 0.05, inner), 1.0, pow(max(vFacing, 0.0), 1.0 + 0.6 * inner)) * (0.3 + 1.4 * n) * power;
      // Bright knots at even steps, as in a jet's exhaust.
      heat *= 1.0 + inner * 0.7 * pow(max(sin(t * 26.0 - 0.6), 0.0), 3.0) * (1.0 - t);
      vec3 light = mix(vTint.rgb * 2.8, mix(vTint.rgb, vec3(1.0), 0.45) * 5.0, inner);
      gl_FragColor = vec4(light * heat, 1.0);
    }`;

  function buildFlames(count) {
    const shape = [], index = [], along = 14, around = 16;
    for (let shell = 0; shell < 2; shell++) {
      const base = shape.length / 3;
      for (let i = 0; i <= along; i++) {
        for (let j = 0; j <= around; j++) shape.push(i / along, (j / around) * Math.PI * 2, shell);
      }
      for (let i = 0; i < along; i++) {
        for (let j = 0; j < around; j++) {
          const a = base + i * (around + 1) + j, b = a + around + 1;
          index.push(a, b, a + 1, a + 1, b, b + 1);
        }
      }
    }
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute("aShape", new THREE.Float32BufferAttribute(shape, 3));
    geometry.setIndex(index);
    const attributes = {};
    for (const name of ["aNozzle", "aTurn", "aTint"]) {
      attributes[name] = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4);
      attributes[name].setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, attributes[name]);
    }
    geometry.instanceCount = 0;
    const material = new THREE.ShaderMaterial({
      uniforms: { uTime: clock, tNoise: { value: noise } }, vertexShader: FLAME_VERTEX, fragmentShader: FLAME_FRAGMENT,
      blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    owned.push(geometry, material);
    return { mesh, geometry, ...attributes };
  }

  // ---- Shockwaves, the flash light and debris ----

  const WAVE_VERTEX = `
    varying vec3 vNormal;
    varying vec3 vToEye;
    varying vec3 vWorld;
    void main() {
      vec4 world = modelMatrix * vec4(position, 1.0);
      vNormal = normal;
      vToEye = cameraPosition - world.xyz;
      vWorld = world.xyz;
      gl_Position = projectionMatrix * viewMatrix * world;
    }`;

  const WAVE_FRAGMENT = `
    uniform sampler2D tNoise;
    uniform vec4 uLight;
    varying vec3 vNormal;
    varying vec3 vToEye;
    varying vec3 vWorld;
    void main() {
      // Only the shell's edge, where the eye looks along it, shows.
      float rim = pow(max(1.0 - abs(dot(normalize(vNormal), normalize(vToEye))), 0.0), 6.0);
      float n = texture2D(tNoise, vWorld.xz * 0.05 + vWorld.y * 0.03).a;
      gl_FragColor = vec4(uLight.rgb * uLight.a * rim * (0.4 + 1.2 * n), 1.0);
    }`;

  function buildWaves(count) {
    const geometry = new THREE.SphereGeometry(1, 48, 24);
    owned.push(geometry);
    for (let i = 0; i < count; i++) {
      const material = new THREE.ShaderMaterial({
        uniforms: { tNoise: { value: noise }, uLight: { value: new THREE.Vector4() } },
        vertexShader: WAVE_VERTEX, fragmentShader: WAVE_FRAGMENT,
        blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.visible = false;
      mesh.renderOrder = 6;
      scene.add(mesh);
      owned.push(material);
      waves.push({ mesh, born: 0, life: 1, reach: 1, color: new THREE.Color(), gain: 1 });
    }
  }

  function wave(p, color, gain, reach, life) {
    let pick = waves[0];
    for (const one of waves) if (!one.mesh.visible || one.born < pick.born) pick = one;
    pick.born = time;
    pick.life = life;
    pick.reach = reach;
    pick.gain = gain;
    pick.color.copy(color);
    pick.mesh.position.copy(p);
    pick.mesh.visible = true;
  }

  // A burst of light on the scene: the strongest one drives the one flash light.
  function burst(p, color, strength, life) {
    let pick = bursts[0];
    for (const one of bursts) if (one.born < pick.born) pick = one;
    pick.born = time;
    pick.life = life;
    pick.strength = strength;
    pick.color.copy(color);
    pick.at.copy(p);
    pick.at.y = Math.max(pick.at.y, 1.5);
  }

  const chunk = new THREE.Object3D();
  function buildDebris(count) {
    const geometry = new THREE.TetrahedronGeometry(0.11);
    geometry.scale(1.5, 0.5, 1);
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55, metalness: 0.6 });
    const mesh = new THREE.InstancedMesh(geometry, material, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.setColorAt(0, WHITE);
    mesh.count = 0;
    mesh.frustumCulled = false;
    owned.push(geometry, material);
    return {
      mesh, count, next: 0, live: 0,
      pos: new Float32Array(count * 3), vel: new Float32Array(count * 3), axis: new Float32Array(count * 3),
      spin: new Float32Array(count), born: new Float32Array(count).fill(-1e9), life: new Float32Array(count),
      size: new Float32Array(count), color: Array.from({ length: count }, () => new THREE.Color()),
    };
  }

  function debrisStep(dt) {
    const d = debris;
    if (!d.live && !d.mesh.count) return;
    let drawn = 0;
    d.live = 0;
    for (let i = 0; i < d.count; i++) {
      const age = time - d.born[i];
      if (age < 0 || age >= d.life[i]) continue;
      d.live++;
      const o = i * 3;
      d.vel[o + 1] -= 13 * dt;
      for (let a = 0; a < 3; a++) d.pos[o + a] += d.vel[o + a] * dt;
      if (d.pos[o + 1] < 0.06 && d.vel[o + 1] < 0) {
        d.pos[o + 1] = 0.06;
        d.vel[o + 1] *= -0.38;
        d.vel[o] *= 0.7;
        d.vel[o + 2] *= 0.7;
        d.spin[i] *= 0.6;
      }
      chunk.position.set(d.pos[o], d.pos[o + 1], d.pos[o + 2]);
      chunk.quaternion.setFromAxisAngle(tmpA.set(d.axis[o], d.axis[o + 1], d.axis[o + 2]), d.spin[i] * age);
      chunk.scale.setScalar(d.size[i] * Math.min(1, (d.life[i] - age) * 4));
      chunk.updateMatrix();
      d.mesh.setMatrixAt(drawn, chunk.matrix);
      d.mesh.setColorAt(drawn, d.color[i]);
      drawn++;
    }
    d.mesh.count = drawn;
    d.mesh.instanceMatrix.needsUpdate = true;
    d.mesh.instanceColor.needsUpdate = true;
  }

  // ---- The field's flat faces ----

  const face = { point: v(), normal: v() };
  // The distance from `p` to the nearest flat face of the field (floor, walls, corner walls, ceiling),
  // with the nearest point on it and its inward normal in `face`. The goal mouths are not walls.
  function nearestFace(p) {
    const sx = p.x < 0 ? -1 : 1, sz = p.z < 0 ? -1 : 1;
    let best = p.y;
    face.normal.set(0, 1, 0);
    const side = FIELD.halfX * S - Math.abs(p.x), top = FIELD.ceiling * S - p.y;
    const endWall = FIELD.halfY * S - Math.abs(p.z);
    // The corner walls cut 1152 uu off each corner of the field.
    const corner = ((FIELD.halfX + FIELD.halfY - 1152) * S - Math.abs(p.x) - Math.abs(p.z)) * Math.SQRT1_2;
    const inMouth = Math.abs(p.x) < FIELD.goalHalfWidth * S && p.y < FIELD.goalHeight * S;
    if (side < best) {
      best = side;
      face.normal.set(-sx, 0, 0);
    }
    if (endWall < best && !inMouth) {
      best = endWall;
      face.normal.set(0, 0, -sz);
    }
    if (corner < best) {
      best = corner;
      face.normal.set(-sx * Math.SQRT1_2, 0, -sz * Math.SQRT1_2);
    }
    if (top < best) {
      best = top;
      face.normal.set(0, -1, 0);
    }
    face.point.copy(p).addScaledVector(face.normal, -best);
    return best;
  }

  // ---- What happens ----

  const tmpA = v(), tmpB = v(), tmpC = v(), nozzle = v(), back = v(), wheel = v(), spot = v();
  const tint = new THREE.Color(), pale = new THREE.Color();

  function spots(car) {
    const state = cars[car.index];
    if (state.body !== car.body) {
      // Where the exhaust and the rear tyres are: the car model says, or the body's hitbox and wheels do.
      const object = NV.cars && NV.cars.objects && NV.cars.objects[car.index];
      const given = (object && object.userData) || {}, body = NV.BODIES[car.body] || NV.BODIES[0];
      const low = -(NV.REST_HEIGHT[car.body] || 17) * S;
      state.body = car.body;
      if (given.nozzle) state.nozzle.copy(given.nozzle);
      else state.nozzle.set((body.offset[0] - body.size[0] / 2) * S, body.offset[2] * S * 0.85, 0);
      for (let side = 0; side < 2; side++) {
        if (given.rearWheels && given.rearWheels[side]) state.wheels[side].copy(given.rearWheels[side]);
        else state.wheels[side].set(body.back.at[0] * S, low, body.back.at[1] * S * (side ? 1 : -1));
      }
    }
    return state;
  }

  function carStep(car, dt, amount) {
    const state = spots(car), teamTint = tints[car.team] || tints[0];
    nozzle.copy(state.nozzle).applyQuaternion(car.quat).add(car.pos);
    back.set(-1, 0, 0).applyQuaternion(car.quat);
    const speed = car.vel.length();
    // A jump in place (a new kick-off, a respawn) leaves no trail.
    if (!state.seen || nozzle.distanceTo(state.last) > 2 + speed * dt * 3) {
      for (let k = 0; k < 3; k++) ribbons.end(1 + car.index * 3 + k);
      state.last.copy(nozzle);
      state.ground = car.onGround;
      state.jumped = car.jumped;
      state.vel.copy(car.vel);
    }
    const firing = car.boosting && !car.demoed;
    state.power = warm && dt === 0 ? +firing : NV.damp(state.power, +firing, firing ? 28 : 12, dt);

    if (state.power > 0.02) {
      const i = flames.geometry.instanceCount++, power = state.power * (car.supersonic ? 1.12 : 1);
      flames.aNozzle.setXYZW(i, nozzle.x, nozzle.y, nozzle.z, power);
      flames.aTurn.setXYZW(i, car.quat.x, car.quat.y, car.quat.z, car.quat.w);
      flames.aTint.setXYZW(i, teamTint.r, teamTint.g, teamTint.b, car.index * 0.37);
      // A soft glow over the nozzle, and the flame's light on the nearest floor or wall.
      const flicker = 0.9 + 0.1 * Math.sin(time * 47 + car.index * 3);
      spot.copy(nozzle).addScaledVector(back, 0.16);
      at(spot);
      going(0, 0, 0);
      lit(tint.copy(teamTint).lerp(WHITE, 0.2), 1.1 * power * flicker);
      P.opacity = 0;
      P.shape = DISC + STEADY;
      P.size0 = 0.22;
      glows.add();
      spot.copy(nozzle).addScaledVector(back, 0.55);
      const gap = nearestFace(spot);
      if (gap < 2.2) {
        const near = 1 - gap / 2.2;
        at(face.point.addScaledVector(face.normal, 0.025));
        going(face.normal.x, face.normal.y, face.normal.z);
        lit(teamTint, 0.6 * power * flicker * near * near);
        P.shape = DISC + FLAT + STEADY;
        P.size0 = 0.9 + gap;
        glows.add();
      }
    }

    if (dt > 0) {
      if (firing) {
        // The trail: glowing puffs dropped along the way the nozzle moved this frame; fewer of them are
        // bigger and brighter.
        const thin = 1 + (1 - amount) * 0.7;
        state.owed += dt * 110 * amount;
        while (state.owed >= 1) {
          state.owed--;
          const k = Math.random();
          spot.lerpVectors(state.last, nozzle, k).addScaledVector(back, rand(0.25, 0.8));
          at(spot);
          scatter(tmpA).multiplyScalar(0.5).addScaledVector(car.vel, 0.3).addScaledVector(back, rand(1.5, 4.5));
          going(tmpA.x, tmpA.y, tmpA.z);
          lit(teamTint, rand(0.25, 0.5) * thin);
          P.opacity = 0;
          P.shape = SMOKE;
          P.life = rand(0.24, 0.42);
          P.size0 = 0.17 * thin;
          P.size1 = rand(0.34, 0.5) * thin;
          P.drag = 2.5;
          P.gravity = 0;
          P.turn = rand(-2, 2);
          P.curve = 1.4;
          particles.add((1 - k) * dt);
        }
        state.haze += dt * 12 * amount;
        while (state.haze >= 1) {
          state.haze--;
          spot.copy(nozzle).addScaledVector(back, rand(0.6, 1.1));
          at(spot);
          scatter(tmpA).multiplyScalar(0.35).addScaledVector(car.vel, 0.12).addScaledVector(back, 1.2);
          going(tmpA.x, tmpA.y + 0.5, tmpA.z);
          lit(WHITE, rand(0.2, 0.3));
          P.opacity = 0.06;
          P.shape = SMOKE;
          P.life = rand(0.7, 1.05);
          P.size0 = 0.2;
          P.size1 = rand(0.8, 1.15);
          P.drag = 2;
          P.gravity = 0;
          P.turn = rand(-1.5, 1.5);
          P.curve = 1.3;
          particles.add();
        }
      }
      if (car.supersonic && !car.demoed) {
        // Thin streaks of air flying by.
        state.streak += dt * 26 * amount;
        while (state.streak >= 1) {
          state.streak--;
          tmpA.set(rand(-0.9, 0.5), rand(0.05, 0.55), rand(-0.75, 0.75)).applyQuaternion(car.quat);
          at(spot.copy(car.pos).add(tmpA));
          going(car.vel.x * 0.45, car.vel.y * 0.45, car.vel.z * 0.45);
          lit(tint.copy(teamTint).lerp(WHITE, 0.7), rand(0.3, 0.7));
          P.opacity = 0;
          P.shape = SPARK;
          P.life = rand(0.12, 0.22);
          P.size0 = P.size1 = 0.014;
          P.drag = 0;
          P.gravity = 0;
          P.turn = 0.045;
          P.curve = 1;
          particles.add();
        }
      }
      if (state.seen) {
        tmpB.set(0, 1, 0).applyQuaternion(car.quat);
        const fall = -state.vel.dot(tmpB);
        if (car.onGround && !state.ground && fall > 2.5) dust(car, tmpB, NV.clamp(fall / 9, 0.3, 1), amount);
        else if (car.jumped && !state.jumped && state.ground) dust(car, tmpB, 0.3, amount);
        if (!car.onGround && !car.demoed && speed > 7) scrape(car, tmpB, speed, dt, amount, state);
      }
    }

    if (state.power > 0.05) ribbons.push(3 + car.index * 3, spot.copy(nozzle).addScaledVector(back, 0.35), state.power);
    else ribbons.end(3 + car.index * 3);
    for (let side = 0; side < 2; side++) {
      const ribbon = 1 + car.index * 3 + side;
      if (car.supersonic && !car.demoed) {
        wheel.copy(state.wheels[side]).applyQuaternion(car.quat).add(car.pos);
        ribbons.push(ribbon, wheel, 1);
      } else ribbons.end(ribbon);
    }
    state.seen = true;
    state.last.copy(nozzle);
    state.ground = car.onGround;
    state.jumped = car.jumped;
    state.vel.copy(car.vel);
  }

  // Dust thrown out sideways under a car that lands or jumps; `up` is the car's up.
  function dust(car, up, strength, amount) {
    const count = Math.round((4 + 8 * strength) * amount);
    spot.copy(car.pos).addScaledVector(up, -0.12);
    for (let i = 0; i < count; i++) {
      scatter(tmpA).addScaledVector(up, -tmpA.dot(up)).normalize();
      at(tmpC.copy(spot).addScaledVector(tmpA, rand(0.3, 0.7)));
      tmpA.multiplyScalar(rand(1.5, 4) * (0.5 + strength)).addScaledVector(up, rand(0.2, 0.9)).addScaledVector(car.vel, 0.25);
      going(tmpA.x, tmpA.y, tmpA.z);
      lit(WHITE, rand(0.3, 0.45));
      P.opacity = 0.16 * (0.5 + strength);
      P.shape = SMOKE;
      P.life = rand(0.45, 0.8);
      P.size0 = 0.18;
      P.size1 = rand(0.6, 1.1) * (0.6 + 0.4 * strength);
      P.drag = 3;
      P.gravity = 0;
      P.turn = rand(-2, 2);
      P.curve = 1.2;
      particles.add();
    }
  }

  // Sparks where a car slides along a face of the field on anything but its wheels.
  function scrape(car, up, speed, dt, amount, state) {
    const gap = nearestFace(car.pos);
    if (gap > 0.55 || up.dot(face.normal) > 0.6) return;
    state.sparks += dt * 70 * amount * NV.clamp(speed / 20, 0.3, 1);
    while (state.sparks >= 1) {
      state.sparks--;
      at(tmpA.copy(face.point).addScaledVector(face.normal, 0.03));
      scatter(tmpA, face.normal).multiplyScalar(rand(1, 4)).addScaledVector(car.vel, rand(0.35, 0.8));
      going(tmpA.x, tmpA.y, tmpA.z);
      lit(FIRE, rand(4, 9));
      P.opacity = 0;
      P.shape = SPARK;
      P.life = rand(0.2, 0.5);
      P.size0 = P.size1 = 0.016;
      P.drag = 1.5;
      P.gravity = 9;
      P.turn = 0.03;
      P.curve = 1;
      particles.add();
    }
  }

  function ballStep(world) {
    const now = world.ball, speed = now.vel.length();
    if (!ball.seen || now.pos.distanceTo(ball.last) > 2 + speed * 0.1) ribbons.end(0);
    const strength = NV.clamp((speed - 7) / 28, 0, 1);
    if (strength > 0) ribbons.push(0, now.pos, strength * (0.35 + 0.65 * strength));
    else ribbons.end(0);
    ball.seen = true;
    ball.last.copy(now.pos);
  }

  // A touch shows a burst only when it gives the car its flip back (a flip reset): the step that ends with the
  // touch carries the mark, a frame or two after the touch itself.
  const FLIP_RESET = 16;
  function flipReset(event) {
    const rec = NV.rec, last = Math.min(rec.count - 1, event.frame + Math.ceil(rec.fps / rec.decisionsPerSecond) + 1);
    for (let f = event.frame; f <= last; f++) {
      if ((((rec.c(f, event.car, rec.L.flags) | 0) >> NV.data.FLAG.marksShift) & FLIP_RESET) !== 0) return true;
    }
    return false;
  }

  function touch(event) {
    const world = NV.world, car = world.cars[event.car], state = cars[event.car];
    if (!car || time - state.touched < 0.12 || !flipReset(event)) return;
    const towardCar = tmpB.copy(event.pos).sub(world.ball.pos);
    if (towardCar.lengthSq() < 1e-6) return;
    towardCar.normalize();
    const hit = tmpA.copy(world.ball.vel).sub(car.vel).length();
    const strength = NV.clamp(Math.max(hit, world.ball.vel.length() * 0.6) / 30, 0, 1);
    if (strength < 0.12) return;
    state.touched = time;
    const amount = NV.q.particles, teamTint = tint.copy(tints[car.team] || WHITE).lerp(WHITE, 0.6);
    spot.copy(world.ball.pos).addScaledVector(towardCar, FIELD.ballRadius * S);
    at(spot);
    going(0, 0, 0);
    lit(teamTint, 1.5 + 5 * strength);
    P.opacity = 0;
    P.shape = DISC;
    P.life = 0.14;
    P.size0 = 0.3 + 0.5 * strength;
    P.size1 = 0.8 + 1.4 * strength;
    P.drag = P.gravity = P.turn = 0;
    P.curve = 1.5;
    particles.add();
    going(towardCar.x, towardCar.y, towardCar.z);
    lit(teamTint, 1.2 + 3.2 * strength);
    P.shape = RING + FLAT;
    P.life = 0.26 + 0.2 * strength;
    P.size0 = 0.35;
    P.size1 = 1 + 2.4 * strength;
    particles.add();
    if (strength > 0.45) {
      for (let i = Math.round(14 * strength * amount); i > 0; i--) {
        scatter(tmpA).addScaledVector(towardCar, -tmpA.dot(towardCar) * 0.85).normalize();
        tmpA.multiplyScalar(rand(4, 13) * strength).addScaledVector(world.ball.vel, 0.3);
        going(tmpA.x, tmpA.y, tmpA.z);
        lit(teamTint, rand(3, 6));
        P.shape = SPARK;
        P.life = rand(0.2, 0.4);
        P.size0 = P.size1 = 0.018;
        P.drag = 2;
        P.gravity = 5;
        P.turn = 0.035;
        P.curve = 1;
        particles.add();
      }
    }
  }

  function demo(event) {
    const car = NV.world.cars[event.car], amount = NV.q.particles, p = event.pos;
    const body = tints[car ? car.team : 0];
    api.shake = Math.max(api.shake, 0.4);
    burst(p, FIRE, 260, 0.55);
    at(p);
    going(0, 0, 0);
    lit(tint.copy(FIRE).lerp(WHITE, 0.6), 9);
    P.opacity = 0;
    P.shape = DISC;
    P.life = 0.2;
    P.size0 = 1.2;
    P.size1 = 5.5;
    P.drag = P.gravity = P.turn = 0;
    P.curve = 1.5;
    particles.add();
    // The fireball, then the smoke it leaves.
    for (let i = Math.round(10 + 14 * amount); i > 0; i--) {
      scatter(tmpA, UP).multiplyScalar(rand(1, 7));
      at(tmpB.copy(p).addScaledVector(tmpA, 0.08));
      going(tmpA.x, tmpA.y + 1.5, tmpA.z);
      lit(FIRE, rand(2, 10));
      P.opacity = 0;
      P.shape = SMOKE;
      P.life = rand(0.25, 0.6);
      P.size0 = 0.4;
      P.size1 = rand(0.9, 2);
      P.drag = 3.2;
      P.turn = rand(-2, 2);
      P.curve = 1.2;
      particles.add();
    }
    for (let i = Math.round(6 + 10 * amount); i > 0; i--) {
      scatter(tmpA, UP).multiplyScalar(rand(1.5, 5));
      at(tmpB.copy(p).addScaledVector(tmpA, 0.15));
      going(tmpA.x, tmpA.y + 2, tmpA.z);
      lit(WHITE, rand(0.07, 0.16));
      P.opacity = rand(0.5, 0.7);
      P.shape = SMOKE;
      P.life = rand(1, 1.35);
      P.size0 = 0.6;
      P.size1 = rand(2, 3);
      P.drag = 2.4;
      P.turn = rand(-1.5, 1.5);
      P.curve = 0.9;
      particles.add(-0.1);
    }
    // A ring of smoke rolling out over the floor.
    const ringCount = Math.round(8 + 12 * amount);
    for (let i = 0; i < ringCount; i++) {
      const angle = ((i + Math.random()) / ringCount) * Math.PI * 2;
      tmpA.set(Math.cos(angle), 0, Math.sin(angle));
      at(tmpB.copy(p).addScaledVector(tmpA, 0.6));
      P.y = Math.max(0.35, p.y);
      going(tmpA.x * 8, 0.4, tmpA.z * 8);
      lit(WHITE, rand(0.2, 0.32));
      P.opacity = 0.4;
      P.shape = SMOKE;
      P.life = rand(0.9, 1.25);
      P.size0 = 0.4;
      P.size1 = rand(1.1, 1.6);
      P.drag = 3.2;
      P.turn = rand(-2, 2);
      P.curve = 1;
      particles.add();
    }
    for (let i = Math.round(60 * amount); i > 0; i--) {
      scatter(tmpA, UP).multiplyScalar(rand(6, 24));
      at(p);
      going(tmpA.x, tmpA.y + 3, tmpA.z);
      lit(tint.copy(FIRE).lerp(WHITE, rand(0, 0.5)), rand(4, 10));
      P.opacity = 0;
      P.shape = SPARK;
      P.life = rand(0.45, 1.15);
      P.size0 = P.size1 = 0.022;
      P.drag = 1.2;
      P.gravity = 13;
      P.turn = 0.028;
      P.curve = 1;
      particles.add();
    }
    if (nearestFace(p) < 1.5) {
      at(face.point.addScaledVector(face.normal, 0.03));
      going(face.normal.x, face.normal.y, face.normal.z);
      lit(tint.copy(FIRE).lerp(WHITE, 0.5), 2);
      P.shape = RING + FLAT;
      P.life = 0.4;
      P.size0 = 0.8;
      P.size1 = 5.5;
      P.drag = P.gravity = P.turn = 0;
      P.curve = 1.3;
      particles.add();
    }
    const d = debris;
    for (let i = Math.round(5 + 7 * amount); i > 0; i--) {
      const n = d.next, o = n * 3;
      d.next = (n + 1) % d.count;
      scatter(tmpA, UP).multiplyScalar(rand(3, 11));
      d.pos[o] = p.x + tmpA.x * 0.04;
      d.pos[o + 1] = Math.max(0.1, p.y + tmpA.y * 0.04);
      d.pos[o + 2] = p.z + tmpA.z * 0.04;
      d.vel[o] = tmpA.x;
      d.vel[o + 1] = tmpA.y + 3;
      d.vel[o + 2] = tmpA.z;
      scatter(tmpA);
      d.axis[o] = tmpA.x;
      d.axis[o + 1] = tmpA.y;
      d.axis[o + 2] = tmpA.z;
      d.spin[n] = rand(6, 22);
      d.born[n] = time;
      d.life[n] = rand(0.9, 1.3);
      d.size[n] = rand(0.6, 1.5);
      if (i % 3) d.color[n].setScalar(rand(0.02, 0.08));
      else d.color[n].copy(body).multiplyScalar(0.35);
    }
    d.live = 1;
  }

  function goal(event) {
    const amount = NV.q.particles, p = event.pos, color = tints[event.team] || tints[0];
    pale.copy(color).lerp(WHITE, 0.5);
    // Into the field, away from the goal the ball is in.
    const out = tmpC.set(0, 0, Math.abs(p.z) > 30 ? -Math.sign(p.z) : 0);
    api.shake = 1;
    burst(p, pale, 1500, 1.5);
    wave(p, color, 5, 34, 1);
    at(p);
    going(0, 0, 0);
    lit(pale, 6);
    P.opacity = 0;
    P.shape = DISC;
    P.life = 0.4;
    P.size0 = 4;
    P.size1 = 26;
    P.drag = P.gravity = P.turn = 0;
    P.curve = 2;
    particles.add();
    // Rings: one racing over the floor, one standing in the goal mouth.
    at(tmpA.set(p.x, 0.04, p.z));
    going(0, 1, 0);
    lit(color, 3.5);
    P.shape = RING + FLAT;
    P.life = 1.4;
    P.size0 = 2;
    P.size1 = 44;
    P.curve = 1.4;
    particles.add();
    if (out.z) {
      at(p);
      going(0, 0, out.z);
      lit(pale, 4);
      P.life = 0.9;
      P.size0 = 1;
      P.size1 = 15;
      particles.add();
    }
    for (let i = Math.round(10 + 14 * amount); i > 0; i--) {
      scatter(tmpA, out.z ? out : null).multiplyScalar(rand(3, 14));
      at(tmpB.copy(p).addScaledVector(tmpA, 0.1));
      going(tmpA.x, tmpA.y, tmpA.z);
      lit(tint.copy(color).lerp(WHITE, rand(0, 0.35)), rand(2, 7));
      P.shape = SMOKE;
      P.life = rand(0.6, 1.3);
      P.size0 = 1.5;
      P.size1 = rand(3.5, 7);
      P.drag = 2;
      P.turn = rand(-1.5, 1.5);
      P.curve = 1.5;
      particles.add();
    }
    for (let i = Math.round(320 * amount); i > 0; i--) {
      scatter(tmpA, out.z ? out : null).multiplyScalar(rand(8, 46));
      at(p);
      going(tmpA.x, tmpA.y + 4, tmpA.z);
      lit(tint.copy(color).lerp(WHITE, Math.random() < 0.3 ? 0.7 : 0.1), rand(5, 14));
      P.shape = DISC;
      P.life = rand(1.2, 2.6);
      P.size0 = rand(0.16, 0.34);
      P.size1 = 0.06;
      P.drag = rand(1.2, 2);
      P.gravity = 4.5;
      P.turn = 0;
      P.curve = 0.6;
      particles.add();
    }
    for (let i = Math.round(90 * amount); i > 0; i--) {
      scatter(tmpA, out.z ? out : null).multiplyScalar(rand(30, 80));
      at(p);
      going(tmpA.x, tmpA.y, tmpA.z);
      lit(pale, rand(5, 10));
      P.shape = SPARK;
      P.life = rand(0.5, 1.2);
      P.size0 = P.size1 = 0.07;
      P.drag = 2.6;
      P.gravity = 3;
      P.turn = 0.04;
      P.curve = 1;
      particles.add();
    }
    // Coloured haze that hangs for the rest of the explosion.
    for (let i = Math.round(10 + 16 * amount); i > 0; i--) {
      scatter(tmpA, out.z ? out : null).multiplyScalar(rand(3, 11));
      at(p);
      going(tmpA.x, Math.abs(tmpA.y) * 0.6, tmpA.z);
      lit(color, rand(0.5, 0.9));
      P.opacity = 0.16;
      P.shape = SMOKE;
      P.life = rand(1.8, 2.6);
      P.size0 = 2;
      P.size1 = rand(5, 8);
      P.drag = 1.5;
      P.gravity = 0;
      P.turn = rand(-1, 1);
      P.curve = 1.2;
      particles.add(-0.1);
    }
    P.opacity = 0;
    // Smaller bursts around it, a little later.
    for (let pop = 0; pop < 5; pop++) {
      const later = rand(0.15, 0.95);
      scatter(tmpA, out.z ? out : null).multiplyScalar(rand(4, 10));
      tmpB.copy(p).add(tmpA);
      tmpB.y = Math.max(1, tmpB.y);
      at(tmpB);
      going(0, 0, 0);
      lit(pale, 5);
      P.shape = DISC;
      P.life = 0.3;
      P.size0 = 0.8;
      P.size1 = 5;
      P.drag = P.gravity = P.turn = 0;
      P.curve = 1.6;
      particles.add(-later);
      for (let i = Math.round(34 * amount); i > 0; i--) {
        scatter(tmpA).multiplyScalar(rand(4, 16));
        going(tmpA.x, tmpA.y, tmpA.z);
        lit(tint.copy(color).lerp(WHITE, rand(0, 0.6)), rand(4, 10));
        P.life = rand(0.6, 1.3);
        P.size0 = rand(0.12, 0.24);
        P.size1 = 0.04;
        P.drag = 2.4;
        P.gravity = 5;
        P.curve = 0.8;
        particles.add(-later);
      }
    }
  }

  // Ends every trail where it is; what is already in the air plays out.
  function cut() {
    for (const state of cars) state.seen = false;
    ball.seen = false;
  }

  function clear() {
    since.value = time;
    ribbons.clear();
    debris.born.fill(-1e9);
    debris.live = debris.mesh.count = 0;
    for (const one of waves) one.mesh.visible = false;
    for (const one of bursts) one.born = -1e9;
    for (const state of cars) state.power = state.owed = state.haze = state.streak = state.sparks = 0;
    api.shake = 0;
    cut();
    warm = true;
  }

  // Starts the effect clock again from zero. Times are kept as 32-bit floats, whose steps grow with the
  // number: a clock left running for hours would move everything in jerks.
  function rebase() {
    const shift = time;
    for (let o = 3; o < particles.data.length; o += STRIDE) particles.data[o] -= shift;
    particles.buffer.clearUpdateRanges();
    particles.buffer.needsUpdate = true;
    for (let o = 3; o < ribbons.data.length; o += 8) ribbons.data[o] -= shift;
    ribbons.changed = true;
    for (let i = 0; i < ribbons.laid.length; i++) ribbons.laid[i] -= shift;
    for (let i = 0; i < debris.count; i++) debris.born[i] -= shift;
    for (const one of [...waves, ...bursts]) one.born -= shift;
    for (const state of cars) state.touched -= shift;
    since.value -= shift;
    time = 0;
  }

  function step(world, dt) {
    time += dt;
    const amount = NV.q.particles;
    for (const car of world.cars) carStep(car, dt, amount);
    ballStep(world);
    if (dt > 0) debrisStep(dt);
  }

  const api = {
    // How hard the camera should shake, 0..1; set by explosions and fading by itself.
    shake: 0,

    build(toScene) {
      scene = toScene;
      noise = noiseTexture();
      const rec = NV.rec, count = rec ? rec.cars : 0;
      for (const team of NV.TEAMS) {
        // The team's colour as a flame burns in it: full strength, its weaker parts lifted (a pure red
        // or blue turns pink under the tone curve when it is bright).
        const color = new THREE.Color(team.color), top = Math.max(color.r, color.g, color.b);
        tints.push(color.setRGB((color.r / top) ** 0.6, (color.g / top) ** 0.6, (color.b / top) ** 0.6));
      }
      for (let i = 0; i < count; i++) {
        cars.push({
          body: -1, nozzle: v(), wheels: [v(), v()], last: v(), vel: v(), seen: false, ground: true, jumped: false,
          power: 0, owed: 0, haze: 0, streak: 0, sparks: 0, touched: -1e9,
        });
      }
      const spriteMaterial = new THREE.ShaderMaterial({
        uniforms: { uTime: clock, uSince: since, uHeight: pictureHeight, tNoise: { value: noise } },
        vertexShader: SPRITE_VERTEX, fragmentShader: SPRITE_FRAGMENT,
        premultipliedAlpha: true, depthWrite: false, transparent: true, side: THREE.DoubleSide,
      });
      particles = new Sprites(3072, spriteMaterial);
      glows = new Sprites(Math.max(4, count * 2), spriteMaterial);
      // Ribbon 0 is the ball's; each car has two for its rear wheels and one for its boost trail.
      ribbons = new Ribbons(1 + count * 3, 40);
      ribbons.style(0, new THREE.Color(0.75, 0.88, 1), 0.55, FIELD.ballRadius * S * 0.8, 0.42, 0.4, 0.02);
      for (let i = 0; i < count; i++) {
        const color = tints[rec.teams[i]] || WHITE;
        ribbons.style(1 + i * 3, tint.copy(color).lerp(WHITE, 0.25), 1.1, 0.045, 0.45, 0.4, 0.04);
        ribbons.style(2 + i * 3, tint, 1.1, 0.045, 0.45, 0.4, 0.04);
        ribbons.style(3 + i * 3, color, 1.1, 0.13, 0.3, 2.2, 0.1);
      }
      flames = buildFlames(Math.max(1, count));
      debris = buildDebris(36);
      buildWaves(2);
      flash = new THREE.PointLight(0xffffff, 0, 120, 1.3);
      for (let i = 0; i < 4; i++) bursts.push({ born: -1e9, life: 1, strength: 0, color: new THREE.Color(), at: v() });
      [glows.mesh, ribbons.mesh, flames.mesh, particles.mesh].forEach((mesh, order) => {
        mesh.renderOrder = 2 + order;
      });
      scene.add(glows.mesh, ribbons.mesh, flames.mesh, particles.mesh, debris.mesh, flash);
      owned.push(noise, spriteMaterial);
      if (rec) history = NV.data.newWorld(rec);
      // The flash light costs every lit surface a little, all the time: the lowest level goes without.
      const afford = () => {
        flash.visible = NV.q.tier !== "low";
      };
      afford();
      const whileBuilt = (react) => (event) => particles && react(event);
      NV.on("quality", afford);
      NV.on("seek", whileBuilt(clear));
      NV.on("restart", whileBuilt(cut));
      NV.on("touch", whileBuilt(touch));
      NV.on("demo", whileBuilt(demo));
      NV.on("goal", whileBuilt(goal));
      lastReal = performance.now();
    },

    update(world, dt) {
      if (!particles) return;
      const now = performance.now(), real = Math.min(0.1, (now - lastReal) / 1000);
      lastReal = now;
      if (time > 600) rebase();
      flames.geometry.instanceCount = 0;
      if (warm && history && world.t > 0) {
        // After a jump in time: the last half second is played through, so the trails are there.
        const steps = 30, each = Math.min(1 / 60, world.t / steps);
        for (let i = steps; i > 0; i--) step(NV.rec.sample(world.t - i * each, history), each);
        flames.geometry.instanceCount = 0;
        glows.head = glows.added = 0;
      }
      step(world, dt);
      warm = false;

      let light = 0;
      for (const one of bursts) {
        const k = (time - one.born) / one.life;
        if (k < 0 || k >= 1) continue;
        const strength = one.strength * Math.min(k * 14, 1) * (1 - k) * (1 - k);
        if (strength > light) {
          light = strength;
          flash.position.copy(one.at);
          flash.color.copy(one.color);
        }
      }
      flash.intensity = light;
      for (const one of waves) {
        if (!one.mesh.visible) continue;
        const k = (time - one.born) / one.life;
        if (k >= 1) {
          one.mesh.visible = false;
          continue;
        }
        one.mesh.scale.setScalar(0.5 + one.reach * (1 - (1 - k) * (1 - k)));
        const fade = one.gain * (1 - k) * (1 - k) * Math.min(k * 20, 1);
        one.mesh.material.uniforms.uLight.value.set(one.color.r, one.color.g, one.color.b, fade);
      }
      api.shake *= Math.exp(-2.4 * (dt > 0 ? dt : real));
      if (api.shake < 0.003) api.shake = 0;

      clock.value = time;
      if (NV.renderer) pictureHeight.value = NV.renderer.domElement.height;
      particles.send();
      glows.sendAll();
      ribbons.send();
      flames.aNozzle.needsUpdate = flames.aTurn.needsUpdate = flames.aTint.needsUpdate = true;
    },

    dispose() {
      for (const one of owned) one.dispose();
      owned.length = 0;
      if (scene && particles) {
        scene.remove(glows.mesh, ribbons.mesh, flames.mesh, particles.mesh, debris.mesh, flash);
        for (const one of waves) scene.remove(one.mesh);
      }
      particles = null;
    },
  };
  return api;
})();
