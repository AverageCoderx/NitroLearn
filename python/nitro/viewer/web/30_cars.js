// The cars: six bodies of our own design, built in code to the hitboxes in NV.BODIES. A body is a loft
// (cross-section rings along its length) with the wheel wells cut in, a glass cabin lofted the same way and
// details placed on its surface. Everything of one body is merged into a few geometries that all cars of
// that body share. Colours come from small palette textures (a vertex's uv picks a cell), so a car draws
// with three materials: clearcoated paint, the other parts, and glass.
//
// Lengths in this file are uu in the car's frame (x forward, y up, z to the right, origin at the car's
// origin) until a geometry is finished, which scales it to metres. Designs give lengths along the body as
// a share of the hitbox length (0 rear face, 1 front face), heights in uu above the floor the car rests
// on, and widths as a share of the hitbox half-width.

NV.cars = (() => {
  const S = NV.S;
  const PAINT = { primary: 0, secondary: 1, accent: 2, gloss: 3 };
  const PART = {
    black: 0, rubber: 1, alloy: 2, dark: 3, chrome: 4, lamp: 5, lens: 6, glow: 7, accent: 8, amber: 9, inner: 10,
    ember: 11,
  };
  const CELLS = { paint: 4, parts: 16, hub: 16, glass: 1, brake: 1 };
  const targets = (set, cells) => Object.fromEntries(Object.keys(cells).map((name) => [name, [set, cells[name]]]));
  // Where a piece of geometry goes: [set, palette cell].
  const paint = targets("paint", PAINT), parts = targets("parts", PART), hub = targets("hub", PART);
  const GLASS = ["glass", 0], BRAKE = ["brake", 0];
  // Lowest the wheels hang below their resting place when they leave the ground (uu).
  const DROOP = 5;
  // How far the middle of a flank stands out from its top and bottom edges (uu).
  const BOW = 1.1;
  // How bright the lamps are (the palette's light colours times this), and the tail lamps: driving, braking.
  const LIGHT = 4, TAIL = [1.3, 5];

  // ---- Geometry kit -----------------------------------------------------------------------------------

  const MIRROR = new THREE.Matrix4().makeScale(1, 1, -1);
  const tp = new THREE.Vector3(), tn = new THREE.Vector3();

  // Collects triangles into named sets, one geometry each, through the transform in force.
  class Mesher {
    constructor() {
      this.sets = {};
      this.matrix = new THREE.Matrix4();
      this.normalMatrix = new THREE.Matrix3();
      this.flip = false;
    }

    set(name) {
      return (this.sets[name] ||= { position: [], normal: [], uv: [], index: [], cells: CELLS[name] });
    }

    within(matrix, fn) {
      const saved = this.matrix, flip = this.flip;
      this.matrix = saved.clone().multiply(matrix);
      this.normalMatrix.getNormalMatrix(this.matrix);
      this.flip = this.matrix.determinant() < 0;
      fn();
      this.matrix = saved;
      this.normalMatrix.getNormalMatrix(saved);
      this.flip = flip;
    }

    // Runs fn twice: as it is, and mirrored to the car's other side.
    both(fn) {
      fn();
      this.within(MIRROR, fn);
    }

    vertex(set, cell, x, y, z, nx, ny, nz) {
      tp.set(x, y, z).applyMatrix4(this.matrix);
      tn.set(nx, ny, nz).applyMatrix3(this.normalMatrix).normalize();
      set.position.push(tp.x, tp.y, tp.z);
      set.normal.push(tn.x, tn.y, tn.z);
      set.uv.push((cell + 0.5) / set.cells, 0.5);
      return set.position.length / 3 - 1;
    }

    triangle(set, a, b, c) {
      if (this.flip) set.index.push(a, c, b);
      else set.index.push(a, b, c);
    }

    // A three.js geometry, as one cell.
    add([name, cell], geometry) {
      const set = this.set(name), base = set.position.length / 3;
      const p = geometry.attributes.position, n = geometry.attributes.normal, index = geometry.index;
      for (let i = 0; i < p.count; i++) {
        this.vertex(set, cell, p.getX(i), p.getY(i), p.getZ(i), n.getX(i), n.getY(i), n.getZ(i));
      }
      const at = (i) => base + (index ? index.getX(i) : i), count = index ? index.count : p.count;
      for (let i = 0; i < count; i += 3) this.triangle(set, at(i), at(i + 1), at(i + 2));
      geometry.dispose();
    }

    // The finished geometry of a set, in metres (null when nothing was added).
    geometry(name) {
      const set = this.sets[name];
      if (!set || !set.index.length) return null;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(set.position.map((v) => v * S), 3));
      geometry.setAttribute("normal", new THREE.Float32BufferAttribute(set.normal, 3));
      geometry.setAttribute("uv", new THREE.Float32BufferAttribute(set.uv, 2));
      geometry.setIndex(set.index);
      geometry.computeBoundingSphere();
      return geometry;
    }
  }

  // A smooth curve through keys [[x, value], ...] (x rising) that never overshoots them; a number is a
  // constant.
  function curve(keys) {
    if (typeof keys === "number") return () => keys;
    const n = keys.length, slope = [], tangent = [];
    for (let i = 0; i < n - 1; i++) slope.push((keys[i + 1][1] - keys[i][1]) / (keys[i + 1][0] - keys[i][0]));
    for (let i = 0; i < n; i++) {
      const a = slope[i - 1], b = slope[i];
      tangent.push(a === undefined ? b : b === undefined ? a : a * b <= 0 ? 0 : (2 * a * b) / (a + b));
    }
    return (x) => {
      if (x <= keys[0][0]) return keys[0][1];
      if (x >= keys[n - 1][0]) return keys[n - 1][1];
      let i = 0;
      while (x > keys[i + 1][0]) i++;
      const h = keys[i + 1][0] - keys[i][0], t = (x - keys[i][0]) / h, t2 = t * t, t3 = t2 * t;
      return (2 * t3 - 3 * t2 + 1) * keys[i][1] + (t3 - 2 * t2 + t) * h * tangent[i]
        + (3 * t2 - 2 * t3) * keys[i + 1][1] + (t3 - t2) * h * tangent[i + 1];
    };
  }

  // Places from `from` to `to` about `step` apart, with the `fixed` ones among them kept exactly.
  function stations(from, to, step, fixed) {
    const keep = fixed.filter((x) => x >= from && x <= to), count = Math.ceil((to - from) / step);
    for (let k = 0; k <= count; k++) {
      const x = from + ((to - from) * k) / count;
      if (!keep.some((other) => Math.abs(other - x) < step * 0.3)) keep.push(x);
    }
    return keep.sort((a, b) => a - b).filter((x, i, all) => !i || x - all[i - 1] > 1e-3);
  }

  const inside = (u, [from, to]) => u >= from && u <= to;

  // A surface through rings of points [x, y, z] (every ring the same count). Normals are smooth except
  // across edges sharper than `crease` (the cosine of the angle) and along the points named in hardRows.
  // cell(i, j) gives the [set, cell] of the quad after ring i and point j, or null to leave it out.
  function loft(mesh, rings, { closed = false, cell, flip, hardRows = null, capStart = null, capEnd = null, crease = 0.45 }) {
    const n = rings.length, m = rings[0].length, cols = closed ? m : m - 1;
    const faces = new Float64Array((n - 1) * cols * 3);
    let outward = 0;
    for (let i = 0; i < n - 1; i++) {
      const A = rings[i], B = rings[i + 1];
      let cx = 0, cy = 0, cz = 0;
      for (const p of A) {
        cx += p[0] / m;
        cy += p[1] / m;
        cz += p[2] / m;
      }
      for (let j = 0; j < cols; j++) {
        const a = A[j], b = B[j], c = B[(j + 1) % m], d = A[(j + 1) % m], k = (i * cols + j) * 3;
        const ux = c[0] - a[0], uy = c[1] - a[1], uz = c[2] - a[2];
        const vx = d[0] - b[0], vy = d[1] - b[1], vz = d[2] - b[2];
        const fx = uy * vz - uz * vy, fy = uz * vx - ux * vz, fz = ux * vy - uy * vx;
        faces.set([fx, fy, fz], k);
        outward += fx * (a[0] + d[0] - 2 * cx) + fy * (a[1] + d[1] - 2 * cy) + fz * (a[2] + d[2] - 2 * cz);
      }
    }
    // Without a word on it, the surface faces away from the middle of its rings.
    const sign = (flip === undefined ? outward < 0 : flip) ? -1 : 1;
    const out = [0, 0, 0];
    // The normal of quad (qi, qj) at its corner (vi, vj): its own, blended with the neighbours it runs
    // smoothly into.
    const corner = (qi, qj, vi, vj) => {
      const own = (qi * cols + qj) * 3, ox = faces[own], oy = faces[own + 1], oz = faces[own + 2];
      const ol = Math.hypot(ox, oy, oz);
      let sx = 0, sy = 0, sz = 0;
      for (let di = -1; di <= 0; di++) {
        for (let dj = -1; dj <= 0; dj++) {
          const i = vi + di;
          let j = vj + dj;
          if (i < 0 || i >= n - 1) continue;
          if (closed) j = (j + cols) % cols;
          else if (j < 0 || j >= cols) continue;
          if (j !== qj && hardRows && hardRows.has(vj % m)) continue;
          const f = (i * cols + j) * 3, fx = faces[f], fy = faces[f + 1], fz = faces[f + 2];
          const fl = Math.hypot(fx, fy, fz);
          if (fl < 1e-9 || fx * ox + fy * oy + fz * oz < crease * fl * ol) continue;
          sx += fx;
          sy += fy;
          sz += fz;
        }
      }
      const l = Math.hypot(sx, sy, sz) * sign || 1;
      out[0] = sx / l;
      out[1] = sy / l;
      out[2] = sz / l;
      return out;
    };
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < cols; j++) {
        const k = (i * cols + j) * 3;
        if (Math.hypot(faces[k], faces[k + 1], faces[k + 2]) < 1e-7) continue;
        const to = cell(i, j);
        if (!to) continue;
        const set = mesh.set(to[0]), j1 = (j + 1) % m, ids = [];
        for (const [vi, vj] of [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]) {
          const normal = corner(i, j, vi, vj), p = rings[vi][vj === j ? j : j1];
          ids.push(mesh.vertex(set, to[1], p[0], p[1], p[2], normal[0], normal[1], normal[2]));
        }
        if (sign > 0) {
          mesh.triangle(set, ids[0], ids[1], ids[2]);
          mesh.triangle(set, ids[0], ids[2], ids[3]);
        } else {
          mesh.triangle(set, ids[0], ids[2], ids[1]);
          mesh.triangle(set, ids[0], ids[3], ids[2]);
        }
      }
    }
    // A flat lid over an end ring, facing away from the ring next to it.
    const cap = (ring, next, to) => {
      let cx = 0, cy = 0, cz = 0, ax = 0, ay = 0, az = 0, nx = 0, ny = 0, nz = 0;
      for (let j = 0; j < m; j++) {
        const p = ring[j], q = ring[(j + 1) % m], r = next[j];
        cx += p[0] / m;
        cy += p[1] / m;
        cz += p[2] / m;
        ax += p[0] - r[0];
        ay += p[1] - r[1];
        az += p[2] - r[2];
        nx += (p[1] - q[1]) * (p[2] + q[2]);
        ny += (p[2] - q[2]) * (p[0] + q[0]);
        nz += (p[0] - q[0]) * (p[1] + q[1]);
      }
      const l = Math.hypot(nx, ny, nz);
      if (l < 1e-7) return;
      const turn = nx * ax + ny * ay + nz * az < 0 ? -1 : 1, set = mesh.set(to[0]);
      nx *= turn / l;
      ny *= turn / l;
      nz *= turn / l;
      const centre = mesh.vertex(set, to[1], cx, cy, cz, nx, ny, nz);
      const ids = ring.map((p) => mesh.vertex(set, to[1], p[0], p[1], p[2], nx, ny, nz));
      for (let j = 0; j < m; j++) {
        if (turn > 0) mesh.triangle(set, centre, ids[j], ids[(j + 1) % m]);
        else mesh.triangle(set, centre, ids[(j + 1) % m], ids[j]);
      }
    };
    if (capStart) cap(rings[0], rings[1], capStart);
    if (capEnd) cap(rings[n - 1], rings[n - 2], capEnd);
  }

  // A surface of revolution around +z from a profile of [radius, along z]. The surface faces the right-hand
  // side of someone walking the profile with the radius to their right and +z ahead.
  function lathe(mesh, to, profile, segments) {
    const rings = profile.map(([r, a]) => {
      const ring = [];
      for (let s = 0; s < segments; s++) {
        const t = (s / segments) * Math.PI * 2;
        ring.push([r * Math.cos(t), r * Math.sin(t), a]);
      }
      return ring;
    });
    loft(mesh, rings, { closed: true, cell: () => to, flip: true });
  }

  // A tube along points; radius is one number or [across, up].
  function tube(mesh, to, points, radius, sides = 6) {
    const t = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3(), up = new THREE.Vector3();
    const ra = radius[0] ?? radius, rb = radius[1] ?? radius;
    const rings = points.map((p, i) => {
      const q0 = points[Math.max(0, i - 1)], q1 = points[Math.min(points.length - 1, i + 1)];
      t.set(q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]).normalize();
      up.set(0, 1, 0);
      if (Math.abs(t.y) > 0.9) up.set(1, 0, 0);
      a.crossVectors(t, up).normalize();
      b.crossVectors(t, a);
      const ring = [];
      for (let s = 0; s < sides; s++) {
        const angle = ((s + 0.5) / sides) * Math.PI * 2, ca = Math.cos(angle) * ra, cb = Math.sin(angle) * rb;
        ring.push([p[0] + a.x * ca + b.x * cb, p[1] + a.y * ca + b.y * cb, p[2] + a.z * ca + b.z * cb]);
      }
      return ring;
    });
    loft(mesh, rings, { closed: true, cell: () => to, capStart: to, capEnd: to });
  }

  // A flat outline of points pushed out to a slab along `depth`.
  function plate(mesh, to, outline, depth) {
    const rings = [-0.5, 0.5].map((k) => outline.map((p) => [p[0] + depth[0] * k, p[1] + depth[1] * k, p[2] + depth[2] * k]));
    loft(mesh, rings, { closed: true, cell: () => to, capStart: to, capEnd: to });
  }

  // A box with rounded edges, its sides along the axes.
  function pod(mesh, to, x, y, z, lx, ly, lz, round) {
    const r = Math.min(round, lx / 2, ly / 2, lz / 2), h = lx / 2;
    const section = (at, inset) => {
      const a = ly / 2 - inset, b = lz / 2 - inset, c = Math.max(0.01, r - inset * 0.6);
      return [[b - c, -a], [b, -a + c], [b, a - c], [b - c, a], [-b + c, a], [-b, a - c], [-b, -a + c], [-b + c, -a]]
        .map(([pz, py]) => [x + at, y + py, z + pz]);
    };
    const rings = [[-h, 0.75], [-h + r * 0.3, 0.3], [-h + r, 0], [h - r, 0], [h - r * 0.3, 0.3], [h, 0.75]]
      .map(([at, inset]) => section(at, r * inset));
    loft(mesh, rings, { closed: true, cell: () => to, capStart: to, capEnd: to, crease: 0.3 });
  }

  // ---- Wheels -----------------------------------------------------------------------------------------

  // rim: the rim's radius as a share of the wheel's; spoke: a spoke's width, likewise; dish: how deep the
  // face sits in the rim (uu); twist: how far the spokes sweep (rad); holes: a plain disc with this many.
  const RIMS = {
    rally: { rim: 0.62, spokes: 6, spoke: 0.2, dish: 2.6, face: "accent", lip: "alloy", cap: "dark" },
    classic: { rim: 0.7, spokes: 5, spoke: 0.22, dish: 1.8, face: "chrome", lip: "chrome", cap: "dark" },
    turbine: { rim: 0.72, spokes: 9, spoke: 0.13, dish: 0.6, twist: 0.35, face: "dark", lip: "dark", cap: "accent" },
    blade: { rim: 0.74, spokes: 7, spoke: 0.13, dish: 1, face: "alloy", lip: "alloy", cap: "accent" },
    sport: { rim: 0.72, spokes: 10, spoke: 0.09, dish: 1.2, face: "alloy", lip: "dark", cap: "dark" },
    steel: { rim: 0.6, holes: 8, dish: 1.8, face: "alloy", lip: "alloy", cap: "chrome" },
  };
  const wheelCache = new Map();

  // A wheel around +z, its outer face toward +z: a tyre with a rounded shoulder and a groove, a rim with a
  // lip and a spoked face over a brake disc, closed at the back.
  function wheelGeometry(r, width, name) {
    const key = `${r}|${width}|${name}`;
    if (wheelCache.has(key)) return wheelCache.get(key);
    const mesh = new Mesher(), style = RIMS[name], h = width / 2, R = r * style.rim, segments = 24;
    const side = [[R - 0.5, 0.8], [NV.lerp(R, r, 0.4), 1], [r - 1.6, 0.94], [r - 0.4, 0.74], [r, 0.5], [r, 0.15], [r - 0.7, 0.1]];
    const tyre = side.map(([pr, a]) => [pr, -a * h]).concat(side.map(([pr, a]) => [pr, a * h]).reverse());
    lathe(mesh, parts.rubber, tyre, segments);
    const face = h * 0.9 - style.dish, back = face - 3.6, boss = r * 0.17;
    lathe(mesh, parts[style.lip], [[R + 0.6, h * 0.7], [R + 0.6, h * 0.93], [R - 0.7, h * 0.93], [R - 1, back]], segments);
    lathe(mesh, parts.inner, [[R - 1, back], [0, back]], segments);
    lathe(mesh, parts.inner, [[0, -h * 0.75], [R - 0.4, -h * 0.8]], segments);
    lathe(mesh, parts.dark, [[R * 0.74, back + 0.3], [R * 0.74, back + 1], [0, back + 1]], 16);
    const cap = [[boss * 1.15, face - 0.2], [boss, face + 0.8], [boss * 0.55, face + 1.1], [0, face + 1.1]];
    lathe(mesh, parts[style.cap], cap, 12);
    // The face: a disc with the windows between the spokes cut out.
    const disc = new THREE.Shape(), ring = (path, points) => {
      points.forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y)));
      path.closePath();
      return path;
    };
    disc.absarc(0, 0, R - 0.9, 0, Math.PI * 2, false);
    if (style.holes) {
      for (let k = 0; k < style.holes; k++) {
        const angle = (k / style.holes) * Math.PI * 2, cx = Math.cos(angle) * R * 0.66, cy = Math.sin(angle) * R * 0.66;
        const points = [];
        for (let s = 0; s < 8; s++) {
          points.push([cx + Math.cos((s * Math.PI) / 4) * R * 0.14, cy + Math.sin((s * Math.PI) / 4) * R * 0.14]);
        }
        disc.holes.push(ring(new THREE.Path(), points));
      }
    } else {
      const count = style.spokes, spoke = style.spoke * r, outer = R - 1.5, twist = style.twist || 0;
      const inner = Math.max(boss * 1.45, (spoke / (2 * Math.sin(Math.PI / count))) * 1.2);
      // Half the angle a window spans at a radius: what the two spokes beside it leave free.
      const reach = (radius) => Math.PI / count - Math.asin(Math.min(1, spoke / (2 * radius)));
      for (let k = 0; k < count; k++) {
        const mid = ((k + 0.5) / count) * Math.PI * 2, points = [];
        for (let s = 0; s <= 5; s++) {
          const angle = mid + twist + reach(outer) * (1 - s / 2.5);
          points.push([Math.cos(angle) * outer, Math.sin(angle) * outer]);
        }
        for (let s = -1; s <= 1; s++) {
          const angle = mid + reach(inner) * s;
          points.push([Math.cos(angle) * inner, Math.sin(angle) * inner]);
        }
        disc.holes.push(ring(new THREE.Path(), points));
      }
    }
    const spokes = new THREE.ExtrudeGeometry(disc, { depth: 1.3, bevelEnabled: false, curveSegments: segments / 2 });
    spokes.translate(0, 0, face - 1.3);
    mesh.add(parts[style.face], spokes);
    const geometry = mesh.geometry("parts");
    wheelCache.set(key, geometry);
    return geometry;
  }

  // ---- The six designs --------------------------------------------------------------------------------
  //
  // span        where the body ends, along the hitbox (rear, front)
  // floor, shoulder, top   the underside, the line where the flank meets the deck, and the deck's middle
  // belt, tuck  half-width at the shoulder, and how much narrower the body is at the sill
  // square      how boxy the deck's cross-section is (2 round, 5 nearly flat)
  // crisp       the shoulder is a sharp edge that overhangs the flank a little
  // flare       how far the flanks swell around the wheel arches (uu), and over what distance
  // clad        how many of the six flank rows from the sill up are cladding instead of paint
  // side(row, u)        another colour for a flank row, or null
  // band, stripes(u, band)   half-widths (uu) of the two stripe bands along the middle of the deck (band 6
  //             the middle one, band 5 beside it), and where they show
  // wheels      rim: the style; width: the tyre's; shift: how far the drawn wheel stands out sideways from
  //             the suspension's anchor (uu); kind: "open" (an arch cut in the flank), "covered" (a well
  //             under a closed flank) or "free" (outside the body)
  // cabin       line: the roof line; base: the height its walls start at (inside the body); width: its
  //             half-width at the base; tumble: how much each side leans in at the top (uu); crown: how far
  //             the middle of the roof stands above its edges; side, top: the stretches that are glass;
  //             pillars: stretches of the side that are not; rail, roof, panel, tail: the colours of the
  //             roof's edge, the roof, a side without glass and a cut-off rear end; bands: a roof stripe
  // detail(k)   everything else, placed with the kit k

  const DESIGNS = [
    // 0: a short, tall rally hatch on high suspension.
    {
      span: [-0.02, 1],
      floor: [[-0.02, 23], [0.1, 12], [0.26, 10], [0.7, 10], [0.9, 11.5], [1, 16.5]],
      shoulder: [[-0.02, 36.5], [0.02, 38], [0.25, 38.5], [0.72, 37], [0.9, 34.5], [0.97, 31.5], [1, 28]],
      top: [[-0.02, 38.5], [0.02, 40.5], [0.2, 41.5], [0.72, 41], [0.9, 38.5], [0.97, 35], [1, 30.5]],
      belt: [
        [-0.02, 0.76], [0.01, 0.83], [0.08, 0.87], [0.3, 0.87], [0.6, 0.86], [0.85, 0.87], [0.95, 0.84], [0.985, 0.78],
        [1, 0.66],
      ],
      tuck: 0.05, square: 3, flare: [3.5, 5], clad: 2, band: [5, 9],
      stripes: (u, band) => (band === 6 && u > 0.7 ? paint.accent : null),
      wheels: { rim: "rally", front: { width: 13, shift: 4 }, back: { width: 14.5, shift: 3 } },
      cabin: {
        line: [[0, 40], [0.13, 56.2], [0.35, 56.6], [0.56, 54.6], [0.73, 39]],
        base: 37.5, width: [[0, 0.72], [0.35, 0.77], [0.73, 0.72]], tumble: 6.5, crown: 1.2,
        side: [0.04, 0.73], top: [[0, 0.125], [0.565, 0.73]], pillars: [[0.33, 0.36], [0.125, 0.16]],
        rail: paint.gloss, roof: paint.accent, panel: paint.primary,
      },
      detail(k) {
        const { X, Y } = k, front = k.wheels[0], flap = front.x - front.arch - 0.7;
        const mudflap = [[4, -6.5], [15, -6.5], [15, 6.5], [4, 6.5]].map(([h, z]) => [flap, Y(h), front.z + z]);
        k.patch(true, parts.black, 21.5, 29.5, -18, 18, 0.5);
        k.patch(true, parts.dark, 23.5, 24.4, -16.5, 16.5, 0.8);
        k.patch(true, parts.dark, 26.5, 27.4, -16.5, 16.5, 0.8);
        k.patch(true, parts.alloy, 15.5, 20.5, -23, 23, 0.7);
        k.patch(false, parts.black, 24, 29.5, -31, 31, 0.9);
        k.both(() => {
          k.headlamp(31, 35.5, 15, 28, 2);
          k.patch(true, parts.amber, 24.5, 26.5, 22, 30, 0.5);
          k.patch(false, parts.black, 31, 38.5, 20, 35, 0.4);
          k.patch(false, BRAKE, 32, 37.5, 22, 34, 0.8);
          k.mirror(0.685, 43.5);
          k.glow(0.3, 0.64);
          k.seam(0.37, 14, 37.4);
          k.seam(0.61, 14, 36.8);
          k.trim(parts.black, 0.28, 0.67, 24.5, 0.9);
          k.pod(parts.accent, X(1) + 0.2, Y(18), 14, 2.6, 2.2, 3, 0.8);
          k.plate(parts.black, mudflap, [0.9, 0, 0]);
          k.tube(parts.black, [[X(0.15), Y(57), 20.5], [X(0.5), Y(56.5), 21]], 0.7, 5);
        });
        k.tube(parts.black, [[X(0.53), Y(55.6), -22], [X(0.53), Y(55.6), 22]], 0.9);
        for (const z of [-16.5, -5.5, 5.5, 16.5]) {
          k.pod(parts.black, X(0.527), Y(56), z, 2.6, 5, 5.6, 1.2);
          k.lamp(parts.lamp, X(0.538), Y(56), z, 2.1, 1);
        }
        k.wing(paint.gloss, { x: X(0.085), y: Y(56.4), span: 26, chord: 8, thick: 1.5, pitch: -0.1 });
        k.nozzle(33, 4.2, 7);
      },
    },
    // 1: a long, low muscle coupe: long bonnet, cabin set back, short tail.
    {
      span: [0, 1],
      floor: [[0, 11.5], [0.07, 8.5], [0.9, 7.5], [1, 9.5]],
      shoulder: [[0, 30.5], [0.02, 32.5], [0.12, 34], [0.25, 34], [0.45, 31.5], [0.8, 30.5], [0.94, 28], [1, 23.5]],
      top: [[0, 33.5], [0.02, 36], [0.05, 36.5], [0.12, 36], [0.3, 36], [0.58, 35.5], [0.85, 33.5], [0.96, 30.5], [1, 25.5]],
      belt: [
        [0, 0.8], [0.015, 0.88], [0.06, 0.94], [0.2, 0.97], [0.45, 0.92], [0.8, 0.95], [0.93, 0.92], [0.985, 0.84],
        [1, 0.72],
      ],
      tuck: 0.04, square: 3.5, flare: [2, 5], clad: 0, band: [3, 9], crisp: true,
      stripes: (u, band) => (band === 5 ? paint.accent : null),
      wheels: { rim: "classic", front: { width: 13, shift: 2 }, back: { width: 14.5, shift: 1.5 } },
      cabin: {
        line: [[0.1, 34], [0.3, 49.1], [0.42, 49.6], [0.48, 48.7], [0.63, 34]],
        base: 32.5, width: [[0.1, 0.6], [0.35, 0.74], [0.63, 0.72]], tumble: 7, crown: 1,
        side: [0.2, 0.63], top: [[0.1, 0.3], [0.485, 0.63]], pillars: [],
        rail: paint.primary, roof: paint.primary, panel: paint.primary, bands: true,
      },
      detail(k) {
        const { X, Y } = k;
        k.patch(true, parts.black, 12.5, 21, -27, 27, 0.5);
        k.patch(true, parts.chrome, 21.3, 22.3, -30, 30, 0.7);
        k.lip(true, parts.black, 11, 8.6, -30, 30, 2.6, 1);
        k.patch(false, parts.black, 25.5, 31.6, -33, 33, 0.4);
        k.patch(false, BRAKE, 26.8, 30.3, -31, 31, 0.8);
        k.patch(false, parts.black, 13.5, 19.5, -30, 30, 0.5);
        k.patch(false, parts.chrome, 20.6, 21.6, -31, 31, 0.7);
        k.both(() => {
          k.headlamp(24.2, 27.4, 14.5, 33.5, 2);
          k.mirror(0.6, 37.5);
          k.glow(0.32, 0.66);
          k.seam(0.37, 10, 33);
          k.seam(0.6, 10, 31);
          k.pod(parts.chrome, X(0.4), Y(28.5), k.skin(X(0.4), Y(28.5)), 3.4, 0.9, 1, 0.3);
          k.tube(parts.chrome, [[X(0.28), Y(9), 38], [X(0.42), Y(9), 39.5]], 1.5, 8);
        });
        k.pod(paint.primary, X(0.74), k.deck(X(0.74), 0) - 0.2, 0, 26, 3.2, 15, 1.6);
        k.pod(parts.black, X(0.835), k.deck(X(0.835), 0) + 0.9, 0, 4, 1.3, 10, 0.5);
        k.wing(paint.primary, { x: X(0.02), y: Y(37.4), span: 33, chord: 7, thick: 1.3, pitch: -0.3 });
        k.nozzle(16.5, 4, 6);
      },
    },
    // 2: a very flat, wide wedge with a canopy and twin tail fins; the big rear wheels run inside the body.
    {
      span: [0, 1],
      floor: [[0, 12.5], [0.08, 8.5], [0.85, 7.5], [1, 9]],
      shoulder: [[0, 33.5], [0.02, 35.5], [0.16, 36], [0.5, 32], [0.81, 28.5], [0.95, 22], [1, 16.5]],
      top: [[0, 37.5], [0.02, 39.5], [0.16, 40.5], [0.5, 36], [0.81, 30.5], [0.95, 24], [1, 18]],
      belt: [[0, 0.88], [0.015, 0.94], [0.1, 0.98], [0.5, 0.97], [0.8, 0.93], [0.93, 0.86], [0.985, 0.78], [1, 0.66]],
      tuck: 0.03, square: 4.5, flare: [1.5, 4], clad: 1, band: [6, 13],
      stripes: (u, band) => (band === 5 && u > 0.25 ? paint.accent : null),
      wheels: { rim: "turbine", front: { width: 13, shift: 3 }, back: { width: 15, shift: 0, kind: "covered" } },
      cabin: {
        line: [[0.22, 37], [0.38, 46.1], [0.5, 46.6], [0.58, 44.2], [0.76, 30]],
        base: 29, width: [[0.22, 0.38], [0.45, 0.56], [0.76, 0.48]], tumble: 7, crown: 1.6,
        side: [0.22, 0.76], top: [[0.22, 0.76]], pillars: [],
        rail: paint.gloss, roof: GLASS, panel: paint.gloss,
      },
      detail(k) {
        const { X, Y } = k;
        const fin = [[0.26, 37.6], [0.08, 46.5], [0.005, 46.5], [0.005, 35]].map(([u, h]) => [X(u), Y(h), 34.5]);
        k.patch(true, parts.lamp, 13.6, 15, -31, 31, 0.7);
        k.patch(true, parts.black, 9.4, 12.4, -24, 24, 0.5);
        k.lip(true, parts.black, 10, 8, -27, 27, 2, 0.9);
        k.patch(false, parts.black, 15, 34.5, -37, 37, 0.5);
        k.patch(false, BRAKE, 31, 32.8, -35, 35, 0.9);
        for (const h of [18.5, 21.5, 24.5, 27.5]) k.patch(false, parts.dark, h, h + 1, -33, 33, 1);
        for (const u of [0.05, 0.09, 0.13, 0.17]) k.pod(parts.black, X(u), k.deck(X(u), 0) + 0.1, 0, 3.4, 1.4, 34, 0.5);
        k.both(() => {
          k.plate(paint.primary, fin, [0, 0, 2.2]);
          k.pod(parts.black, X(0.34), Y(26), k.skin(X(0.34), Y(26)) - 0.4, 17, 6, 3, 1.2);
          k.glow(0.3, 0.68);
          k.seam(0.44, 10, 31.5);
          k.seam(0.66, 10, 28.5);
        });
        k.nozzle(23, 5.6, 6);
      },
    },
    // 3: a long dart: a narrow pointed nose between free-standing front wheels, wide hips, a rear wing.
    {
      span: [0, 1],
      floor: [[0, 12], [0.08, 8], [0.5, 8], [0.8, 12], [1, 15.5]],
      shoulder: [[0, 32], [0.02, 34], [0.14, 35], [0.4, 33], [0.6, 30], [0.8, 26], [1, 18.6]],
      top: [[0, 34.5], [0.02, 36.5], [0.14, 38], [0.45, 37], [0.62, 33.5], [0.8, 28.5], [0.96, 22], [1, 19.4]],
      belt: [
        [0, 0.82], [0.015, 0.9], [0.07, 0.96], [0.2, 0.985], [0.34, 0.95], [0.5, 0.66], [0.64, 0.46], [0.8, 0.36],
        [0.94, 0.25], [1, 0.1],
      ],
      tuck: 0.04, square: 3, flare: [1.5, 4], clad: 0, band: [3.5, 4],
      stripes: (u, band) => (band === 6 ? paint.accent : null),
      wheels: { rim: "blade", front: { width: 12.5, shift: 2, kind: "free" }, back: { width: 15, shift: 0 } },
      cabin: {
        line: [[0.16, 37.5], [0.36, 46], [0.47, 46.4], [0.54, 44.5], [0.7, 31.5]],
        base: 31, width: [[0.16, 0.3], [0.4, 0.5], [0.7, 0.36]], tumble: 6, crown: 1.5,
        side: [0.3, 0.7], top: [[0.47, 0.7]], pillars: [],
        rail: paint.gloss, roof: paint.gloss, panel: paint.gloss,
      },
      detail(k) {
        const { X, Y } = k, front = k.wheels[0], ends = paint.accent;
        const fin = [[0.34, 44], [0.1, 45.6], [0.035, 45.6], [0.05, 36], [0.3, 37.5]].map(([u, h]) => [X(u), Y(h), 0]);
        k.wing(paint.gloss, { x: X(0.975), y: Y(9.6), span: 33, chord: 10, thick: 1.1, pitch: 0.06, plates: 4, ends });
        k.tube(parts.black, [[X(0.965), Y(10), 0], [X(0.955), Y(16.5), 0]], [1.6, 3.2], 6);
        k.wing(paint.gloss, { x: X(0.03), y: Y(45.3), span: 34, chord: 11, thick: 1.4, pitch: -0.16, plates: 6, ends });
        k.plate(paint.primary, fin, [0, 0, 1.4]);
        k.patch(false, parts.black, 13.5, 24.5, -30, 30, 0.5);
        k.both(() => {
          // The wishbones that hold the front wheels.
          for (const [dx, dy] of [[5, 2.5], [-5, 2.5], [4, -3.5], [-4, -3.5]]) {
            const x = front.x + dx, y = front.y + dy, upright = [front.x + dx * 0.3, y, front.z - front.width / 2 - 1];
            k.tube(parts.dark, [[x, y + 1.5, k.skin(x, y + 1.5) - 1], upright], 0.65, 5);
          }
          k.patch(true, parts.lamp, 20.5, 22.5, 3.5, 8, 0.6);
          k.pod(parts.black, X(0.47), Y(21), k.skin(X(0.47), Y(21)) + 0.4, 15, 9, 3, 1.2);
          k.patch(false, parts.black, 26.5, 33, 14, 36, 0.4);
          k.patch(false, BRAKE, 28, 31.5, 16, 34, 0.8);
          k.glow(0.16, 0.42);
          k.seam(0.36, 10, 33);
        });
        k.fender(front);
        k.nozzle(19, 4.8, 7);
      },
    },
    // 4: a cab-forward sports coupe: short nose, raked screen, a long fastback to a cut-off tail.
    {
      span: [0, 1],
      floor: [[0, 13], [0.08, 8.5], [0.9, 8], [1, 10.5]],
      shoulder: [[0, 36], [0.02, 38], [0.13, 39], [0.5, 36], [0.79, 33.5], [0.94, 29], [1, 24]],
      top: [[0, 41.5], [0.02, 43.3], [0.05, 43.6], [0.3, 43], [0.72, 39], [0.9, 34.5], [0.97, 30.5], [1, 26]],
      belt: [
        [0, 0.82], [0.015, 0.9], [0.07, 0.95], [0.16, 0.97], [0.5, 0.9], [0.79, 0.92], [0.93, 0.88], [0.985, 0.8],
        [1, 0.68],
      ],
      tuck: 0.05, square: 3, flare: [3, 5], clad: 1, band: [4, 10],
      stripes: (u, band) => (band === 6 && u > 0.78 ? paint.accent : null),
      wheels: { rim: "sport", front: { width: 13, shift: 4 }, back: { width: 14.5, shift: 4 } },
      cabin: {
        line: [[0.02, 41.5], [0.1, 47.5], [0.38, 55.6], [0.52, 56], [0.6, 54.4], [0.8, 37]],
        base: 35.5, width: [[0.02, 0.58], [0.3, 0.74], [0.6, 0.76], [0.8, 0.7]], tumble: 8.5, crown: 1.4,
        side: [0.26, 0.8], top: [[0.07, 0.375], [0.605, 0.8]], pillars: [[0.44, 0.465]],
        rail: paint.gloss, roof: paint.accent, panel: paint.primary,
      },
      detail(k) {
        const { X, Y } = k;
        k.patch(true, parts.black, 11.5, 19, -24, 24, 0.5);
        k.lip(true, parts.black, 12, 9.4, -28, 28, 2.4, 1);
        k.patch(false, parts.black, 31.5, 37.5, -34, 34, 0.4);
        k.patch(false, BRAKE, 33.6, 35.4, -32, 32, 0.8);
        k.patch(false, parts.black, 14, 23, -24, 24, 0.5);
        k.both(() => {
          k.headlamp(27.8, 30.4, 13, 31, 3);
          k.patch(true, parts.black, 12, 18, 27, 34, 0.5);
          k.mirror(0.75, 41.5);
          k.glow(0.3, 0.62);
          k.seam(0.4, 11, 37);
          k.seam(0.62, 11, 35.5);
          k.pod(parts.black, X(0.3), Y(26), k.skin(X(0.3), Y(26)) + 0.2, 11, 7, 2.4, 1);
        });
        k.wing(paint.gloss, { x: X(0.025), y: Y(45), span: 30, chord: 7, thick: 1.2, pitch: -0.25 });
        k.nozzle(18.5, 4.2, 6);
      },
    },
    // 5: a tall, boxy van: short sloped nose, upright screen, a long flat roof with a rack.
    {
      span: [0, 1],
      floor: [[0, 14], [0.06, 10.5], [0.92, 10.5], [1, 14]],
      shoulder: [[0, 41], [0.02, 42], [0.5, 42], [0.8, 41], [0.93, 37], [1, 31]],
      top: [[0, 43], [0.02, 44], [0.6, 44], [0.82, 43], [0.93, 39.5], [1, 33]],
      belt: [[0, 0.9], [0.012, 0.95], [0.04, 0.975], [0.5, 0.98], [0.9, 0.97], [0.97, 0.93], [1, 0.82]],
      tuck: 0.03, square: 5, flare: [1.6, 4], clad: 1, band: [4, 10], crisp: true,
      stripes: () => null,
      side: (row) => (row === 3 ? paint.accent : null),
      wheels: { rim: "steel", front: { width: 13.5, shift: 3.5 }, back: { width: 14, shift: 1.5 } },
      cabin: {
        line: [[0.004, 59.6], [0.3, 60.2], [0.66, 59.8], [0.72, 58.4], [0.885, 41]],
        base: 40, width: [[0.004, 0.89], [0.04, 0.94], [0.5, 0.95], [0.885, 0.9]], tumble: 3.5, crown: 1,
        side: [0.5, 0.885], top: [[0.725, 0.885]], pillars: [[0.5, 0.52]],
        rail: paint.primary, roof: paint.accent, panel: paint.primary, tail: paint.primary,
      },
      detail(k) {
        const { X, Y } = k, tail = X(0.004), bars = [0.15, 0.37, 0.59];
        k.patch(true, parts.black, 24, 31.5, -17, 17, 0.5);
        k.patch(true, parts.dark, 26, 26.8, -15.5, 15.5, 0.8);
        k.patch(true, parts.dark, 28.6, 29.4, -15.5, 15.5, 0.8);
        k.patch(true, parts.black, 14, 21, -34, 34, 1.2);
        k.patch(false, parts.black, 14.5, 21, -34, 34, 1.2);
        k.tube(parts.black, [[tail - 0.1, Y(22), 0], [tail - 0.1, Y(59), 0]], 0.35, 4);
        k.both(() => {
          k.headlamp(25.5, 31, 20, 31.5, 1);
          k.patch(true, parts.amber, 17, 19, 25, 32, 0.4);
          k.pod(parts.black, tail, Y(49), 31, 1.6, 13, 5.6, 0.6);
          k.pod(BRAKE, tail - 0.5, Y(49), 31, 1.6, 11, 4, 0.6);
          k.pod(GLASS, tail, Y(51), 11.5, 1, 10, 17, 0.4);
          k.pod(parts.chrome, tail - 0.4, Y(39), 3.5, 1, 1.2, 4, 0.3);
          k.tube(parts.dark, [[X(0.12), Y(61.3), 26], [X(0.62), Y(61.3), 26]], 0.8);
          for (const u of bars) k.tube(parts.dark, [[X(u), Y(59.4), 26], [X(u), Y(61.3), 26]], 0.6, 5);
          k.mirror(0.8, 47);
          k.glow(0.3, 0.66);
          k.seam(0.5, 12, 41);
          k.seam(0.74, 12, 40.5);
          k.seam(0.26, 12, 41);
          k.trim(parts.black, 0.27, 0.49, 37, 0.5);
        });
        for (const u of bars) k.tube(parts.dark, [[X(u), Y(61.3), -26], [X(u), Y(61.3), 26]], 0.6, 5);
        k.nozzle(27, 4.4, 6);
      },
    },
  ];

  // ---- Building a body --------------------------------------------------------------------------------

  // The heights of the flank's rows between its foot and the shoulder, and how a cross-section is counted:
  // five steps of underside, the flank's rows, seven steps of deck.
  const SIDE = [0, 0.14, 0.3, 0.48, 0.66, 0.84, 1];
  const UNDER = 5, ROWS = SIDE.length - 1, HALF = UNDER + ROWS + 7;
  const bodyCache = new Map();

  function bodyGeometry(index) {
    if (bodyCache.has(index)) return bodyCache.get(index);
    const B = NV.BODIES[index], D = DESIGNS[index], rest = NV.REST_HEIGHT[index], mesh = new Mesher();
    const length = B.size[0], half = B.size[1] / 2, rear = B.offset[0] - length / 2;
    const X = (u) => rear + u * length, Y = (h) => h - rest, Z = (f) => f * half, U = (x) => (x - rear) / length;
    const along = (keys, map) => curve(typeof keys === "number" ? map(keys) : keys.map(([u, v]) => [X(u), map(v)]));
    const floor = along(D.floor, Y), shoulder = along(D.shoulder, Y), crest = along(D.top, Y);
    const belt = along(D.belt, Z), square = along(D.square, (v) => v), tuck = Z(D.tuck);
    const x0 = X(D.span[0]), x1 = X(D.span[1]);
    const wheels = ["front", "back"].map((name) => {
      const w = B[name], d = D.wheels[name], z = w.at[1] + d.shift;
      // The well leaves room for a front wheel to turn.
      return {
        x: w.at[0], y: w.r - rest, z, r: w.r, width: d.width, arch: w.r + 3, kind: d.kind || "open",
        inner: z - d.width / 2 - (name === "front" ? 5 : 2), outer: z + d.width / 2 + 2,
      };
    });
    const wells = wheels.filter((w) => w.kind !== "free");

    const flare = (x, y) => {
      let most = 0;
      for (const w of wells) {
        if (w.kind !== "open") continue;
        const d = Math.max(0, Math.hypot(x - w.x, Math.max(0, y - w.y)) - w.arch) / D.flare[1];
        most = Math.max(most, Math.exp(-d * d));
      }
      return most * D.flare[0];
    };
    const top = (x) => Math.max(crest(x), shoulder(x) + 0.05);
    // The deck's half-width: the belt, or where the bowed flank ends when the two run into each other.
    const brim = (x) => belt(x) - (D.crisp ? 0 : BOW);
    // The body's half-width at a height.
    const skin = (x, y) => {
      const yF = floor(x), yS = shoulder(x);
      if (y >= yS) {
        const n = square(x), t = Math.min(1, (y - yS) / (top(x) - yS));
        return brim(x) * Math.pow(1 - Math.pow(t, n), 1 / n);
      }
      const u = NV.clamp((y - yF) / (yS - yF), 0, 1);
      return belt(x) - tuck * (1 - u) * (1 - u) - BOW * (2 * u - 1) ** 2 + flare(x, y) * (1 - u * u * u);
    };
    // The deck's height at a distance from the middle.
    const deck = (x, z) => {
      const n = square(x), yS = shoulder(x), g = Math.min(1, Math.abs(z) / brim(x));
      return yS + (top(x) - yS) * Math.pow(1 - Math.pow(g, n), 1 / n);
    };

    // Half a cross-section at x as [y, z], from the middle of the underside up to the middle of the deck:
    // the underside with the wheel well, the flank, then the deck.
    function section(x) {
      const yF = floor(x), yS = shoulder(x), wD = brim(x), edge = skin(x, yF);
      let well = null, yA = yF;
      for (const w of wells) {
        const dx = Math.abs(x - w.x);
        if (dx < w.arch) {
          well = w;
          yA = Math.max(yF, w.y + Math.sqrt(w.arch * w.arch - dx * dx));
        }
      }
      const points = [[yF, 0]];
      let low = yF;
      if (well && yA > yF + 1e-4 && well.kind === "open") {
        low = Math.min(yA, yS - 1);
        const lip = skin(x, low), inner = Math.min(well.inner, lip - 0.5);
        points.push([yF, inner], [low, inner], [low, lip], [low, lip], [low, lip]);
      } else if (well && yA > yF + 1e-4) {
        points.push([yF, well.inner], [yA, well.inner], [yA, well.outer], [yF, well.outer], [yF, edge]);
      } else {
        points.push([yF, edge * 0.45], [yF, edge * 0.45], [yF, edge * 0.8], [yF, edge * 0.8], [yF, edge]);
      }
      for (let k = 1; k <= ROWS; k++) points.push([low + (yS - low) * SIDE[k], skin(x, low + (yS - low) * SIDE[k] - 1e-6)]);
      for (const g of [0.985, 0.93, 0.82, 0.62, Math.min(0.45, D.band[1] / wD), Math.min(0.3, D.band[0] / wD), 0]) {
        points.push([deck(x, g * wD), g * wD]);
      }
      return points;
    }

    // The rings stand evenly, closer at the two ends, and around each wheel arch, with one just outside
    // each end of an arch so that its edge stands upright.
    const fixed = [];
    for (const d of [0.4, 1, 2, 3.4]) fixed.push(x0 + d, x1 - d);
    for (const w of wells) {
      for (let k = 0; k <= 16; k++) fixed.push(w.x + w.arch * Math.cos((Math.PI * k) / 16) * 0.9995);
      fixed.push(w.x - w.arch - 0.03, w.x + w.arch + 0.03);
    }
    const xs = stations(x0, x1, 4.6, fixed);
    const rings = xs.map((x) => {
      const p = section(x), ring = [];
      for (let k = 0; k <= HALF; k++) ring.push([x, p[k][0], p[k][1]]);
      for (let k = HALF - 1; k >= 1; k--) ring.push([x, p[k][0], -p[k][1]]);
      return ring;
    });
    loft(mesh, rings, {
      closed: true,
      hardRows: D.crisp ? new Set([UNDER + ROWS, 2 * HALF - UNDER - ROWS]) : null,
      capStart: paint.primary,
      capEnd: paint.primary,
      cell: (i, j) => {
        const step = j < HALF ? j : 2 * HALF - 1 - j, u = U((xs[i] + xs[i + 1]) / 2), row = step - UNDER;
        if (row < D.clad) return parts.black;
        if (row < ROWS) return (D.side && D.side(row, u)) || paint.primary;
        return D.stripes(u, row - ROWS) || paint.primary;
      },
    });

    // The cabin: an open shell from one side's base over the roof to the other's. Each half of a ring is
    // the side glass, three steps of roof edge and two of roof.
    const C = D.cabin, roofLine = along(C.line, Y), cabinWidth = along(C.width, Z), cabinBase = Y(C.base);
    const c0 = X(C.line[0][0]), c1 = X(C.line[C.line.length - 1][0]), middle = (c0 + c1) / 2;
    const tallest = Math.max(...C.line.map(([, h]) => Y(h))) - cabinBase, cutOff = roofLine(c0) - cabinBase > 1;
    const marks = C.line.map(([u]) => X(u));
    for (const range of [C.side, ...C.top, ...C.pillars]) marks.push(X(range[0]), X(range[1]));
    for (const d of [0.6, 1.8]) marks.push(c0 + d, c1 - d);
    const cs = stations(c0, c1, 4.2, marks);
    const cabinRings = cs.map((x) => {
      const h = Math.max(0, roofLine(x) - cabinBase), t = h / tallest, wb = cabinWidth(x), wt = wb - C.tumble * t;
      const crown = C.crown * t, he = h - crown, c = Math.min(3.6, wt * 0.4) * Math.min(1, t * 1.6);
      const points = [
        [0, wb], [he * 0.8, NV.lerp(wb, wt, 0.8)], [he * 0.93, NV.lerp(wb, wt, 0.93)], [he * 0.99, wt - c * 0.3],
        [he + crown * 0.12, wt - c], [he + crown * 0.78, (wt - c) * 0.5], [h, 0],
      ];
      // The screens bow in plan: the middle of the glass sits ahead of (behind) its corners.
      const bow = ((x - middle) / (c1 - middle)) * (cutOff && x < middle ? 0 : 2.5);
      const at = ([y, z], side) => [x + bow * (1 - (z / wb) ** 2), cabinBase + y, z * side];
      return points.map((p) => at(p, 1)).concat(points.slice(0, 6).reverse().map((p) => at(p, -1)));
    });
    loft(mesh, cabinRings, {
      capStart: cutOff ? C.tail || GLASS : null,
      cell: (i, j) => {
        const step = j < 6 ? j : 11 - j, u = U((cs[i] + cs[i + 1]) / 2);
        if (step === 0) return !inside(u, C.side) ? C.panel : C.pillars.some((range) => inside(u, range)) ? paint.gloss : GLASS;
        if (step < 4) return C.rail;
        if (C.top.some((range) => inside(u, range))) return GLASS;
        return (C.bands && step === 5 && paint.accent) || C.roof;
      },
    });

    // The outline of the body at a height near one end, seen from above: from the middle of the nose (tail)
    // outward and then back along the flank. Gives the place [x, z] a distance along it (to the other side
    // for a negative one).
    function outline(front, y) {
      const start = front ? x1 : x0, dir = front ? -1 : 1, points = [], walked = [0];
      for (let k = 0; k <= 130; k++) {
        const x = start + dir * k * 0.5;
        if (y < floor(x) - 1e-6 || y > top(x) + 1e-6) {
          if (points.length) break;
          continue;
        }
        const z = skin(x, y);
        if (!points.length) for (let q = 0; q < 6; q++) points.push([x, (z * q) / 6]);
        points.push([x, z]);
      }
      if (!points.length) points.push([start, 0], [start, 0]);
      for (let k = 1; k < points.length; k++) {
        walked.push(walked[k - 1] + Math.hypot(points[k][0] - points[k - 1][0], points[k][1] - points[k - 1][1]));
      }
      return (s) => {
        const d = Math.min(Math.abs(s), walked[walked.length - 1] - 1e-3);
        let k = 1;
        while (walked[k] < d) k++;
        const t = (d - walked[k - 1]) / (walked[k] - walked[k - 1] || 1), a = points[k - 1], b = points[k];
        return [NV.lerp(a[0], b[0], t), NV.lerp(a[1], b[1], t) * Math.sign(s || 1)];
      };
    }

    // A thin panel lying on the nose or the tail, between two heights and two distances from the middle
    // line (measured along the surface): lamps, grilles, bumpers.
    function patch(front, to, h0, h1, s0, s1, thick = 0.5) {
      const rows = Math.max(1, Math.ceil((h1 - h0) / 3)), cols = Math.max(2, Math.ceil(Math.abs(s1 - s0) / 2.5));
      const grid = [];
      for (let r = 0; r <= rows; r++) {
        const y = Y(NV.lerp(h0, h1, r / rows)), at = outline(front, y);
        for (let c = 0; c <= cols; c++) {
          const [x, z] = at(NV.lerp(s0, s1, c / cols));
          (grid[c] ||= []).push(new THREE.Vector3(x, y, z));
        }
      }
      const du = new THREE.Vector3(), dv = new THREE.Vector3(), centre = new THREE.Vector3((x0 + x1) / 2, 0, 0);
      const slabs = grid.map((column, c) => {
        const outer = [], inner = [];
        column.forEach((p, r) => {
          du.subVectors(grid[Math.min(cols, c + 1)][r], grid[Math.max(0, c - 1)][r]);
          dv.subVectors(column[Math.min(rows, r + 1)], column[Math.max(0, r - 1)]);
          const normal = du.cross(dv).normalize();
          centre.y = p.y;
          if (normal.dot(dv.subVectors(p, centre)) < 0) normal.negate();
          outer.push([p.x + normal.x * thick, p.y + normal.y * thick, p.z + normal.z * thick]);
          inner.push([p.x - normal.x * thick, p.y - normal.y * thick, p.z - normal.z * thick]);
        });
        return outer.concat(inner.reverse());
      });
      // The back of the panel is inside the body: only its face and its four edges are drawn.
      loft(mesh, slabs, { closed: true, cell: (c, j) => (j > rows && j <= 2 * rows ? null : to), capStart: to, capEnd: to });
    }

    // A flat blade standing out from the foot of the nose or tail (a splitter): it follows the outline at
    // the height `follow` and lies at the height `h`.
    function lip(front, to, follow, h, s0, s1, reach, thick) {
      const at = outline(front, Y(follow)), y = Y(h), cols = Math.max(2, Math.ceil(Math.abs(s1 - s0) / 3));
      const line = [];
      for (let c = 0; c <= cols; c++) line.push(at(NV.lerp(s0, s1, c / cols)));
      const blade = line.map(([x, z], c) => {
        const a = line[Math.max(0, c - 1)], b = line[Math.min(cols, c + 1)];
        const turn = (b[1] - a[1]) * (front ? 1 : -1) < 0 ? -1 : 1, l = (Math.hypot(b[1] - a[1], a[0] - b[0]) || 1) * turn;
        const ox = ((b[1] - a[1]) / l) * reach, oz = ((a[0] - b[0]) / l) * reach, t = thick / 2;
        return [[x + ox, y + t, z + oz], [x + ox, y - t, z + oz], [x - ox, y - t, z - oz], [x - ox, y + t, z - oz]];
      });
      loft(mesh, blade, { closed: true, cell: () => to, capStart: to, capEnd: to });
    }

    // Puts what fn draws around +z at a place, turned to face along the car (1 forward, -1 backward).
    const facing = (x, y, z, way, fn) => {
      const turn = new THREE.Matrix4().makeRotationY((way * Math.PI) / 2);
      mesh.within(new THREE.Matrix4().makeTranslation(x, y, z).multiply(turn), fn);
    };

    const meta = { wheels, nozzle: [x0, 0, 0] };
    // What a design's detail() draws with.
    const kit = {
      X, Y, skin, deck, wheels, patch, lip,
      both: (fn) => mesh.both(fn),
      tube: (to, points, radius, sides) => tube(mesh, to, points, radius, sides),
      plate: (to, shape, depth) => plate(mesh, to, shape, depth),
      pod: (to, ...box) => pod(mesh, to, ...box),
      // A headlamp on the nose: a dark surround, a lens, and bright elements side by side in it.
      headlamp(h0, h1, s0, s1, elements) {
        const step = (s1 - s0) / elements, inset = Math.min(0.9, (h1 - h0) * 0.22);
        patch(true, parts.black, h0 - 1, h1 + 1, s0 - 1.5, s1 + 1.5, 0.4);
        patch(true, parts.lens, h0, h1, s0, s1, 0.65);
        for (let e = 0; e < elements; e++) {
          patch(true, parts.lamp, h0 + inset, h1 - inset, s0 + step * e + inset, s0 + step * (e + 1) - inset, 0.9);
        }
      },
      // A round lamp facing forward.
      lamp(to, x, y, z, radius, depth) {
        const profile = [[radius, -depth], [radius, 0], [radius * 0.6, depth * 0.3], [0, depth * 0.4]];
        facing(x, y, z, 1, () => lathe(mesh, to, profile, 12));
      },
      // A wing across the car: a blade with a rounded front and a thin back edge, with end plates if asked.
      wing(to, { x, y, span, chord, thick, pitch = 0, plates = 0, ends = to }) {
        const shape = [[0.5, 0], [0.28, 0.5], [-0.15, 0.42], [-0.5, 0.07], [-0.5, -0.07], [-0.15, -0.3], [0.28, -0.42]];
        const c = Math.cos(pitch), s = Math.sin(pitch);
        const blade = [-span, span].map((z) => shape.map(([a, b]) => {
          return [x + a * chord * c - b * thick * s, y + a * chord * s + b * thick * c, z];
        }));
        loft(mesh, blade, { closed: true, cell: () => to, capStart: to, capEnd: to, crease: 0.2 });
        if (!plates) return;
        const h = chord * 0.62, low = y - plates * 0.35, high = y + plates * 0.65;
        for (const z of [-span, span]) {
          plate(mesh, ends, [[x + h, low, z], [x + h * 0.6, high, z], [x - h, high, z], [x - h, low, z]], [0, 0, 1]);
        }
      },
      // A door mirror on a short stalk, at the foot of the cabin.
      mirror(u, h) {
        const x = X(u), y = Y(h), z = Math.min(half - 2.6, cabinWidth(x) + 5);
        tube(mesh, parts.black, [[x, y - 1.2, z - 6], [x, y - 0.4, z - 1]], [1.2, 0.6], 5);
        pod(mesh, paint.secondary, x - 0.4, y, z, 3.4, 3.6, 5.4, 1.3);
        pod(mesh, parts.chrome, x - 2, y, z, 0.5, 2.6, 4.2, 0.2);
      },
      // A panel gap down the flank.
      seam(u, h0, h1) {
        const x = X(u), points = [];
        for (let k = 0; k <= 6; k++) {
          const y = Y(NV.lerp(h0, h1, k / 6));
          points.push([x, y, skin(x, y) - 0.12]);
        }
        tube(mesh, parts.black, points, 0.32, 4);
      },
      // A strip along the flank at one height.
      trim(to, u0, u1, h, radius) {
        const points = [], count = Math.ceil(((u1 - u0) * length) / 5);
        for (let k = 0; k <= count; k++) {
          const x = X(NV.lerp(u0, u1, k / count));
          points.push([x, Y(h), skin(x, Y(h))]);
        }
        tube(mesh, to, points, radius, 5);
      },
      // A lit strip under the sill.
      glow(u0, u1) {
        const points = [];
        for (let k = 0; k <= 6; k++) {
          const x = X(NV.lerp(u0, u1, k / 6));
          points.push([x, floor(x) - 0.3, skin(x, floor(x)) - 2.2]);
        }
        tube(mesh, parts.glow, points, 0.7, 5);
      },
      // The boost exhaust: a flared pipe out of the middle of the tail, in a collar.
      nozzle(h, radius, reach) {
        const y = Y(h), x = outline(false, y)(0)[0] - reach * 0.55, r = radius;
        facing(x, y, 0, -1, () => {
          lathe(mesh, parts.dark, [[r * 0.78, -reach], [r * 0.94, -reach * 0.3], [r * 1.08, 0], [r * 0.84, 0]], 14);
          lathe(mesh, parts.inner, [[r * 0.84, 0], [r * 0.55, -reach * 0.8]], 14);
          lathe(mesh, parts.ember, [[r * 0.55, -reach * 0.8], [0, -reach * 0.8]], 14);
          lathe(mesh, parts.black, [[r * 1.5, -reach], [r * 1.5, -reach * 0.42], [r * 0.9, -reach * 0.34]], 14);
        });
        meta.nozzle = [x, y, 0];
      },
      // A mudguard over a free-standing wheel, around the wheel's own middle: it is drawn with the wheel's
      // hub, so it steers and rides with it.
      fender(w) {
        const R = w.r + 2.2, hw = w.width / 2 + 0.8, rings = [];
        const shape = [[-hw, R], [-hw + 1.5, R + 1.1], [hw - 1.5, R + 1.1], [hw, R], [hw, R - 0.7], [-hw, R - 0.7]];
        for (let k = 0; k <= 10; k++) {
          const angle = NV.lerp(-0.35, 2.5, k / 10);
          rings.push(shape.map(([z, r]) => [Math.cos(angle) * r, Math.sin(angle) * r, z]));
        }
        loft(mesh, rings, { closed: true, cell: () => hub.black, capStart: hub.black, capEnd: hub.black });
        tube(mesh, hub.dark, [[0, 0, -w.width / 2], [0, R - 0.4, -w.width / 2 + 1]], 0.8, 5);
      },
    };
    D.detail(kit);

    const built = {
      meta,
      paint: mesh.geometry("paint"),
      parts: mesh.geometry("parts"),
      glass: mesh.geometry("glass"),
      brake: mesh.geometry("brake"),
      hub: mesh.geometry("hub"),
      wheels: wheels.map((w) => wheelGeometry(w.r, w.width, D.wheels.rim)),
      hitbox: new THREE.EdgesGeometry(new THREE.BoxGeometry(B.size[0] * S, B.size[2] * S, B.size[1] * S)),
    };
    bodyCache.set(index, built);
    return built;
  }

  // ---- Materials --------------------------------------------------------------------------------------

  // What sets team-mates apart (stripe, roof, some rims), by a car's place in its team: Blue's, Orange's.
  const ACCENTS = [
    [0xf2f4f7, 0x14161b, 0xffd21f, 0x30e3c4],
    [0xf2f4f7, 0x14161b, 0x35d9ff, 0xc6f432],
  ];
  // The trim colour of each team: a dark shade that sits well beside its paint.
  const TRIM = [0x1f2a44, 0x3a2c25];
  // The parts palette: colour, roughness, metalness, light given off (a colour, times LIGHT). A missing
  // colour is the car's accent; a missing light is the team's colour.
  const PARTS = {
    black: [0x121417, 0.55, 0, 0], rubber: [0x0b0b0c, 0.9, 0, 0], alloy: [0xc9cdd3, 0.3, 0.9, 0],
    dark: [0x33363c, 0.38, 0.85, 0], chrome: [0xf1f1f1, 0.08, 0.95, 0], lamp: [0xffffff, 0.2, 0, 0xfff3dc],
    lens: [0xd8dde4, 0.15, 0, 0x242422], glow: [0x000000, 0.4, 0, null], accent: [null, 0.4, 0.2, 0],
    amber: [0xff8a00, 0.3, 0, 0x7a3600], inner: [0x050506, 0.8, 0, 0], ember: [0x1a0d06, 0.6, 0, 0x2b1204],
  };
  const materials = new Map();
  const bytes = (data, cell, hex) => data.set([(hex >> 16) & 255, (hex >> 8) & 255, hex & 255, 255], cell * 4);

  function paletteTexture(data, colour) {
    const texture = new THREE.DataTexture(data, data.length / 4, 1, THREE.RGBAFormat);
    texture.magFilter = texture.minFilter = THREE.NearestFilter;
    if (colour) texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  function shared(key, make) {
    if (!materials.has(key)) materials.set(key, make());
    return materials.get(key);
  }

  // The materials of a car, by team and place in the team; cars alike share them.
  function materialsFor(team, variant) {
    const side = NV.TEAMS[team], accent = ACCENTS[team][variant % ACCENTS[team].length];
    return {
      paint: shared(`paint ${team} ${variant}`, () => {
        const map = new Uint8Array(CELLS.paint * 4);
        [side.color, TRIM[team], accent, 0x0c0d10].forEach((hex, cell) => bytes(map, cell, hex));
        return new THREE.MeshPhysicalMaterial({
          map: paletteTexture(map, true), metalness: 0.25, roughness: 0.45, clearcoat: 1, clearcoatRoughness: 0.05,
        });
      }),
      parts: shared(`parts ${team} ${variant}`, () => {
        const size = CELLS.parts * 4, map = new Uint8Array(size), surface = new Uint8Array(size), light = new Uint8Array(size);
        for (const [name, [hex, roughness, metalness, glow]] of Object.entries(PARTS)) {
          const cell = PART[name];
          bytes(map, cell, hex ?? accent);
          // Roughness is read from green and metalness from blue.
          surface.set([255, Math.round(roughness * 255), Math.round(metalness * 255), 255], cell * 4);
          bytes(light, cell, glow ?? side.color);
        }
        const finish = shared("finish", () => paletteTexture(surface, false));
        return new THREE.MeshStandardMaterial({
          map: paletteTexture(map, true), roughnessMap: finish, metalnessMap: finish, roughness: 1, metalness: 1,
          emissiveMap: paletteTexture(light, true), emissive: 0xffffff, emissiveIntensity: LIGHT,
        });
      }),
      glass: shared("glass", () => new THREE.MeshPhysicalMaterial({ color: 0x05080c, roughness: 0.06, metalness: 0, ior: 1.9 })),
    };
  }

  const hitboxMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, toneMapped: false });

  // A soft-edged rounded rectangle, for the light and the shadow a car lays on the ground.
  function blobTexture(softness) {
    const size = 64, canvas = document.createElement("canvas"), round = 0.25, core = 1 - softness - round;
    canvas.width = canvas.height = size;
    const context = canvas.getContext("2d"), image = context.createImageData(size, size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const px = Math.abs((x + 0.5) / size - 0.5) * 2, py = Math.abs((y + 0.5) / size - 0.5) * 2;
        const edge = NV.clamp((Math.hypot(Math.max(0, px - core), Math.max(0, py - core)) - round) / softness, 0, 1);
        image.data.set([255, 255, 255, Math.round(255 * (1 - NV.smooth(edge)) ** 1.5)], (y * size + x) * 4);
      }
    }
    context.putImageData(image, 0, 0);
    return new THREE.CanvasTexture(canvas);
  }

  function decal(texture, color, blending) {
    const material = new THREE.MeshBasicMaterial({
      map: texture, color, blending, transparent: true, depthWrite: false, opacity: 0,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    });
    const mesh = new THREE.Mesh(flat, material);
    mesh.renderOrder = 2;
    return mesh;
  }

  // ---- Putting a car together -------------------------------------------------------------------------

  const flat = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
  const cars = [];
  let showBoxes = false;

  // The parts of one body as a group in the car's frame: the chassis (which leans a little), four wheels
  // (an arm that rides and steers, holding the part that spins) and the hitbox.
  function model(body, team, variant) {
    const built = bodyGeometry(body), kinds = materialsFor(team, variant), group = new THREE.Group();
    const chassis = new THREE.Group(), B = NV.BODIES[body];
    for (const name of ["paint", "parts", "glass"]) {
      const part = new THREE.Mesh(built[name], kinds[name]);
      part.castShadow = part.receiveShadow = true;
      chassis.add(part);
    }
    const brake = new THREE.MeshStandardMaterial({ color: 0x300404, roughness: 0.3, emissive: 0xff0b03 });
    brake.emissiveIntensity = TAIL[0];
    chassis.add(new THREE.Mesh(built.brake, brake));
    group.add(chassis);
    const wheels = [];
    for (let k = 0; k < 4; k++) {
      const w = built.meta.wheels[k >> 1], right = k & 1, arm = new THREE.Group(), spin = new THREE.Group();
      const wheel = new THREE.Mesh(built.wheels[k >> 1], kinds.parts);
      wheel.castShadow = wheel.receiveShadow = true;
      if (!right) wheel.rotation.y = Math.PI;
      spin.add(wheel);
      arm.add(spin);
      arm.position.set(w.x * S, w.y * S, (right ? w.z : -w.z) * S);
      if (w.kind === "free") {
        const guard = new THREE.Mesh(built.hub, kinds.parts);
        guard.castShadow = true;
        if (!right) guard.scale.z = -1;
        arm.add(guard);
      }
      group.add(arm);
      wheels.push({ arm, spin, rest: w.y * S, radius: w.r, front: k < 2, drop: 0 });
    }
    const hitbox = new THREE.LineSegments(built.hitbox, hitboxMaterial);
    hitbox.position.set(B.offset[0] * S, B.offset[2] * S, B.offset[1] * S);
    hitbox.visible = showBoxes;
    group.add(hitbox);
    return { group, chassis, wheels, brake, hitbox, built };
  }

  // Gives a car the body it should have, in place of the one it has.
  function dress(state, body) {
    const index = NV.BODIES[body] ? body : 0, object = state.object, data = object.userData;
    if (state.model) {
      object.remove(state.model.group);
      state.model.brake.dispose();
    }
    state.model = model(index, state.team, state.variant);
    state.body = body;
    object.add(state.model.group);
    const B = NV.BODIES[index], meta = state.model.built.meta, back = meta.wheels[1], ground = -NV.REST_HEIGHT[index] * S;
    data.body = index;
    data.size = B.size;
    data.offset = B.offset;
    data.nozzle.set(meta.nozzle[0] * S, meta.nozzle[1] * S, meta.nozzle[2] * S);
    data.rearWheels[0].set(back.x * S, ground, -back.z * S);
    data.rearWheels[1].set(back.x * S, ground, back.z * S);
    state.glow.scale.set(B.size[0] * S * 1.7, 1, B.size[1] * S * 1.9);
    state.shadow.scale.set(B.size[0] * S * 1.25, 1, B.size[1] * S * 1.3);
    state.fresh = true;
  }

  const up = new THREE.Vector3(), forward = new THREE.Vector3(), local = new THREE.Vector3(), turn = new THREE.Quaternion();
  const UP = new THREE.Vector3(0, 1, 0);

  return {
    // One group per car, at the car's origin. userData: nozzle (where the boost leaves, in the group's
    // frame), rearWheels (where the two back tyres meet the ground, left then right), body, size, offset.
    objects: [],

    build(scene, rec) {
      const glowTexture = blobTexture(0.8), shadowTexture = blobTexture(0.45), seen = [0, 0];
      for (let i = 0; i < rec.cars; i++) {
        const team = rec.teams[i] ? 1 : 0, object = new THREE.Group();
        const v = () => new THREE.Vector3();
        object.userData = { nozzle: v(), rearWheels: [v(), v()], body: 0, size: null, offset: null };
        const state = {
          object, team, variant: seen[team]++, body: -1, model: null, fresh: true, vel: v(), pitch: 0, lean: 0, lit: 0,
          glow: decal(glowTexture, NV.TEAMS[team].color, THREE.AdditiveBlending),
          shadow: decal(shadowTexture, 0x000000, THREE.NormalBlending),
        };
        scene.add(object, state.shadow, state.glow);
        dress(state, 0);
        cars.push(state);
        this.objects.push(object);
      }
      // After a jump in time nothing eases from where it was.
      const jump = () => cars.forEach((state) => (state.fresh = true));
      NV.on("seek", jump);
      NV.on("restart", jump);
    },

    // A stand-alone model of a body at rest (for a menu or a preview): a group in the car's frame.
    create(body, team = 0, variant = 0) {
      return model(body, team, variant).group;
    },

    showHitboxes(on) {
      showBoxes = !!on;
      for (const state of cars) state.model.hitbox.visible = showBoxes;
    },

    update(world, time) {
      const dt = Math.max(0, time);
      for (let i = 0; i < cars.length; i++) {
        const car = world.cars[i], state = cars[i], object = state.object;
        if (car.body !== state.body) dress(state, car.body);
        const { wheels, chassis, brake } = state.model, rest = NV.REST_HEIGHT[object.userData.body] * S, snap = state.fresh;
        object.visible = !car.demoed;
        object.position.copy(car.pos);
        object.quaternion.copy(car.quat);
        let contacts = 0;
        for (let k = 0; k < 4; k++) {
          const wheel = wheels[k], drop = car.wheelContact[k] ? 0 : -DROOP * S;
          if (car.wheelContact[k]) contacts++;
          wheel.drop = snap ? drop : NV.damp(wheel.drop, drop, 14, dt);
          wheel.arm.position.y = wheel.rest + wheel.drop;
          wheel.arm.rotation.y = wheel.front ? car.steerAngle : 0;
          wheel.spin.rotation.z = -car.roll / wheel.radius;
        }
        // The chassis leans against its acceleration while the wheels hold the ground.
        if (snap) {
          state.vel.copy(car.vel);
          state.pitch = state.lean = 0;
          state.fresh = false;
        } else if (dt > 0) {
          local.copy(car.vel).sub(state.vel).divideScalar(dt).applyQuaternion(turn.copy(car.quat).invert());
          state.vel.copy(car.vel);
          const grip = contacts >= 3 ? 0.0022 : 0;
          state.pitch = NV.damp(state.pitch, NV.clamp(local.x * grip, -0.05, 0.035), 7, dt);
          state.lean = NV.damp(state.lean, NV.clamp(-local.z * grip, -0.05, 0.05), 7, dt);
        }
        chassis.rotation.set(state.lean, 0, state.pitch);
        const braking = car.throttle < -0.1 && contacts > 0 ? 1 : 0;
        state.lit = snap ? braking : NV.damp(state.lit, braking, 20, dt);
        brake.emissiveIntensity = NV.lerp(TAIL[0], TAIL[1], state.lit);

        // The light and shadow under the car: on the wall or slope the wheels stand on, else on the floor
        // below, fading as the car rises.
        const { glow, shadow } = state;
        let near = 1;
        up.set(0, 1, 0).applyQuaternion(car.quat);
        if (contacts >= 3 && (up.y < 0.97 || car.pos.y > rest + 0.3)) {
          glow.position.copy(car.pos).addScaledVector(up, 0.03 - rest);
          glow.quaternion.copy(car.quat);
        } else {
          near = 1 - NV.smooth(NV.clamp((car.pos.y - rest) / 4, 0, 1));
          forward.set(1, 0, 0).applyQuaternion(car.quat);
          glow.position.set(car.pos.x, 0.012, car.pos.z);
          glow.quaternion.setFromAxisAngle(UP, Math.atan2(-forward.z, forward.x));
        }
        shadow.position.copy(glow.position);
        shadow.quaternion.copy(glow.quaternion);
        glow.visible = shadow.visible = !car.demoed && near > 0.01;
        glow.material.opacity = 0.5 * near;
        shadow.material.opacity = (NV.q.shadows ? 0.35 : 0.7) * near;
      }
    },
  };
})();
