// The arena: its shell (walls, ramps, ceiling, goals) drawn from the collision meshes or generated in the same
// shape, the pitch, the stadium around it, the lighting and the boost pads.
//
// The shell is glass seen from the field's side only, so a camera outside looks through the wall nearest to
// it. Its pieces do not overlap, so one layer of glass shows at every pixel without sorting or depth.

NV.arena = (() => {
  const F = NV.FIELD;
  // Seen from above, the wall line is an octagon with rounded corners (uu, game axes): the diagonal walls lie
  // on |x| + |y| = DIAGONAL and the walls bend into each other with the radius BEND.
  const DIAGONAL = 8064, BEND = 874, RAMP_LOW = 256, RAMP_HIGH = 512;
  const GOAL_RAMP = 93; // the small ramp around a goal's mouth, where the floor meets its sides
  const WALL = 0, CEILING = 1;
  const CREASE = Math.cos((38 * Math.PI) / 180);
  const HEX = 1.0; // width of a wall panel, in metres (rounded so the pattern closes around the arena)
  // The stands begin on this line around the arena (m): a rectangle with rounded corners.
  const STANDS = { halfX: 53, halfZ: 69, bend: 27 };
  // A light bar on the back wall over each goal (m): half width and height. The end's point light sits by it.
  const GOAL_LIGHT = { width: 4.5, height: 8.6, power: 220 };
  const TIER = 0, STRUCTURE = 1, BOARD = 2, FASCIA = 3, ROOF = 4, LAMPS = 5, FACADE = 6;
  const time = { value: 0 };
  const teamColor = (team) => new THREE.Color(NV.TEAMS[team].color);

  // ---------------------------------------------------------------------------------------------- outlines

  // A convex outline with rounded corners: the polygon `points` (counter-clockwise) grown by `radius`.
  // `stations` are points along it with their inward normal and the length along the outline so far.
  function roundedOutline(points, radius, steps) {
    const edges = points.map(([ax, ay], i) => {
      const [bx, by] = points[(i + 1) % points.length];
      const len = Math.hypot(bx - ax, by - ay);
      return { ax, ay, dx: (bx - ax) / len, dy: (by - ay) / len, len, base: 0, turn: 0 };
    });
    const stations = [];
    let total = 0;
    edges.forEach((e, i) => {
      const next = edges[(i + 1) % edges.length];
      e.base = total;
      e.turn = Math.atan2(e.dx * next.dy - e.dy * next.dx, e.dx * next.dx + e.dy * next.dy);
      // The two ends of the straight piece, then the bend around its far end.
      for (let k = -1; k < steps; k++) {
        const angle = k < 0 ? 0 : (e.turn * k) / steps, along = k < 0 ? 0 : e.len;
        const ox = e.dy * Math.cos(angle) + e.dx * Math.sin(angle);
        const oy = e.dy * Math.sin(angle) - e.dx * Math.cos(angle);
        stations.push({
          x: e.ax + e.dx * along + ox * radius, y: e.ay + e.dy * along + oy * radius,
          nx: -ox, ny: -oy, u: total + along + radius * angle,
        });
      }
      total += e.len + radius * e.turn;
    });
    // Where a point lies: `u` along the outline (of the nearest point on it) and `d`, its distance inward.
    function locate(x, y, out) {
      let best = Infinity, inside = true;
      for (const e of edges) {
        const px = x - e.ax, py = y - e.ay;
        const t = NV.clamp(px * e.dx + py * e.dy, 0, e.len);
        const beyond = px * e.dy - py * e.dx > 0;
        if (beyond) inside = false;
        const qx = px - e.dx * t, qy = py - e.dy * t, dist = Math.hypot(qx, qy);
        if (dist < best) {
          best = dist;
          // Past an end of the edge, the nearest point is on the bend there.
          const bend = beyond && dist > 1e-6 ? Math.atan2(e.dy * qy + e.dx * qx, e.dy * qx - e.dx * qy) : 0;
          out.u = e.base + t + radius * bend;
        }
      }
      out.d = inside ? radius + best : radius - best;
      out.u = ((out.u % total) + total) % total;
      return out;
    }
    return { stations, total, locate };
  }

  const fieldLine = (() => {
    const hx = F.halfX - BEND, hy = F.halfY - BEND, c = DIAGONAL - BEND * Math.SQRT2;
    const line = roundedOutline(
      [[hx, hx - c], [hx, c - hx], [c - hy, hy], [hy - c, hy], [-hx, c - hx], [-hx, hx - c], [hy - c, -hy], [c - hy, -hy]],
      BEND, 8,
    );
    // Where the straight walls end and the bends begin.
    line.straightX = c - hy;
    line.straightY = c - hx;
    return line;
  })();

  // ------------------------------------------------------------------------------------------ triangle soup

  // Triangles in game axes and units, each with a kind. A triangle's front is the side (b - a) x (c - a)
  // points to, as in the collision meshes (whose fronts look into the arena); `face` turns a triangle to
  // look along `toward` when that is given.
  function soup() {
    const xyz = [], uv = [], kinds = [];
    function face(a, b, c, kind, toward, ua, ub, uc) {
      if (toward) {
        const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        const along = (uy * vz - uz * vy) * toward[0] + (uz * vx - ux * vz) * toward[1] + (ux * vy - uy * vx) * toward[2];
        if (along < 0) [b, c, ub, uc] = [c, b, uc, ub];
      }
      xyz.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
      kinds.push(kind);
      if (ua) uv.push(ua[0], ua[1], ub[0], ub[1], uc[0], uc[1]);
    }
    function quad(a, b, c, d, kind, toward, ua, ub, uc, ud) {
      face(a, b, c, kind, toward, ua, ub, uc);
      face(a, c, d, kind, toward, ua, uc, ud);
    }
    return { xyz, uv, kinds, face, quad };
  }

  // The part of a polygon (flat x, y, z list) where sign * coordinate[axis] <= limit.
  function clip(poly, axis, sign, limit) {
    const out = [];
    for (let i = 0; i < poly.length; i += 3) {
      const j = (i + 3) % poly.length;
      const a = sign * poly[i + axis] - limit, b = sign * poly[j + axis] - limit;
      if (a <= 0) out.push(poly[i], poly[i + 1], poly[i + 2]);
      if (a < 0 !== b < 0 && a !== 0 && b !== 0) {
        const k = a / (a - b);
        for (let n = 0; n < 3; n++) out.push(poly[i + n] + (poly[j + n] - poly[i + n]) * k);
      }
    }
    return out;
  }

  // A soup as a geometry in scene space with smooth normals: corners at the same place share a normal
  // unless their faces meet at more than the crease angle. With `line`, each corner also gets `aUv`, its
  // place on the wall in panel widths (along the outline, and up the wall's profile), and `aKind`.
  function geometryOf(tris, line) {
    const count = tris.kinds.length, xyz = tris.xyz;
    const faceNormal = new Float32Array(count * 3), weight = new Float32Array(count * 3);
    const ids = new Int32Array(count * 3), fans = [], lookup = new Map();
    for (let c = 0; c < count * 3; c++) {
      const key = `${Math.round(xyz[c * 3] * 2)},${Math.round(xyz[c * 3 + 1] * 2)},${Math.round(xyz[c * 3 + 2] * 2)}`;
      let id = lookup.get(key);
      if (id === undefined) {
        id = fans.length;
        lookup.set(key, id);
        fans.push([]);
      }
      fans[id].push(c);
      ids[c] = id;
    }
    const e = new Float32Array(9);
    for (let f = 0; f < count; f++) {
      const at = f * 9;
      for (let k = 0; k < 3; k++) {
        const from = at + k * 3, to = at + ((k + 1) % 3) * 3;
        for (let n = 0; n < 3; n++) e[k * 3 + n] = xyz[to + n] - xyz[from + n];
      }
      // (b - a) x (c - a), from the edges b - a (0..2) and a - c (6..8).
      const nx = e[2] * e[7] - e[1] * e[8], ny = e[0] * e[8] - e[2] * e[6], nz = e[1] * e[6] - e[0] * e[7];
      const len = Math.hypot(nx, ny, nz) || 1;
      faceNormal.set([nx / len, ny / len, nz / len], f * 3);
      // Each face counts at a corner by its angle there.
      for (let k = 0; k < 3; k++) {
        const p = k * 3, q = ((k + 2) % 3) * 3;
        const dot = -(e[p] * e[q] + e[p + 1] * e[q + 1] + e[p + 2] * e[q + 2]);
        const size = Math.hypot(e[p], e[p + 1], e[p + 2]) * Math.hypot(e[q], e[q + 1], e[q + 2]);
        weight[f * 3 + k] = size > 0 ? Math.acos(NV.clamp(dot / size, -1, 1)) : 0;
      }
    }
    const position = new Float32Array(count * 9), normal = new Float32Array(count * 9);
    const uv = line || tris.uv.length ? new Float32Array(count * 6) : null;
    const kind = uv ? new Float32Array(count * 3) : null;
    const cells = line ? Math.round(line.total / (HEX * 100)) / line.total : 0.01;
    const where = { u: 0, d: 0 }, us = [0, 0, 0];
    for (let f = 0; f < count; f++) {
      for (let k = 0; k < 3; k++) {
        const c = f * 3 + k, x = xyz[c * 3], y = xyz[c * 3 + 1], z = xyz[c * 3 + 2];
        let nx = 0, ny = 0, nz = 0;
        const fx = faceNormal[f * 3], fy = faceNormal[f * 3 + 1], fz = faceNormal[f * 3 + 2];
        for (const other of fans[ids[c]]) {
          const g = Math.floor(other / 3) * 3;
          if (fx * faceNormal[g] + fy * faceNormal[g + 1] + fz * faceNormal[g + 2] > CREASE) {
            nx += faceNormal[g] * weight[other];
            ny += faceNormal[g + 1] * weight[other];
            nz += faceNormal[g + 2] * weight[other];
          }
        }
        // Where a ramp runs into one of the flat planes, it takes the plane's normal.
        if (line) {
          if (Math.abs(Math.abs(x) - F.halfX) < 0.5 && -Math.sign(x) * fx > CREASE) [nx, ny, nz] = [-Math.sign(x), 0, 0];
          else if (z < 0.5 && fz > CREASE) [nx, ny, nz] = [0, 0, 1];
          else if (z > F.ceiling - 0.5 && -fz > CREASE) [nx, ny, nz] = [0, 0, -1];
        }
        const len = Math.hypot(nx, ny, nz) || 1;
        // The swap of y and z mirrors the triangle, so its second and third corners trade places.
        const to = (f * 3 + (k === 0 ? 0 : 3 - k)) * 3;
        position.set([x * NV.S, z * NV.S, y * NV.S], to);
        normal.set([nx / len, nz / len, ny / len], to);
        if (line) {
          line.locate(x, y, where);
          us[k] = where.u;
          uv[(to / 3) * 2 + 1] = (z + (z > F.ceiling / 2 ? where.d : -where.d)) * cells;
        } else if (uv) {
          uv[(to / 3) * 2] = tris.uv[c * 2];
          uv[(to / 3) * 2 + 1] = tris.uv[c * 2 + 1];
        }
        if (kind) kind[to / 3] = tris.kinds[f];
      }
      if (line) {
        // A triangle across the outline's start keeps its corners on one side of it.
        const far = Math.max(us[0], us[1], us[2]) - Math.min(us[0], us[1], us[2]) > line.total / 2;
        for (let k = 0; k < 3; k++) {
          uv[(f * 3 + (k === 0 ? 0 : 3 - k)) * 2] = (us[k] + (far && us[k] < line.total / 2 ? line.total : 0)) * cells;
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
    if (uv) geometry.setAttribute("aUv", new THREE.BufferAttribute(uv, 2));
    if (kind) geometry.setAttribute("aKind", new THREE.BufferAttribute(kind, 1));
    geometry.userData.cell = 0.01 / cells;
    return geometry;
  }

  // ------------------------------------------------------------------------------------------- the shell

  function decode(base64) {
    const text = atob(base64), bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
    const view = new DataView(bytes.buffer), files = [];
    let at = 4;
    for (let n = view.getUint32(0, true); n > 0; n--) {
      const size = view.getUint32(at, true);
      files.push(new DataView(bytes.buffer, at + 4, size));
      at += 4 + size;
    }
    return files;
  }

  // The collision meshes as shell and net triangles. The meshes overlap each other and reach behind the
  // planes the simulation adds (floor, ceiling, side walls), so each is cut to its own region first: a
  // corner to its bend, a goal end and a side to the straight walls between the bends. That leaves one
  // layer of glass everywhere.
  function readMeshes(files, shell, net) {
    const UU = 50;
    for (const file of files) {
      const faces = file.getUint32(0, true), points = file.getUint32(4, true), start = 8 + faces * 12;
      const at = (index, axis) => file.getFloat32(start + (index * 3 + axis) * 4, true) * UU;
      let minX = Infinity, minY = Infinity, maxY = 0, sumX = 0, sumY = 0;
      for (let i = 0; i < points; i++) {
        minX = Math.min(minX, Math.abs(at(i, 0)));
        minY = Math.min(minY, Math.abs(at(i, 1)));
        maxY = Math.max(maxY, Math.abs(at(i, 1)));
        sumX += at(i, 0);
        sumY += at(i, 1);
      }
      const sx = Math.sign(sumX), sy = Math.sign(sumY);
      const cuts = [[0, 1, F.halfX], [0, -1, F.halfX], [2, -1, 0], [2, 1, F.ceiling]];
      if (minX > 2000 && minY > 3000) cuts.push([0, -sx, -fieldLine.straightX], [1, -sy, -fieldLine.straightY]);
      else if (maxY > F.halfY + F.goalDepth / 2) cuts.push([0, 1, fieldLine.straightX], [0, -1, fieldLine.straightX]);
      else cuts.push([1, 1, fieldLine.straightY], [1, -1, fieldLine.straightY]);
      for (let f = 0; f < faces; f++) {
        let poly = [];
        for (let k = 0; k < 3; k++) {
          const index = file.getUint32(8 + (f * 3 + k) * 4, true);
          poly.push(at(index, 0), at(index, 1), at(index, 2));
        }
        const ux = poly[3] - poly[0], uy = poly[4] - poly[1], vx = poly[6] - poly[0], vy = poly[7] - poly[1];
        const uz = poly[5] - poly[2], vz = poly[8] - poly[2];
        const up = (ux * vy - uy * vx) / (Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) || 1);
        for (const [axis, sign, limit] of cuts) if (poly.length >= 9) poly = clip(poly, axis, sign, limit);
        if (poly.length < 9) continue;
        let middle = 0, top = 0;
        for (let i = 0; i < poly.length; i += 3) {
          middle += Math.abs(poly[i + 1]) / (poly.length / 3);
          top = Math.max(top, poly[i + 2]);
        }
        const inGoal = middle > F.halfY + 1;
        // The goal's floor is the pitch running on.
        if (inGoal && up > 0.98 && top < 40) continue;
        const corner = (i) => [poly[i * 3], poly[i * 3 + 1], poly[i * 3 + 2]];
        for (let i = 2; i < poly.length / 3; i++) (inGoal ? net : shell).face(corner(0), corner(i - 1), corner(i), WALL);
      }
    }
    // The side walls between the ramps (the upper ramp's mesh meets the wall a little above its arc's end).
    for (const side of [-1, 1]) {
      const x = side * F.halfX, y = fieldLine.straightY, low = RAMP_LOW, high = F.ceiling - RAMP_HIGH + 4;
      shell.quad([x, -y, low], [x, y, low], [x, y, high], [x, -y, high], WALL, [-side, 0, 0]);
    }
  }

  // A shell of the same shape without the meshes: the wall's profile (ramp, wall, ramp) swept around the
  // wall line, open at the goals, and a box behind each goal.
  function generateShell(shell, net) {
    const profile = [], mouth = [[0, F.goalHeight]], lowSteps = 8, highSteps = 10;
    for (let i = 0; i <= lowSteps; i++) {
      const a = (i / lowSteps) * Math.PI * 0.5;
      profile.push([RAMP_LOW * (1 - Math.sin(a)), RAMP_LOW * (1 - Math.cos(a))]);
    }
    for (let i = 0; i <= highSteps; i++) {
      const a = (i / highSteps) * Math.PI * 0.5;
      const point = [RAMP_HIGH * (1 - Math.cos(a)), F.ceiling - RAMP_HIGH * (1 - Math.sin(a))];
      profile.push(point);
      mouth.push(point);
    }
    const centre = [0, 0, F.ceiling / 2];
    const at = (s, [d, z]) => [s.x + s.nx * d, s.y + s.ny * d, z];
    const toward = (p) => [centre[0] - p[0], centre[1] - p[1], centre[2] - p[2]];
    const sweep = (a, b, shape) => {
      for (let i = 1; i < shape.length; i++) {
        const p = at(a, shape[i - 1]);
        shell.quad(p, at(b, shape[i - 1]), at(b, shape[i]), at(a, shape[i]), WALL, toward(at(a, shape[i])));
      }
    };
    const stations = fieldLine.stations, w = F.goalHalfWidth;
    stations.forEach((a, i) => {
      const b = stations[(i + 1) % stations.length];
      const backWall = Math.abs(a.ny) > 0.999 && Math.abs(b.ny) > 0.999 && Math.abs(a.x - b.x) > 2 * w;
      if (!backWall) return sweep(a, b, profile);
      // Along a back wall: up to the near post, over the goal, on from the far post.
      const dir = Math.sign(b.x - a.x);
      const near = { ...a, x: -dir * w }, far = { ...a, x: dir * w };
      sweep(a, near, profile);
      sweep(near, far, mouth);
      sweep(far, b, profile);
      // The ramp ends at each post in a flat cap.
      for (const post of [near, far]) {
        for (let k = 1; k <= lowSteps; k++) {
          shell.face(at(post, [0, 0]), at(post, profile[k - 1]), at(post, profile[k]), WALL, [-post.x, 0, 0]);
        }
      }
    });
    for (const end of [-1, 1]) {
      const y0 = end * F.halfY, y1 = end * (F.halfY + F.goalDepth), h = F.goalHeight;
      for (const side of [-1, 1]) {
        net.quad([side * w, y0, 0], [side * w, y1, 0], [side * w, y1, h], [side * w, y0, h], WALL, [-side, 0, 0]);
      }
      net.quad([-w, y1, 0], [w, y1, 0], [w, y1, h], [-w, y1, h], WALL, [0, -end, 0]);
      net.quad([-w, y0, h], [w, y0, h], [w, y1, h], [-w, y1, h], WALL, [0, 0, -1]);
    }
  }

  // A flat cover over the wall line, or over what lies `inset` inside it.
  function cover(tris, z, inset, kind, toward) {
    const stations = fieldLine.stations;
    stations.forEach((a, i) => {
      const b = stations[(i + 1) % stations.length];
      const at = (s) => [s.x + s.nx * inset, s.y + s.ny * inset, z];
      tris.face([0, 0, z], at(a), at(b), kind, toward);
    });
  }

  const GLSL_COMMON = /* glsl */ `
    // A number in 0..1 for the whole-numbered cell p lies in.
    float hash21(vec2 p) {
      uvec2 cell = uvec2(ivec2(floor(p)));
      uint h = (cell.x * 0x9E3779B1u) ^ ((cell.y + 0x7F4A7C15u) * 0x85EBCA6Bu);
      h ^= h >> 15;
      h *= 0x2C1B3C6Du;
      h ^= h >> 12;
      h *= 0x297A2D39u;
      h ^= h >> 15;
      return float(h >> 8) * (1.0 / 16777216.0);
    }
    // 1 on a line of half width w around d = 0, with an edge one pixel soft.
    float stroke(float d, float w) {
      float soft = fwidth(d);
      return 1.0 - smoothstep(w - soft, w + soft, abs(d));
    }
    // The hexagon around p (cells one unit wide): x = distance to its edge, yz = its centre.
    vec3 hexCell(vec2 p) {
      const vec2 s = vec2(1.0, 1.7320508);
      vec2 a = mod(p, s) - 0.5 * s, b = mod(p - 0.5 * s, s) - 0.5 * s;
      vec2 g = dot(a, a) < dot(b, b) ? a : b;
      vec2 q = abs(g);
      return vec3(0.5 - max(q.x, dot(q, vec2(0.5, 0.8660254))), p - g);
    }
  `;

  // Glass panels: dark, a little reflective, with a fine hexagon pattern that glows in the team's colour
  // toward each end, solid near the floor and clear toward the ceiling.
  function shellMaterial(cell, pulse) {
    const material = new THREE.MeshStandardMaterial({
      color: 0x0b1018, roughness: 0.2, metalness: 0, transparent: true, depthWrite: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, {
        uBlue: { value: teamColor(0) }, uOrange: { value: teamColor(1) }, uCell: { value: cell }, uTime: time,
        uPulse: pulse,
      });
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", `#include <common>
          attribute vec2 aUv;
          attribute float aKind;
          varying vec2 vPanel;
          varying float vKind;
          varying vec3 vWorld;`)
        .replace("#include <begin_vertex>", `#include <begin_vertex>
          vPanel = aUv;
          vKind = aKind;
          vWorld = (modelMatrix * vec4(position, 1.0)).xyz;`);
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", `#include <common>
          uniform vec3 uBlue, uOrange, uPulse;
          uniform float uCell, uTime;
          varying vec2 vPanel;
          varying float vKind;
          varying vec3 vWorld;
          vec3 wallGlow;
          float wallSheen;
          ${GLSL_COMMON}`)
        .replace("#include <color_fragment>", `
          {
            vec3 hex = hexCell(vPanel);
            float soft = max(fwidth(vPanel.x), fwidth(vPanel.y));
            float seam = (1.0 - smoothstep(0.02, 0.02 + soft * 1.5, hex.x)) * (1.0 - smoothstep(0.2, 0.6, soft));
            float tone = hash21(hex.yz * vec2(2.0, 1.1547005) + 0.5);
            float up = vPanel.y * uCell;
            float trim = stroke(up - ${(RAMP_LOW * NV.S).toFixed(2)}, 0.035)
              + stroke(up - ${((F.ceiling - RAMP_HIGH) * NV.S).toFixed(2)}, 0.035);
            float end = smoothstep(8.0, 50.0, abs(vWorld.z));
            vec3 team = vWorld.z < 0.0 ? uBlue : uOrange;
            vec3 tint = mix(vec3(0.42, 0.6, 0.8), team, 0.25 + 0.75 * end);
            float low = 1.0 - smoothstep(0.0, 5.0, vWorld.y);
            // A slow shimmer wanders over the panels.
            float shimmer = smoothstep(0.75, 1.0, sin(tone * 6.2832 + uTime * 0.6 + vWorld.z * 0.05));
            wallSheen = (1.0 - smoothstep(1.0, 5.0, vWorld.y)) * (1.0 - vKind);
            if (vKind < 0.5) {
              float high = smoothstep(2.0, 16.0, vWorld.y);
              diffuseColor.a = 0.1 + 0.55 * low * low + 0.08 * seam + 0.05 * tone + 0.3 * trim;
              diffuseColor.rgb = mix(diffuseColor.rgb, team * 0.06, end * 0.6);
              wallGlow = tint * (seam * (0.035 + 0.17 * end) * (1.0 - 0.6 * high) + trim * (0.45 + 0.8 * end)
                + (0.008 + 0.025 * shimmer) * (0.3 + end) + 0.05 * end * low)
                + uPulse * (seam * 0.9 + trim + 0.03 + 0.1 * shimmer);
            } else {
              float grid = max(stroke(fract(vWorld.x / 5.12) - 0.5, 0.008), stroke(fract(vWorld.z / 5.12) - 0.5, 0.008));
              diffuseColor.a = 0.04 + 0.05 * grid;
              wallGlow = tint * grid * 0.012;
            }
          }`)
        .replace("#include <emissivemap_fragment>", "totalEmissiveRadiance += wallGlow;")
        // Reflections and glow do not fade with the glass's opacity. The lamps' own highlights stay on the
        // ramps by the floor (higher up they would wander over the glass), and the ceiling hardly reflects.
        .replace("#include <opaque_fragment>", `
          outgoingLight = totalDiffuse * diffuseColor.a + reflectedLight.directSpecular * wallSheen
            + reflectedLight.indirectSpecular * (1.0 - 0.8 * vKind) + totalEmissiveRadiance;
          #include <opaque_fragment>`);
    };
    return material;
  }

  // The wall nearest to an outside camera is seen from its back: only its frame shows, faintly.
  function outsideMaterial(cell) {
    return new THREE.ShaderMaterial({
      uniforms: { uBlue: { value: teamColor(0) }, uOrange: { value: teamColor(1) }, uCell: { value: cell } },
      side: THREE.BackSide, transparent: true, depthWrite: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      vertexShader: /* glsl */ `
        attribute vec2 aUv;
        attribute float aKind;
        varying vec2 vPanel;
        varying float vKind;
        varying vec3 vWorld;
        void main() {
          vPanel = aUv;
          vKind = aKind;
          vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uBlue, uOrange;
        uniform float uCell;
        varying vec2 vPanel;
        varying float vKind;
        varying vec3 vWorld;
        ${GLSL_COMMON}
        void main() {
          float up = vPanel.y * uCell;
          float trim = stroke(up - ${(RAMP_LOW * NV.S).toFixed(2)}, 0.03)
            + stroke(up - ${((F.ceiling - RAMP_HIGH) * NV.S).toFixed(2)}, 0.03);
          float end = smoothstep(8.0, 50.0, abs(vWorld.z));
          vec3 tint = mix(vec3(0.42, 0.6, 0.8), vWorld.z < 0.0 ? uBlue : uOrange, 0.25 + 0.75 * end);
          gl_FragColor = vec4(tint * trim * 0.14 * step(vKind, 0.5), 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
  }

  // The goal's inside: a net of fine diamonds, brighter in the team's colour.
  function netMaterial() {
    return new THREE.ShaderMaterial({
      uniforms: { uBlue: { value: teamColor(0) }, uOrange: { value: teamColor(1) } },
      transparent: true, depthWrite: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      vertexShader: /* glsl */ `
        varying vec3 vWorld;
        varying vec3 vNormal;
        void main() {
          vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
          vNormal = normal;
          gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uBlue, uOrange;
        varying vec3 vWorld;
        varying vec3 vNormal;
        float mesh(vec2 p) {
          vec2 r = vec2(p.x + p.y, p.x - p.y) * 2.6;
          vec2 edge = 0.5 - abs(fract(r) - 0.5), soft = fwidth(r);
          vec2 line = 1.0 - smoothstep(vec2(0.07), vec2(0.07) + soft * 1.5, edge);
          float sharp = max(line.x, line.y);
          return mix(sharp, 0.26, smoothstep(0.25, 0.8, max(soft.x, soft.y)));
        }
        void main() {
          vec3 w = pow(abs(normalize(vNormal)), vec3(6.0));
          w /= w.x + w.y + w.z;
          float knit = mesh(vWorld.zy) * w.x + mesh(vWorld.xz) * w.y + mesh(vWorld.xy) * w.z;
          vec3 team = vWorld.z < 0.0 ? uBlue : uOrange;
          float deep = smoothstep(${(F.halfY * NV.S).toFixed(2)}, ${((F.halfY + F.goalDepth) * NV.S).toFixed(2)}, abs(vWorld.z));
          vec3 color = mix(vec3(0.6), team * 1.4, 0.55) * (0.5 + 0.5 * deep);
          float alpha = knit * 0.45 + 0.05;
          gl_FragColor = vec4(color * knit * 0.45 + team * 0.02, alpha);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
  }

  // The glowing frame around each goal's mouth and a light bar over it, in the team's colour.
  function goalFrames() {
    const position = [], color = [], index = [];
    const box = new THREE.BoxGeometry(1, 1, 1), corner = new THREE.Vector3();
    const bar = (x0, y0, x1, y1, z, depth, tint) => {
      const base = position.length / 3;
      for (let i = 0; i < box.attributes.position.count; i++) {
        corner.fromBufferAttribute(box.attributes.position, i);
        position.push((x0 + x1) / 2 + corner.x * (x1 - x0), (y0 + y1) / 2 + corner.y * (y1 - y0), z + corner.z * depth);
        color.push(tint.r, tint.g, tint.b);
      }
      for (const i of box.index.array) index.push(base + i);
    };
    const w = F.goalHalfWidth * NV.S, h = F.goalHeight * NV.S, t = 0.17;
    NV.TEAMS.forEach((team, i) => {
      const tint = teamColor(i).multiplyScalar(2.8).addScalar(0.12), z = (i ? 1 : -1) * F.halfY * NV.S;
      bar(-w - t, 0, -w, h + t, z, t, tint);
      bar(w, 0, w + t, h + t, z, t, tint);
      bar(-w, h, w, h + t, z, t, tint);
      bar(-GOAL_LIGHT.width, GOAL_LIGHT.height - 0.07, GOAL_LIGHT.width, GOAL_LIGHT.height + 0.07, z, 0.1, tint);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(position, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(color, 3));
    geometry.setIndex(index);
    box.dispose();
    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true }));
  }

  // ------------------------------------------------------------------------------------------- the pitch

  // Turf with mowing stripes, lines drawn from distances (sharp at any range), a tint toward each team's
  // end, dark where it runs under the ramps, and the mirrored scene blended in when there is one.
  function pitchMaterial() {
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0, envMapIntensity: 0.45 });
    material.defines.USE_MIRROR = 0;
    const uniforms = {
      uBlue: { value: teamColor(0) }, uOrange: { value: teamColor(1) },
      uMirror: { value: null }, uMirrorMatrix: { value: new THREE.Matrix4() },
    };
    material.userData.uniforms = uniforms;
    const m = (uu) => (uu * NV.S).toFixed(4);
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vWorld;")
        .replace("#include <begin_vertex>", "#include <begin_vertex>\nvWorld = (modelMatrix * vec4(position, 1.0)).xyz;");
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", `#include <common>
          uniform vec3 uBlue, uOrange;
          #if USE_MIRROR
            uniform sampler2D uMirror;
            uniform mat4 uMirrorMatrix;
          #endif
          varying vec3 vWorld;
          float pitchGloss, pitchPlay;
          vec3 pitchGlow;
          ${GLSL_COMMON}
          float segment(vec2 p, vec2 a, vec2 b) {
            vec2 pa = p - a, ba = b - a;
            return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
          }
          // Distance (m) to the wall line, negative on the field.
          float wallLine(vec2 p) {
            const float bend = ${m(BEND)};
            const vec2 h = vec2(${m(F.halfX - BEND)}, ${m(F.halfY - BEND)});
            const float c = ${m(DIAGONAL - BEND * Math.SQRT2)};
            vec2 q = abs(p), a = vec2(h.x, c - h.x), b = vec2(c - h.y, h.y);
            float inside = max(max(q.x - h.x, q.y - h.y), (q.x + q.y - c) * 0.70710678);
            float outside = min(min(segment(q, vec2(h.x, 0.0), a), segment(q, a, b)), segment(q, b, vec2(0.0, h.y)));
            return (inside > 0.0 ? outside : inside) - bend;
          }
          float grain(vec2 p) {
            vec2 i = floor(p), f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
              mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
          }`)
        .replace("#include <color_fragment>", `
          {
            vec2 p = vWorld.xz, q = abs(p);
            float side = sign(p.y);
            // The flat floor: the field inside the ramps' feet, and the way into each goal (from 5 m before
            // its line to where the net's back begins to rise).
            float foot = wallLine(p) + ${m(RAMP_LOW)};
            float mouth = max(q.x - ${m(F.goalHalfWidth - GOAL_RAMP)}, abs(q.y - ${m(F.halfY + 130)}) - 6.3);
            float flat_ = min(foot, mouth);
            pitchPlay = 1.0 - smoothstep(-0.05, 0.3, flat_);
            float end = smoothstep(0.0, ${m(F.halfY)}, q.y);
            vec3 team = side < 0.0 ? uBlue : uOrange;

            float band = fract(p.y / 12.8);
            float soft = fwidth(p.y / 12.8);
            float stripe = smoothstep(0.0, soft, band) * (1.0 - smoothstep(0.5, 0.5 + soft, band))
              + smoothstep(1.0 - soft, 1.0, band);
            float fine = clamp(1.0 - fwidth(p.x) * 6.0, 0.0, 1.0);
            float blades = mix(0.5, grain(p * 13.0) * 0.5 + grain(p * 47.0) * 0.5, fine);
            vec3 turf = vec3(0.03, 0.135, 0.045) * (0.86 + 0.2 * stripe) * (0.82 + 0.36 * blades);
            turf = mix(turf, turf * (vec3(0.55) + team * 1.5), end * end * 0.75);
            turf *= 0.72 + 0.28 * smoothstep(0.0, 7.0, -flat_);
            float inGoal = smoothstep(${m(F.halfY)} - 0.1, ${m(F.halfY)} + 0.1, q.y);
            turf = mix(turf, turf * 0.35 + team * 0.05, inGoal);

            float lines = stroke(p.y, 0.07);
            lines = max(lines, stroke(length(p) - 9.4, 0.07));
            lines = max(lines, 1.0 - smoothstep(0.0, fwidth(p.x) * 1.5, length(p) - 0.45));
            lines = max(lines, stroke(flat_ + 0.32, 0.07));
            vec2 box = vec2(q.x, q.y - ${m(F.halfY)});
            lines = max(lines, stroke(max(box.x - 12.6, abs(box.y) - 7.2), 0.06));
            lines = max(lines, stroke(max(box.x - 19.5, abs(box.y) - 15.0), 0.06));
            float goalLine = stroke(box.y, 0.09) * step(q.x, ${m(F.goalHalfWidth - GOAL_RAMP)});
            lines = max(lines, goalLine) * pitchPlay * (1.0 - inGoal * (1.0 - goalLine));

            vec3 under = vec3(0.006, 0.008, 0.011);
            diffuseColor.rgb = mix(under, mix(turf, vec3(0.8), lines * 0.92), pitchPlay);
            pitchGloss = mix(0.3, mix(0.66 - 0.1 * stripe, 0.45, lines), pitchPlay);
            pitchGlow = (vec3(0.06) * lines + team * goalLine * 0.9) * pitchPlay;
          }`)
        .replace("#include <roughnessmap_fragment>", "float roughnessFactor = pitchGloss;")
        .replace("#include <emissivemap_fragment>", "totalEmissiveRadiance += pitchGlow;")
        .replace("#include <opaque_fragment>", `
          // Turf has a sheen, not a shine.
          outgoingLight -= reflectedLight.directSpecular * 0.5 * pitchPlay;
          #if USE_MIRROR
            vec4 at = uMirrorMatrix * vec4(vWorld, 1.0);
            vec2 uv = at.xy / at.w;
            // Read from the smaller copies of the mirrored scene: soft, as a reflection in turf is.
            vec3 seen = min(textureLod(uMirror, uv, 1.5).rgb, vec3(1.6)) * 0.55
              + min(textureLod(uMirror, uv, 3.0).rgb, vec3(1.6)) * 0.45;
            float facing = saturate(dot(normal, normalize(vViewPosition)));
            outgoingLight += seen * (0.035 + 0.3 * pow(max(1.0 - facing, 0.0), 3.0)) * pitchPlay;
          #endif
          #include <opaque_fragment>`);
    };
    return material;
  }

  function pitchGeometry() {
    const tris = soup(), up = [0, 0, 1], w = F.goalHalfWidth + 12;
    cover(tris, 0, 0, WALL, up);
    for (const end of [-1, 1]) {
      const y0 = end * (F.halfY - 2), y1 = end * (F.halfY + F.goalDepth);
      tris.quad([-w, y0, 0], [w, y0, 0], [w, y1, 0], [-w, y1, 0], WALL, up);
    }
    return geometryOf(tris, null);
  }

  // ----------------------------------------------------------------------------------------- the stadium

  // The stands: one profile (boards, two tiers of seats, a roof with the floodlights at its edge) swept
  // around the arena. Only the side facing the field is drawn, so a camera among the seats sees through.
  function standsGeometry() {
    const S = STANDS, tris = soup();
    const line = roundedOutline(
      [[S.halfX - S.bend, S.bend - S.halfZ], [S.halfX - S.bend, S.halfZ - S.bend],
       [S.bend - S.halfX, S.halfZ - S.bend], [S.bend - S.halfX, S.bend - S.halfZ]],
      S.bend, 12,
    );
    // Outward distance and height (m) of each stretch of the profile, and what it is.
    const profile = [
      [0, 0, 0, 1.2, BOARD], [0, 1.2, 1.6, 1.2, STRUCTURE], [1.6, 1.2, 21, 11.6, TIER], [21, 11.6, 21, 14.4, FASCIA],
      [21, 14.4, 44, 31, TIER], [44, 31, 44, 39, STRUCTURE], [44, 39, 25, 36, ROOF], [25, 36, 24.2, 37.6, LAMPS],
      [24.2, 37.6, 45, 40.2, STRUCTURE], [45, 40.2, 47, 0, FACADE],
    ];
    // Seats are counted along the outline, so their number closes around the stadium; the second
    // coordinate is the distance (m) up each stretch of the profile.
    const seats = Math.round(line.total / 0.62) / line.total;
    const stations = line.stations;
    stations.forEach((a, i) => {
      const b = stations[(i + 1) % stations.length], ua = a.u * seats, ub = (i + 1 < stations.length ? b.u : line.total) * seats;
      for (const [s0, h0, s1, h1, kind] of profile) {
        const length = Math.hypot(s1 - s0, h1 - h0);
        const at = (p, s, h) => [(p.x - p.nx * s) * 100, (p.y - p.ny * s) * 100, h * 100];
        const toward = [-a.nx * (h0 - h1), -a.ny * (h0 - h1), s1 - s0];
        const corners = [at(a, s0, h0), at(b, s0, h0), at(b, s1, h1), at(a, s1, h1)];
        tris.quad(...corners, kind, toward, [ua, 0], [ub, 0], [ub, length], [ua, length]);
      }
    });
    return geometryOf(tris, null);
  }

  function standsMaterial(uniforms) {
    return new THREE.ShaderMaterial({
      uniforms,
      vertexShader: /* glsl */ `
        attribute vec2 aUv;
        attribute float aKind;
        varying vec2 vSeat;
        varying float vKind;
        varying vec3 vWorld;
        void main() {
          vSeat = aUv;
          vKind = aKind;
          vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uBlue, uOrange;
        uniform float uTime, uCrowd, uCheer;
        varying vec2 vSeat;
        varying float vKind;
        varying vec3 vWorld;
        ${GLSL_COMMON}
        void main() {
          float end = smoothstep(-30.0, 30.0, vWorld.z);
          vec3 team = mix(uBlue, uOrange, end);
          vec3 color = vec3(0.004, 0.005, 0.007);
          if (vKind < ${TIER}.5) {
            // One seat per cell; most hold someone in a shirt of their own, many in the colour of their end.
            vec2 cell = vec2(vSeat.x, vSeat.y / 0.85);
            vec2 id = floor(cell), f = fract(cell);
            float a = hash21(id), b = hash21(id + 31.0), c = hash21(id + 77.0);
            float soft = max(fwidth(cell.x), fwidth(cell.y));
            float distant = smoothstep(0.35, 1.1, soft);
            float sway = 0.04 * sin(uTime * (1.5 + 2.5 * b) + a * 40.0);
            float figure = 1.0 - smoothstep(0.3, 0.42, length((f - vec2(0.5, 0.52 + sway)) * vec2(1.0, 0.8)));
            float taken = step(c, 0.9) * uCrowd;
            vec3 shirt = mix(vec3(0.14 + 0.5 * b, 0.14 + 0.45 * a, 0.16 + 0.4 * c), team * 0.9 + 0.03, step(0.5, a));
            float lit = (0.1 + 0.16 * smoothstep(34.0, 0.0, vWorld.y)) * (1.0 + 0.6 * uCheer)
              * (0.75 + 0.25 * sin(uTime * (1.0 + 2.0 * c + 6.0 * uCheer) + b * 50.0));
            vec3 seat = vec3(0.012, 0.014, 0.02) * (0.5 + 0.9 * smoothstep(0.1, 0.3, f.y));
            vec3 close = mix(seat, shirt * lit, figure * taken);
            vec3 average = mix(seat, mix(vec3(0.3), team * 0.9, 0.5) * lit, 0.45 * uCrowd);
            color = mix(close, average, distant);
            // Now and then a camera flashes; many do after a goal.
            float flash = step(0.99965 - 0.004 * uCheer, hash21(id + floor(uTime * 7.0) * vec2(13.0, 7.0))) * taken;
            color += vec3(5.0, 5.2, 6.0) * flash * mix(figure, 0.2, distant);
          } else if (vKind < ${STRUCTURE}.5) {
            color = vec3(0.006, 0.007, 0.01);
          } else if (vKind < ${BOARD}.5) {
            // Light boards along the front row: blocks of the end's colour with a slow sweep.
            float block = floor(vSeat.x / 14.0);
            float sweep = 0.5 + 0.5 * sin(vSeat.x * 0.35 - uTime * 1.2 + block);
            float inside = stroke(fract(vSeat.x / 14.0) - 0.5, 0.47) * stroke(vSeat.y - 0.6, 0.45);
            color += (team * (0.25 + 0.3 * sweep) + 0.02) * inside * (0.6 + 0.4 * hash21(vec2(block, 3.0)));
          } else if (vKind < ${FASCIA}.5) {
            float h = vWorld.y;
            color += (team * 1.6 + 0.1) * stroke(h - 13.0, 0.09) + vec3(0.02, 0.022, 0.03) * stroke(h - 13.0, 1.3);
          } else if (vKind < ${ROOF}.5) {
            float truss = max(stroke(fract(vSeat.x / 9.0) - 0.5, 0.012), stroke(fract(vSeat.y / 4.0) - 0.5, 0.03));
            color = vec3(0.012, 0.013, 0.017) + vec3(0.03, 0.032, 0.04) * truss;
          } else if (vKind < ${LAMPS}.5) {
            // Floodlights: banks of lamps along the roof's edge.
            vec2 lamp = vec2(fract(vSeat.x / 3.4), fract(vSeat.y / 0.895));
            float bank = step(0.5, fract(vSeat.x / 27.2 + 0.25));
            float glass = stroke(lamp.x - 0.5, 0.36) * stroke(lamp.y - 0.5, 0.36);
            color = mix(vec3(0.01), vec3(11.0, 11.5, 12.5), glass * bank);
          } else {
            // The outside: dark ribs under a band of light.
            float rib = stroke(fract(vSeat.x / 7.0) - 0.5, 0.03) * smoothstep(0.0, 30.0, vWorld.y);
            color = vec3(0.007, 0.008, 0.011) + vec3(0.03, 0.033, 0.045) * rib;
            color += (team * 0.7 + 0.06) * stroke(vWorld.y - 33.0, 0.5);
          }
          gl_FragColor = vec4(color, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
  }

  // Dusk: a dark blue sky with the last light low in the west and the first stars.
  function skyMaterial() {
    return new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false,
      vertexShader: /* glsl */ `
        varying vec3 vAway;
        void main() {
          vAway = position;
          gl_Position = projectionMatrix * viewMatrix * vec4((modelMatrix * vec4(position, 1.0)).xyz, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vAway;
        ${GLSL_COMMON}
        void main() {
          vec3 d = normalize(vAway);
          float h = max(d.y, 0.0);
          float west = 0.5 + 0.5 * dot(normalize(d.xz + 1e-5), normalize(vec2(-0.8, -0.6)));
          vec3 color = mix(vec3(0.035, 0.05, 0.13), vec3(0.004, 0.007, 0.022), pow(max(h, 0.0), 0.45));
          color += vec3(0.22, 0.06, 0.09) * pow(max(1.0 - h, 0.0), 6.0) * (0.25 + west * west);
          color += vec3(0.5, 0.2, 0.05) * pow(max(1.0 - h, 0.0), 16.0) * west * west * west;
          // Stars: one faint point in some cells of a grid over the dome.
          vec2 grid = vec2(atan(d.z, d.x) * 60.0, d.y * 95.0);
          vec2 id = floor(grid), f = fract(grid) - 0.5;
          vec2 spot = (vec2(hash21(id + 5.0), hash21(id + 9.0)) - 0.5) * 0.6;
          float star = step(0.93, hash21(id)) * smoothstep(0.1, 0.0, length(f - spot));
          color += vec3(0.8, 0.85, 1.0) * star * smoothstep(0.15, 0.5, d.y) * hash21(id + 2.0);
          gl_FragColor = vec4(color * step(-0.02, d.y) + vec3(0.003) * step(d.y, -0.02), 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
  }

  // The ground around the arena: a dark apron up to the stands, lit a little by each goal's glow.
  function groundMaterial() {
    const S = STANDS;
    return new THREE.ShaderMaterial({
      uniforms: { uBlue: { value: teamColor(0) }, uOrange: { value: teamColor(1) } },
      vertexShader: /* glsl */ `
        varying vec3 vWorld;
        void main() {
          vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uBlue, uOrange;
        varying vec3 vWorld;
        ${GLSL_COMMON}
        void main() {
          vec2 p = vWorld.xz;
          vec2 q = abs(p) - vec2(${(S.halfX - S.bend).toFixed(1)}, ${(S.halfZ - S.bend).toFixed(1)});
          float edge = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - ${S.bend.toFixed(1)};
          vec3 team = p.y < 0.0 ? uBlue : uOrange;
          float end = smoothstep(20.0, 62.0, abs(p.y));
          float slabs = max(stroke(fract(p.x / 4.0) - 0.5, 0.004), stroke(fract(p.y / 4.0) - 0.5, 0.004));
          vec3 color = vec3(0.012, 0.014, 0.018) * (1.0 - 0.4 * slabs) + team * 0.012 * end;
          color += (team * (0.3 + 0.7 * end) + 0.05) * stroke(edge + 1.2, 0.05) * 0.9;
          gl_FragColor = vec4(edge < 0.0 ? color : vec3(0.004, 0.005, 0.007) + vec3(0.02, 0.017, 0.022) * exp(-edge / 90.0), 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
  }

  // ------------------------------------------------------------------------------------------- lighting

  // Reflections come from the stadium itself: its sky, stands and floodlights seen from the centre spot,
  // with the lit pitch below.
  function environment(surround) {
    const room = new THREE.Scene();
    const turf = new THREE.Mesh(new THREE.CircleGeometry(STANDS.halfX, 32), new THREE.MeshBasicMaterial({ color: 0x1d4a26 }));
    turf.rotation.x = -Math.PI / 2;
    room.add(surround, turf);
    const maker = new THREE.PMREMGenerator(NV.renderer);
    const map = maker.fromScene(room, 0.025, 0.5, 2500, { position: new THREE.Vector3(0, 2.5, 0) }).texture;
    maker.dispose();
    turf.geometry.dispose();
    turf.material.dispose();
    return map;
  }

  // The key light's shadow covers the whole arena: its camera is fitted around the field's box.
  function fitShadow(light) {
    const camera = light.shadow.camera, corner = new THREE.Vector3();
    camera.up.set(0, 0, 1);
    camera.position.copy(light.position);
    camera.lookAt(light.target.position);
    camera.updateMatrixWorld();
    const view = camera.matrixWorld.clone().invert();
    const box = [F.halfX * NV.S, F.ceiling * NV.S, (F.halfY + F.goalDepth * 0.6) * NV.S];
    const low = new THREE.Vector3(Infinity, Infinity, Infinity), high = low.clone().negate();
    for (let i = 0; i < 8; i++) {
      corner.set(i & 1 ? box[0] : -box[0], i & 2 ? box[1] : 0, i & 4 ? box[2] : -box[2]).applyMatrix4(view);
      low.min(corner);
      high.max(corner);
    }
    Object.assign(camera, { left: low.x, right: high.x, bottom: low.y, top: high.y, near: -high.z - 1, far: -low.z + 1 });
    camera.updateProjectionMatrix();
  }

  function lights(scene) {
    const key = new THREE.DirectionalLight(0xfff4e6, 3.1);
    key.position.set(16, 72, -14);
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.035;
    fitShadow(key);
    const rim = new THREE.DirectionalLight(0x8fb4ff, 0.9);
    rim.position.set(-40, 28, 36);
    const fill = new THREE.HemisphereLight(0xa9bcdf, 0x16261a, 0.55);
    const ends = NV.TEAMS.map((team, i) => {
      const glow = new THREE.PointLight(team.color, GOAL_LIGHT.power, 50, 2);
      glow.position.set(0, GOAL_LIGHT.height, (i ? 1 : -1) * (F.halfY * NV.S - 1.5));
      return glow;
    });
    scene.add(key, key.target, rim, fill, ...ends);
    return { key, rim, fill, blue: ends[0], orange: ends[1] };
  }

  // ------------------------------------------------------------------------------------------ boost pads

  // Every pad is a flat marker drawn from distances (its plate, glow and the ring that fills while it comes
  // back); a big pad adds a base plate and an orb floating over it.
  function padsBuild(scene, spots) {
    const count = spots.length, big = spots.filter((spot) => spot[3]);
    if (!count) return () => {};
    const corners = new Float32Array([-1, 0, -1, -1, 0, 1, 1, 0, 1, 1, 0, -1]);
    const marks = new THREE.InstancedBufferGeometry();
    marks.setAttribute("position", new THREE.BufferAttribute(corners, 3));
    marks.setIndex([0, 1, 2, 0, 2, 3]);
    marks.instanceCount = count;
    const place = new Float32Array(count * 4), live = new Float32Array(count * 3);
    spots.forEach((spot, i) => place.set([spot[0] * NV.S, spot[1] * NV.S, spot[3] ? 2.6 : 1.0, spot[3] ? 1 : 0], i * 4));
    const liveAttribute = new THREE.InstancedBufferAttribute(live, 3).setUsage(THREE.DynamicDrawUsage);
    marks.setAttribute("aPad", new THREE.InstancedBufferAttribute(place, 4));
    marks.setAttribute("aLive", liveAttribute);
    const premultiplied = { blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor };
    const markMesh = new THREE.Mesh(marks, new THREE.ShaderMaterial({
      uniforms: { uTime: time },
      transparent: true, depthWrite: false, ...premultiplied,
      vertexShader: /* glsl */ `
        attribute vec4 aPad;
        attribute vec3 aLive;
        varying vec2 vAt;
        varying float vBig;
        varying vec3 vLive;
        void main() {
          vAt = position.xz * aPad.z;
          vBig = aPad.w;
          vLive = aLive;
          vec3 p = vec3(aPad.x + vAt.x, aPad.w > 0.5 ? 0.082 : 0.012, aPad.y + vAt.y);
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        varying vec2 vAt;
        varying float vBig;
        varying vec3 vLive;
        ${GLSL_COMMON}
        void main() {
          float r = length(vAt), soft = fwidth(r);
          float turn = atan(vAt.x, -vAt.y) / 6.2831853 + 0.5;
          float on = vLive.x, back = vLive.y;
          // The part of the ring that is lit: all of it when the pad is there, else what has refilled.
          float filled = max(on, step(turn, back) * 0.7);
          vec3 hot = vec3(1.0, 0.56, 0.1), white = vec3(1.0, 0.86, 0.6);
          vec3 color = vec3(0.0);
          float alpha = 0.0;
          if (vBig > 0.5) {
            float ring = stroke(r - 0.66, 0.045);
            float marks = stroke(r - 0.82, 0.02) * step(0.12, fract(turn * 6.0));
            color += hot * ring * (0.05 + 2.4 * filled) + hot * marks * (0.04 + 0.7 * on);
            color += mix(hot, white, 0.4) * on * 0.3 * exp(-r * r / 0.07);
            // The orb's light on the plate and the turf around it.
            color += hot * on * (0.24 + 0.04 * sin(uTime * 3.0)) * exp(-max(r - 0.3, 0.0) * 1.9);
            color += white * vLive.z * 1.5 * exp(-r * 1.5);
          } else {
            float plate = 1.0 - smoothstep(0.34 - soft, 0.34 + soft, r);
            float core = 1.0 - smoothstep(0.13 - soft, 0.13 + soft, r);
            float ring = stroke(r - 0.25, 0.028);
            alpha = plate * 0.95;
            color = (vec3(0.01, 0.011, 0.014) + vec3(0.07, 0.075, 0.085) * stroke(r - 0.32, 0.012)) * alpha;
            color += mix(hot, white, 0.5 * on) * core * (0.035 + 3.0 * on) + hot * ring * (0.05 + 1.7 * filled);
            color += hot * on * 0.3 * exp(-max(r - 0.1, 0.0) * 4.5) * (1.0 - 0.85 * plate) + white * vLive.z * exp(-r * 3.5);
          }
          float fade = 1.0 - smoothstep(0.8, 1.0, r / (vBig > 0.5 ? 2.6 : 1.0));
          gl_FragColor = vec4(color * fade, alpha);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }));
    markMesh.frustumCulled = false;
    markMesh.renderOrder = -30;

    const plates = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(0.9, 1.0, 0.08, 40, 1),
      new THREE.MeshStandardMaterial({ color: 0x1b1e24, metalness: 0.9, roughness: 0.32 }),
      big.length,
    );
    const matrix = new THREE.Matrix4();
    big.forEach((spot, i) => plates.setMatrixAt(i, matrix.makeTranslation(spot[0] * NV.S, 0.04, spot[1] * NV.S)));
    plates.receiveShadow = true;

    const orbPlace = new Float32Array(big.length * 4);
    const orbAttribute = new THREE.InstancedBufferAttribute(orbPlace, 4).setUsage(THREE.DynamicDrawUsage);
    const orbShape = (base) => {
      const geometry = new THREE.InstancedBufferGeometry();
      geometry.setAttribute("position", base.attributes.position);
      geometry.setIndex(base.index);
      geometry.setAttribute("aOrb", orbAttribute);
      geometry.instanceCount = big.length;
      return geometry;
    };
    const orbVertex = /* glsl */ `
      attribute vec4 aOrb;
      uniform float uTime;
      vec3 orbCentre() {
        return vec3(aOrb.x, 1.02 + 0.07 * sin(uTime * 1.7 + aOrb.z), aOrb.y);
      }
      float orbSize() {
        return aOrb.w * (1.0 + 0.035 * sin(uTime * 3.1 + aOrb.z * 2.0));
      }`;
    const orbs = new THREE.Mesh(orbShape(new THREE.SphereGeometry(0.44, 28, 18)), new THREE.ShaderMaterial({
      uniforms: { uTime: time },
      vertexShader: /* glsl */ `
        ${orbVertex}
        varying vec3 vNormal;
        varying vec3 vToEye;
        void main() {
          vec3 p = orbCentre() + position * orbSize();
          vNormal = position;
          vToEye = cameraPosition - p;
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        varying vec3 vNormal;
        varying vec3 vToEye;
        void main() {
          vec3 n = normalize(vNormal);
          float edge = 1.0 - max(dot(n, normalize(vToEye)), 0.0);
          float swirl = 0.5 + 0.5 * sin(n.y * 7.0 + atan(n.z, n.x) * 3.0 + uTime * 2.2);
          vec3 color = mix(vec3(1.0, 0.6, 0.16) * 2.8, vec3(1.0, 0.24, 0.02) * 0.8, pow(max(edge, 0.0), 0.75));
          gl_FragColor = vec4(color * (0.62 + 0.38 * swirl), 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }));
    const halo = new THREE.Mesh(orbShape(new THREE.PlaneGeometry(2, 2)), new THREE.ShaderMaterial({
      uniforms: { uTime: time },
      transparent: true, depthWrite: false, blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      vertexShader: /* glsl */ `
        ${orbVertex}
        varying vec2 vAt;
        void main() {
          vAt = position.xy;
          vec4 eye = viewMatrix * vec4(orbCentre(), 1.0);
          eye.xy += position.xy * 1.15 * orbSize();
          gl_Position = projectionMatrix * eye;
        }`,
      fragmentShader: /* glsl */ `
        varying vec2 vAt;
        void main() {
          float r = length(vAt);
          float glow = exp(-r * r * 7.0) * 0.32 * (1.0 - smoothstep(0.7, 1.0, r));
          gl_FragColor = vec4(vec3(1.0, 0.5, 0.1) * glow, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }));
    for (const mesh of [orbs, halo]) mesh.frustumCulled = false;
    scene.add(markMesh);
    if (big.length) scene.add(plates, orbs, halo);

    const level = new Float32Array(count).fill(1);
    let snap = true;
    NV.on("seek", () => {
      snap = true;
    });
    return function padsUpdate(world, dt) {
      const k = snap || dt <= 0 ? 1 : 1 - Math.exp(-dt * 9);
      snap = false;
      let orb = 0;
      for (let i = 0; i < count && i < world.pads.length; i++) {
        const pad = world.pads[i], target = pad.active ? 1 : 0;
        const rising = k < 1 ? Math.max(0, target - level[i]) : 0;
        level[i] += (target - level[i]) * k;
        const wait = pad.big ? F.bigPadWait : F.smallPadWait;
        live[i * 3] = level[i];
        live[i * 3 + 1] = pad.active ? 1 : NV.clamp(1 - pad.wait / wait, 0, 1);
        live[i * 3 + 2] = rising;
        if (pad.big) {
          orbPlace[orb * 4] = place[i * 4];
          orbPlace[orb * 4 + 1] = place[i * 4 + 1];
          orbPlace[orb * 4 + 2] = i * 1.7;
          orbPlace[orb * 4 + 3] = NV.smooth(level[i]);
          orb++;
        }
      }
      liveAttribute.needsUpdate = true;
      orbAttribute.needsUpdate = true;
    };
  }

  // ----------------------------------------------------------------------------------------------- build

  const api = {
    floor: null,
    lights: null,
    environment: null,
    build(scene, { meshes, pads }) {
      NV.renderer.shadowMap.enabled = true;
      NV.renderer.shadowMap.type = THREE.PCFShadowMap;

      // The stadium, and with it the light the scene is seen in.
      const surround = new THREE.Group();
      // Floodlights and stands would smear across the pitch: the mirror shows the game, not the stadium.
      if (NV.post && NV.post.mirrorHidden) NV.post.mirrorHidden.push(surround);
      const standUniforms = {
        uBlue: { value: teamColor(0) }, uOrange: { value: teamColor(1) }, uTime: time, uCrowd: { value: 1 }, uCheer: { value: 0 },
      };
      const stands = new THREE.Mesh(standsGeometry(), standsMaterial(standUniforms));
      const sky = new THREE.Mesh(new THREE.SphereGeometry(1800, 32, 16), skyMaterial());
      const ground = new THREE.Mesh(new THREE.CircleGeometry(900, 48), groundMaterial());
      ground.rotation.x = -Math.PI / 2;
      ground.position.y = -0.05;
      // Drawn after everything solid, so only the pixels left open are shaded.
      ground.renderOrder = 9;
      sky.renderOrder = 10;
      surround.add(sky, ground, stands);
      for (const part of surround.children) part.frustumCulled = false;
      scene.environment = api.environment = environment(surround);
      scene.add(surround);
      api.lights = lights(scene);

      // The pitch.
      const pitch = pitchMaterial();
      const floor = new THREE.Mesh(pitchGeometry(), pitch);
      floor.receiveShadow = true;
      scene.add(floor);
      api.floor = floor;

      // The shell: the glass, and its back, faintly.
      const shell = soup(), net = soup();
      if (meshes) readMeshes(decode(meshes), shell, net);
      else generateShell(shell, net);
      // The ceiling reaches a little behind the ramps, which end at different places in the meshes.
      cover(shell, F.ceiling, 400, CEILING, [0, 0, -1]);
      const shape = geometryOf(shell, fieldLine), cell = shape.userData.cell;
      const pulse = { value: new THREE.Color(0, 0, 0) };
      const glass = new THREE.Mesh(shape, shellMaterial(cell, pulse));
      const outside = new THREE.Mesh(shape, outsideMaterial(cell));
      const netting = new THREE.Mesh(geometryOf(net, null), netMaterial());
      // Before anything else that is see-through: the net, the glass in front of it, then the glass's back.
      netting.renderOrder = -22;
      glass.renderOrder = -21;
      outside.renderOrder = -20;
      glass.receiveShadow = true;
      scene.add(netting, glass, outside, goalFrames());

      const padsUpdate = padsBuild(scene, pads || []);
      const quality = (q) => {
        const key = api.lights.key, size = [0, 1024, 2048, 4096][NV.clamp(q.shadows | 0, 0, 3)];
        key.castShadow = size > 0;
        if (size && key.shadow.mapSize.x !== size) {
          key.shadow.mapSize.set(size, size);
          key.shadow.radius = 1.6 * (size / 1024);
          if (key.shadow.map) key.shadow.map.dispose();
          key.shadow.map = null;
        }
        standUniforms.uCrowd.value = q.crowd ? 1 : 0;
      };
      quality(NV.q);
      NV.on("quality", (event) => quality(event.q || NV.q));

      // A goal lights the glass in the scorer's colour and stirs the crowd for a moment.
      let cheer = 0;
      const scorer = new THREE.Color();
      NV.on("goal", (goal) => {
        cheer = 1;
        scorer.copy(teamColor(goal.team));
      });
      NV.on("seek", () => {
        cheer = 0;
      });

      const mirror = pitch.userData.uniforms;
      api.update = (world, dt) => {
        time.value = world.t;
        padsUpdate(world, dt);
        cheer *= Math.exp(-dt * 0.9);
        pulse.value.copy(scorer).multiplyScalar(cheer * cheer * 0.75);
        standUniforms.uCheer.value = cheer;
        api.lights.blue.intensity = api.lights.orange.intensity = GOAL_LIGHT.power * (1 + 2 * cheer);
        // The mirrored scene is compiled into the pitch only while there is one.
        const seen = (NV.q.reflections && NV.post && NV.post.mirror) || null;
        if (!seen !== !pitch.defines.USE_MIRROR) {
          pitch.defines.USE_MIRROR = seen ? 1 : 0;
          pitch.envMapIntensity = seen ? 0.25 : 0.45;
          pitch.needsUpdate = true;
        }
        if (seen) {
          mirror.uMirror.value = seen.texture;
          mirror.uMirrorMatrix.value = seen.textureMatrix;
        }
      };
    },
    update() {},
  };
  return api;
})();
