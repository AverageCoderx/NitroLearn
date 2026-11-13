// The ball: thirty-two panels (twelve dark five-sided ones among twenty light six-sided ones) with sunken
// seams, so its spin shows from any side. The pattern is worked out per texel from the direction on the
// sphere and drawn once into three textures: colour, roughness and a normal map in the ball's own frame.
// Under the ball a ring and a soft shadow lie on the floor; both grow wider and fainter as it rises.

NV.ball = (() => {
  const S = NV.S;
  // Half the width of a seam and the place of the line set into each panel, as angles seen from the centre.
  const SEAM = 0.022, INSET = [0.075, 0.088];
  const COLOURS = {
    light: [233, 235, 239], dark: [30, 33, 40], seam: [10, 11, 13], line: [150, 156, 166], ring: [196, 201, 210],
  };

  // The panels are the faces of a truncated icosahedron pushed out onto the sphere. Each face is given by
  // its outward direction divided by its distance from the centre: the face a direction falls on is the
  // one where the dot product with that is largest.
  function faces() {
    const t = (1 + Math.sqrt(5)) / 2, corners = [], list = [];
    for (const a of [1, -1]) for (const b of [t, -t]) corners.push([0, a, b], [a, b, 0], [b, 0, a]);
    const unit = (v) => {
      const l = Math.hypot(v[0], v[1], v[2]);
      return [v[0] / l, v[1] / l, v[2] / l];
    };
    const near = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2] > 0.4;
    const points = corners.map(unit);
    // 0.81574 and 0.79465: how far the five- and six-sided faces lie from the centre of the solid whose
    // corners are a third of the way along the icosahedron's edges.
    for (const p of points) list.push({ n: p.map((v) => v / 0.81574), five: true });
    for (let i = 0; i < 12; i++) {
      for (let j = i + 1; j < 12; j++) {
        for (let k = j + 1; k < 12; k++) {
          const a = points[i], b = points[j], c = points[k];
          if (!near(a, b) || !near(b, c) || !near(a, c)) continue;
          const middle = unit([a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]);
          list.push({ n: middle.map((v) => v / 0.79465), five: false });
        }
      }
    }
    return list;
  }

  function textures(width, height) {
    const list = faces(), count = list.length, dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    const nx = Float32Array.from(list, (face) => face.n[0]), ny = Float32Array.from(list, (face) => face.n[1]);
    const nz = Float32Array.from(list, (face) => face.n[2]);
    // The faces around each face (those whose directions are closest), and the face on the far side.
    const around = list.map((face, i) => list.map((other, j) => j).filter((j) => {
      return j !== i && dot(face.n, list[j].n) > 0.7 * Math.sqrt(dot(face.n, face.n) * dot(list[j].n, list[j].n));
    }));
    const opposite = list.map((face) => list.findIndex((other) => dot(face.n, other.n) < -0.99 * dot(face.n, face.n)));
    const layers = ["colour", "surface", "normal"].map(() => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      return { canvas, context, image: context.createImageData(width, height) };
    });
    const [colour, surface, normal] = layers.map((layer) => layer.image.data);
    const middle = (INSET[0] + INSET[1]) / 2, band = INSET[1] - INSET[0], pixel = [0, 0, 0];
    let best = 0;
    // The pattern is the same seen from the opposite side, so half the texels are worked out and each is
    // written twice, with the normal turned round.
    for (let py = 0; py < height / 2; py++) {
      // The same wrap as three's sphere: the top row is +y, u runs around from -x.
      const theta = ((py + 0.5) / height) * Math.PI, sin = Math.sin(theta), y = Math.cos(theta);
      for (let px = 0; px < width; px++) {
        const phi = ((px + 0.5) / width) * Math.PI * 2, x = -Math.cos(phi) * sin, z = Math.sin(phi) * sin;
        // The face under this texel: from the last texel's, step to a better neighbour until there is none.
        let bestValue = x * nx[best] + y * ny[best] + z * nz[best], second = -1, secondValue = -9;
        for (let moved = true; moved; ) {
          const near = around[best];
          moved = false;
          second = -1;
          secondValue = -9;
          for (let k = 0; k < near.length; k++) {
            const j = near[k], value = x * nx[j] + y * ny[j] + z * nz[j];
            if (value > bestValue) {
              best = j;
              bestValue = value;
              moved = true;
              break;
            }
            if (value > secondValue) {
              second = j;
              secondValue = value;
            }
          }
        }
        // Toward the nearest seam: the direction in which the two faces' values part, along the surface.
        let gx = nx[best] - nx[second], gy = ny[best] - ny[second], gz = nz[best] - nz[second];
        const lift = gx * x + gy * y + gz * z;
        gx -= lift * x;
        gy -= lift * y;
        gz -= lift * z;
        const slope = Math.hypot(gx, gy, gz) || 1, edge = (bestValue - secondValue) / slope, five = list[best].five;
        const groove = NV.smooth(NV.clamp(edge / SEAM, 0, 1));
        const line = NV.smooth(NV.clamp(1 - Math.abs(edge - middle) / band, 0, 1)) * (five ? 0.9 : 0.45);
        const panel = five ? COLOURS.dark : COLOURS.light, mark = five ? COLOURS.ring : COLOURS.line;
        for (let c = 0; c < 3; c++) pixel[c] = NV.lerp(COLOURS.seam[c], NV.lerp(panel[c], mark[c], line), groove);
        const rough = 255 * NV.lerp(0.8, five ? 0.46 : 0.3, groove);
        // The seam is a rounded groove: inside it the normal leans toward its middle.
        const lean = (Math.sin(groove * Math.PI) * 0.75) / slope;
        const mx = x - gx * lean, my = y - gy * lean, mz = z - gz * lean, ml = 127.5 / Math.hypot(mx, my, mz);
        for (let side = 0; side < 2; side++) {
          const at = side ? ((height - 1 - py) * width + ((px + width / 2) % width)) * 4 : (py * width + px) * 4;
          // A small step in brightness from panel to panel keeps the light ones apart.
          const face = side ? opposite[best] : best, shade = five ? 1 : 0.94 + (0.06 * ((face * 7) % 5)) / 4;
          const turn = side ? -ml : ml;
          colour[at] = pixel[0] * shade;
          colour[at + 1] = pixel[1] * shade;
          colour[at + 2] = pixel[2] * shade;
          surface[at + 1] = rough;
          normal[at] = 127.5 + mx * turn;
          normal[at + 1] = 127.5 + my * turn;
          normal[at + 2] = 127.5 + mz * turn;
          colour[at + 3] = surface[at + 3] = normal[at + 3] = 255;
        }
      }
    }
    return layers.map((layer, i) => {
      layer.context.putImageData(layer.image, 0, 0);
      const texture = new THREE.CanvasTexture(layer.canvas);
      texture.wrapS = THREE.RepeatWrapping;
      texture.anisotropy = 8;
      if (!i) texture.colorSpace = THREE.SRGBColorSpace;
      return texture;
    });
  }

  // A soft disc, or a ring with a faint fill, for the marks on the floor.
  function markTexture(ring) {
    const size = 128, canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const context = canvas.getContext("2d"), image = context.createImageData(size, size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const r = Math.hypot(x + 0.5 - size / 2, y + 0.5 - size / 2) / (size / 2);
        const band = NV.smooth(NV.clamp(1 - Math.abs(r - 0.84) / 0.1, 0, 1));
        const fill = 1 - NV.smooth(NV.clamp(r / 0.95, 0, 1));
        image.data.set([255, 255, 255, 255 * (ring ? Math.max(band, fill * 0.18) : fill)], (y * size + x) * 4);
      }
    }
    context.putImageData(image, 0, 0);
    return new THREE.CanvasTexture(canvas);
  }

  function mark(texture, color, blending) {
    const material = new THREE.MeshBasicMaterial({
      map: texture, color, blending, transparent: true, depthWrite: false,
      polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), material);
    mesh.renderOrder = 2;
    return mesh;
  }

  return {
    object: null,
    ring: null,
    shadow: null,

    build(scene) {
      const radius = NV.FIELD.ballRadius * S, [map, roughnessMap, normalMap] = textures(1024, 512);
      const material = new THREE.MeshStandardMaterial({
        map, roughnessMap, normalMap, normalMapType: THREE.ObjectSpaceNormalMap, metalness: 0,
      });
      this.object = new THREE.Mesh(new THREE.SphereGeometry(radius, 48, 24), material);
      this.object.castShadow = this.object.receiveShadow = true;
      this.ring = mark(markTexture(true), 0xffffff, THREE.AdditiveBlending);
      this.shadow = mark(markTexture(false), 0x000000, THREE.NormalBlending);
      scene.add(this.object, this.shadow, this.ring);
    },

    update(world) {
      const ball = world.ball, radius = NV.FIELD.ballRadius * S;
      this.object.position.copy(ball.pos);
      this.object.quaternion.copy(ball.quat);
      // 0 with the ball on the floor, 1 near the ceiling.
      const high = NV.clamp((ball.pos.y - radius) / (NV.FIELD.ceiling * S * 0.8), 0, 1);
      const ringSize = radius * (1.7 + 2.6 * high), shadowSize = radius * (2.3 + 3 * high);
      this.ring.position.set(ball.pos.x, 0.02, ball.pos.z);
      this.ring.scale.set(ringSize, 1, ringSize);
      this.ring.material.opacity = NV.lerp(0.7, 0.3, Math.sqrt(high));
      this.shadow.position.set(ball.pos.x, 0.015, ball.pos.z);
      this.shadow.scale.set(shadowSize, 1, shadowSize);
      this.shadow.material.opacity = (NV.q.shadows ? 0.3 : 0.6) * (1 - high) ** 2;
    },
  };
})();
