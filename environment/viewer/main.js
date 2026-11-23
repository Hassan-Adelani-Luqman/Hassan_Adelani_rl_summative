// Afya Sentinel viewer: a low-poly diorama of Bugesera District driven by the simulation stream.
// World units are kilometres: env (x east, y north) maps to three.js (x, -z), with y up.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/CSS2DRenderer.js';

const $ = (id) => document.getElementById(id);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const lerp = (a, b, t) => a + (b - a) * t;
const params = new URLSearchParams(location.search);

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ renderer, camera, lights
const container = $('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const labels = new CSS2DRenderer();
labels.setSize(window.innerWidth, window.innerHeight);
Object.assign(labels.domElement.style, { position: 'absolute', top: '0', pointerEvents: 'none' });
container.appendChild(labels.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 600);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = 1.32;
controls.minDistance = 6;
controls.maxDistance = 140;

const SKY = { day: new THREE.Color(0x9fcbe8), night: new THREE.Color(0x0b1630), storm: new THREE.Color(0x6c7a88) };
scene.background = SKY.day.clone();
scene.fog = new THREE.Fog(scene.background, 70, 160);
const hemi = new THREE.HemisphereLight(0xd8ecff, 0x3d4a2c, 0.9);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff0d2, 2.3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -45, right: 45, top: 45, bottom: -45, near: 1, far: 220 });
scene.add(sun, sun.target);

// ------------------------------------------------------------------ state
const S = {
  data: null, view: 'agent', follow: false, paused: false, speed: Number(params.get('speed') || 1),
  ws: null, history: [], villages: [], roads: [], roadByPair: new Map(), locs: [], totalPop: 1,
  van: null, vanAnim: null, effects: [], rainTarget: 0, nightTarget: 0, night: 0, rain: 0, lastFrame: null,
  height: null, center: { x: 0, y: 0 },
};

const toWorld = (x, y, h = 0) => new THREE.Vector3(x - S.center.x, h, -(y - S.center.y));

function distToPolyline(x, y, pts) {
  let best = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i]; const [bx, by] = pts[i + 1];
    const dx = bx - ax; const dy = by - ay;
    const t = clamp01(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy));
    best = Math.min(best, Math.hypot(x - (ax + t * dx), y - (ay + t * dy)));
  }
  return best;
}

// ------------------------------------------------------------------ world building
function makeHeight(data) {
  const lakes = data.lakes.map((l) => ({ x: l.x, y: l.y, r: l.radius }));
  const gashora = data.locations.find((l) => l.name === 'Gashora');
  if (gashora) { // Gashora's eastern lakes (stylised, unnamed)
    lakes.push({ x: gashora.x + 3.2, y: gashora.y - 1.6, r: 1.3 }, { x: gashora.x + 1.4, y: gashora.y + 2.7, r: 0.9 });
  }
  const wet = data.locations.filter((l) => l.wetland);
  const river = data.river.points;
  return (x, y) => {
    let v = 1.7 + 0.55 * Math.sin(x * 0.19 + 1.3) * Math.cos(y * 0.16 - 0.4)
      + 0.35 * Math.sin(x * 0.43 + y * 0.29 + 0.7) + 0.18 * Math.cos(x * 0.83 - y * 0.61);
    for (const l of lakes) v -= 2.6 * Math.exp(-((Math.hypot(x - l.x, y - l.y) / (l.r * 1.15)) ** 2));
    for (const w of wet) v -= 0.45 * Math.exp(-((Math.hypot(x - w.x, y - w.y) / 3.5) ** 2));
    v -= 2.0 * Math.exp(-((distToPolyline(x, y, river) / 1.1) ** 2));
    return v;
  };
}

function terrainColour(h) {
  const stops = [[-1, 0x8a9a5b], [0.25, 0x8fa060], [0.6, 0x5f9e4f], [1.6, 0x7fae52], [2.6, 0xa8925a]];
  for (let i = 0; i < stops.length - 1; i++) {
    const [h0, c0] = stops[i]; const [h1, c1] = stops[i + 1];
    if (h <= h1) return new THREE.Color(c0).lerp(new THREE.Color(c1), clamp01((h - h0) / (h1 - h0)));
  }
  return new THREE.Color(stops.at(-1)[1]);
}

function label(html, cls = 'label') {
  const div = document.createElement('div');
  div.className = cls;
  div.innerHTML = html;
  return new CSS2DObject(div);
}

function buildWorld(data) {
  S.data = data;
  S.locs = data.locations;
  S.totalPop = data.locations.reduce((a, l) => a + (l.population || 0), 0);
  const xs = [...data.locations.map((l) => l.x), ...data.lakes.flatMap((l) => [l.x - l.radius, l.x + l.radius]),
    ...data.river.points.map((p) => p[0])];
  const ys = [...data.locations.map((l) => l.y), ...data.lakes.flatMap((l) => [l.y - l.radius, l.y + l.radius]),
    ...data.river.points.map((p) => p[1])];
  const pad = 14; // generous margin so the terrain's edge stays in the fog
  const [minX, maxX, minY, maxY] = [Math.min(...xs) - pad, Math.max(...xs) + pad, Math.min(...ys) - pad, Math.max(...ys) + pad];
  S.center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  const sizeX = maxX - minX; const sizeY = maxY - minY;
  const h = makeHeight(data);
  S.height = (x, y) => h(x, y);

  // terrain
  const geo = new THREE.PlaneGeometry(sizeX, sizeY, 170, 170);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const colours = [];
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) + S.center.x; const y = -pos.getZ(i) + S.center.y;
    const hh = h(x, y);
    pos.setY(i, hh);
    const c = terrainColour(hh);
    colours.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geo.computeVertexNormals();
  const terrain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.95 }));
  terrain.receiveShadow = true;
  scene.add(terrain);

  // water: one plane; lakes and the river are where the terrain dips below it
  const water = new THREE.Mesh(new THREE.PlaneGeometry(sizeX, sizeY),
    new THREE.MeshStandardMaterial({ color: 0x3c7db5, transparent: true, opacity: 0.86, roughness: 0.25, metalness: 0.1 }));
  water.rotateX(-Math.PI / 2);
  water.position.y = 0.0;
  scene.add(water);
  for (const lake of data.lakes) {
    const l = label(`<small><i>${lake.name}</i></small>`);
    l.position.copy(toWorld(lake.x, lake.y, 0.4));
    scene.add(l);
  }
  const [rx, ry] = data.river.points[1]; // the western stretch, away from the village labels
  const riverLabel = label(`<small><i>${data.river.name} river</i></small>`);
  riverLabel.position.copy(toWorld(rx, ry, 0.4));
  scene.add(riverLabel);

  buildRoads(data);
  buildHub(data.locations[data.hub]);
  for (const loc of data.locations) if (!loc.hub) S.villages[loc.id - 1] = buildVillage(loc);
  S.van = buildVan();
  scene.add(S.van);
  parkVan(data.hub);
  buildRain(sizeX, sizeY);

  const radius = Math.max(sizeX, sizeY) - 2 * pad; // frame the district, not the padding
  const focus = toWorld(S.center.x, S.center.y + 0.5, 0); // centred so neither Mwogo nor Rweru hides under a bar
  camera.position.copy(focus).add(new THREE.Vector3(0, radius * 0.8, radius * 0.68));
  controls.target.copy(focus);
  sun.position.set(-25, 40, 18);
}

function buildRoads(data) {
  for (const road of data.roads) {
    const a = S.locs[road.a]; const b = S.locs[road.b];
    const rng = mulberry32(road.id * 977 + 13);
    const nx = -(b.y - a.y); const ny = b.x - a.x; const len = Math.hypot(nx, ny) || 1;
    const pts = [];
    for (let k = 0; k <= 6; k++) {
      const t = k / 6;
      const wiggle = k === 0 || k === 6 ? 0 : (rng() - 0.5) * 1.3 * Math.sin(Math.PI * t);
      const x = lerp(a.x, b.x, t) + (nx / len) * wiggle; const y = lerp(a.y, b.y, t) + (ny / len) * wiggle;
      pts.push(toWorld(x, y, Math.max(S.height(x, y), 0.1) + 0.12));
    }
    const curve = new THREE.CatmullRomCurve3(pts);
    const material = new THREE.MeshStandardMaterial({ color: 0xb5793f, roughness: 1, emissive: 0x000000 });
    const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 72, 0.13, 6, false), material);
    mesh.receiveShadow = true;
    scene.add(mesh);
    const entry = { ...road, curve, mesh, samples: curve.getSpacedPoints(90), flooded: 0, highlight: 0 };
    S.roads.push(entry);
    S.roadByPair.set(`${road.a}-${road.b}`, entry);
    S.roadByPair.set(`${road.b}-${road.a}`, entry);
  }
}

function buildHub(loc) {
  const g = new THREE.Group();
  const base = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.9, 1.3), new THREE.MeshStandardMaterial({ color: 0xf4f6f8 }));
  base.position.y = 0.45;
  const roof = new THREE.Mesh(new THREE.BoxGeometry(2.05, 0.12, 1.45), new THREE.MeshStandardMaterial({ color: 0x9aa7b4 }));
  roof.position.y = 0.96;
  const red = new THREE.MeshStandardMaterial({ color: 0xe53935, emissive: 0x7a0000 });
  const c1 = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.14, 0.05), red);
  const c2 = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.5, 0.05), red);
  c1.position.set(0, 0.55, 0.68); c2.position.set(0, 0.55, 0.68);
  for (const m of [base, roof, c1, c2]) { m.castShadow = true; g.add(m); }
  g.position.copy(toWorld(loc.x, loc.y, S.height(loc.x, loc.y)));
  const l = label(`${loc.name}<small>district hub · hospital</small>`, 'label hub');
  l.position.set(0, 2.0, 0);
  g.add(l);
  scene.add(g);
}

function rampColour(t) { // 0 green -> 0.5 yellow -> 1 red
  const c = new THREE.Color(0x3ccf6e);
  return t < 0.5 ? c.lerp(new THREE.Color(0xffd23f), t / 0.5) : new THREE.Color(0xffd23f).lerp(new THREE.Color(0xff3b30), (t - 0.5) / 0.5);
}

function buildVillage(loc) {
  const centre = toWorld(loc.x, loc.y, S.height(loc.x, loc.y));
  const group = new THREE.Group();
  scene.add(group);
  const rng = mulberry32(loc.id * 131 + 7);
  const walls = new THREE.MeshStandardMaterial({ color: 0xe9dbb8, roughness: 0.9 });
  const thatch = new THREE.MeshStandardMaterial({ color: 0x8a6a3c, roughness: 1, flatShading: true });
  const huts = Math.round(4 + loc.population / 70);
  for (let j = 0, tries = 0; j < huts && tries < 60; tries++) {
    const ang = rng() * Math.PI * 2; const r = 0.55 + rng() * 1.25;
    const x = loc.x + Math.cos(ang) * r; const y = loc.y + Math.sin(ang) * r;
    const hh = S.height(x, y);
    if (hh < 0.12) continue;
    const s = 0.8 + rng() * 0.5;
    const wall = new THREE.Mesh(new THREE.CylinderGeometry(0.22 * s, 0.24 * s, 0.32 * s, 10), walls);
    const roof = new THREE.Mesh(new THREE.ConeGeometry(0.31 * s, 0.3 * s, 10), thatch);
    wall.position.copy(toWorld(x, y, hh + 0.16 * s));
    roof.position.copy(toWorld(x, y, hh + 0.32 * s + 0.15 * s));
    wall.castShadow = roof.castShadow = true;
    scene.add(wall, roof);
    j++;
  }
  // Agent View: a pillar whose height is the CHW-reported fever rate
  const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 1, 18),
    new THREE.MeshStandardMaterial({ color: 0xffb547, emissive: 0xff8a00, emissiveIntensity: 0.7, transparent: true, opacity: 0.92 }));
  pillar.position.copy(centre);
  pillar.castShadow = true;
  // Reality View: ground glow by true prevalence, a hotspot ring, a mosquito swarm
  const glowY = Math.max(...[0, 1, 2, 3, 4, 5].map((k) => S.height(loc.x + 1.6 * Math.cos(k), loc.y + 1.6 * Math.sin(k))), centre.y) + 0.06;
  const glow = new THREE.Mesh(new THREE.CircleGeometry(2.3, 40),
    new THREE.MeshBasicMaterial({ color: 0x3ccf6e, transparent: true, opacity: 0.6, depthWrite: false }));
  glow.rotateX(-Math.PI / 2);
  glow.position.set(centre.x, glowY, centre.z);
  const ring = new THREE.Mesh(new THREE.RingGeometry(2.45, 2.7, 48),
    new THREE.MeshBasicMaterial({ color: 0xff3b30, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false }));
  ring.rotateX(-Math.PI / 2);
  ring.position.set(centre.x, glowY + 0.02, centre.z);
  const swarmN = 90;
  const swarmGeo = new THREE.BufferGeometry();
  swarmGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(swarmN * 3), 3));
  const swarm = new THREE.Points(swarmGeo, new THREE.PointsMaterial({ color: 0x111111, size: 0.22 }));
  const swarmParams = Array.from({ length: swarmN }, () => ({ a: rng() * 6.28, r: 0.4 + rng() * 2.0, h: 0.4 + rng() * 1.6, w: 0.6 + rng() * 1.8 }));
  // bed-net dome, fading with the days of protection left
  const dome = new THREE.Mesh(new THREE.SphereGeometry(2.15, 36, 14, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshPhongMaterial({ color: 0x7af3ec, emissive: 0x1d6f6a, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, shininess: 90 }));
  dome.position.copy(centre);
  dome.visible = false;
  scene.add(pillar, glow, ring, swarm, dome);
  const tag = label(`${loc.name}<small></small>`);
  tag.position.copy(centre).add(new THREE.Vector3(0, 2.4, 0));
  scene.add(tag);
  return { loc, centre, pillar, glow, ring, swarm, swarmParams, dome, tag, state: null,
           pillarTarget: 0.2, domeTarget: 0, swarmTarget: 0, glowColour: new THREE.Color(0x3ccf6e) };
}

function buildVan() {
  const g = new THREE.Group();
  const white = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1d2733 });
  const red = new THREE.MeshStandardMaterial({ color: 0xe53935, emissive: 0x550000 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.42, 0.95), white); body.position.set(0, 0.36, -0.06);
  const cab = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.34, 0.32), white); cab.position.set(0, 0.32, 0.55);
  const glass = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.16, 0.05), dark); glass.position.set(0, 0.4, 0.72);
  const c1 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.03, 0.1), red); c1.position.set(0, 0.585, -0.06);
  const c2 = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.03, 0.34), red); c2.position.set(0, 0.585, -0.06);
  for (const m of [body, cab, glass, c1, c2]) { m.castShadow = true; g.add(m); }
  for (const [x, z] of [[-0.26, 0.45], [0.26, 0.45], [-0.26, -0.35], [0.26, -0.35]]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.08, 12), dark);
    w.rotation.z = Math.PI / 2; w.position.set(x, 0.12, z); g.add(w);
  }
  g.scale.setScalar(1.25);
  return g;
}

function buildRain(sizeX, sizeY) {
  const n = 7000;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    arr[i * 3] = (Math.random() - 0.5) * sizeX;
    arr[i * 3 + 1] = Math.random() * 20;
    arr[i * 3 + 2] = (Math.random() - 0.5) * sizeY;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  geo.setDrawRange(0, 0);
  S.rainPoints = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xc6dcf5, size: 0.16, transparent: true, opacity: 0.7 }));
  S.rainN = n;
  scene.add(S.rainPoints);
}

// ------------------------------------------------------------------ van movement
function locationPoint(id) {
  const loc = S.locs[id];
  return toWorld(loc.x, loc.y, Math.max(S.height(loc.x, loc.y), 0.1) + 0.12);
}

function parkVan(id) {
  const p = locationPoint(id);
  S.van.position.copy(p).add(new THREE.Vector3(0.9, 0, 0.9));
  S.vanAnim = null;
}

function routePoints(route) {
  const pts = [];
  for (let i = 0; i < route.length - 1; i++) {
    const road = S.roadByPair.get(`${route[i]}-${route[i + 1]}`);
    if (!road) continue;
    const seg = road.a === route[i] ? road.samples : [...road.samples].reverse();
    pts.push(...(pts.length ? seg.slice(1) : seg));
    road.highlight = 1;
  }
  return pts;
}

function startTrip(frame) {
  const pts = routePoints(frame.route);
  if (pts.length < 2) { parkVan(frame.location); return; }
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
  S.vanAnim = { pts, cum, total: cum.at(-1), start: performance.now(), ms: Math.max(300, frame.duration_s * 1000 * 0.92),
                fraction: frame.events.stranded ? 0.55 : 1, end: frame.location, stranded: frame.events.stranded };
}

function pointAlong(anim, dist) {
  const { pts, cum } = anim;
  let i = 1;
  while (i < cum.length - 1 && cum[i] < dist) i++;
  const t = clamp01((dist - cum[i - 1]) / ((cum[i] - cum[i - 1]) || 1));
  return pts[i - 1].clone().lerp(pts[i], t);
}

function updateVan(now) {
  const a = S.vanAnim;
  if (!a) return;
  const raw = clamp01((now - a.start) / a.ms);
  const eased = raw < 0.5 ? 2 * raw * raw : 1 - (-2 * raw + 2) ** 2 / 2;
  const dist = eased * a.fraction * a.total;
  const p = pointAlong(a, dist);
  const ahead = pointAlong(a, Math.min(a.total, dist + 0.3));
  S.van.position.copy(p);
  if (ahead.distanceTo(p) > 1e-3) S.van.lookAt(ahead.x, p.y, ahead.z);
  if (raw >= 1 && !a.stranded) parkVan(a.end);
}

// ------------------------------------------------------------------ effects
function ripple(village, colour = 0x5ad18a) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.05, 48),
    new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
  ring.rotateX(-Math.PI / 2);
  ring.position.copy(village.centre).add(new THREE.Vector3(0, 0.25, 0));
  scene.add(ring);
  S.effects.push({ kind: 'ripple', obj: ring, start: performance.now(), ms: 1500 });
}

function popup(position, text, cls = 'popup', key = null) {
  if (key !== null) { // one pop-up per place: a new one replaces the old
    S.effects.filter((e) => e.key === key).forEach((e) => { e.start = -Infinity; });
  }
  const p = label(text, cls);
  p.position.copy(position).add(new THREE.Vector3(0, 3.7, 0)); // clear of the two-line village label
  scene.add(p);
  S.effects.push({ kind: 'popup', obj: p, start: performance.now(), ms: 1600, y0: p.position.y, key });
}

function updateEffects(now) {
  S.effects = S.effects.filter((e) => {
    const t = (now - e.start) / e.ms;
    if (t >= 1) { scene.remove(e.obj); if (e.obj.element) e.obj.element.remove(); return false; }
    if (e.kind === 'ripple') { e.obj.scale.setScalar(1 + t * 3.2); e.obj.material.opacity = 0.9 * (1 - t); }
    if (e.kind === 'popup') { e.obj.position.y = e.y0 + t * 0.8; e.obj.element.style.opacity = String(1 - t * t); }
    return true;
  });
}

// ------------------------------------------------------------------ applying frames
function applyVillages(frame) {
  frame.villages.forEach((v, i) => {
    const vil = S.villages[i];
    vil.state = v;
    vil.pillarTarget = 0.25 + v.reported * 7.5;
    vil.domeTarget = v.net_days > 0 ? 0.12 + 0.3 * clamp01(v.net_days / S.data.constants.net_days) : 0;
    vil.swarmTarget = clamp01(v.transmission / 0.03);
    vil.glowColour = rampColour(clamp01(v.true_prevalence / S.data.constants.outbreak_threshold));
    updateLabel(vil);
  });
}

function updateLabel(vil) {
  const v = vil.state;
  if (!v) return;
  const sub = S.view === 'agent'
    ? `${v.reported_count} fever reports${v.net_days > 0 ? ` · nets ${v.net_days.toFixed(1)}d` : ''}`
    : `${(v.true_prevalence * 100).toFixed(1)}% infected${v.hotspot ? ' · hotspot' : ''}`;
  vil.tag.element.innerHTML = `${vil.loc.name}<small>${sub}</small>`;
}

function setView(view) {
  S.view = view;
  const badge = $('view-badge');
  badge.textContent = view === 'agent' ? 'AGENT VIEW' : 'REALITY VIEW';
  badge.className = `badge ${view === 'agent' ? 'agent' : 'reality'}`;
  for (const vil of S.villages) {
    vil.pillar.visible = view === 'agent';
    vil.glow.visible = view === 'reality';
    vil.swarm.visible = view === 'reality';
    vil.ring.visible = view === 'reality' && vil.state?.hotspot;
    updateLabel(vil);
  }
}

function applyFrame(frame, isStart = false) {
  S.lastFrame = frame;
  applyVillages(frame);
  setView(S.view);
  const flooded = new Set(frame.flooded_roads);
  for (const r of S.roads) r.floodedTarget = flooded.has(r.id) ? 1 : 0;
  S.rainTarget = frame.weather.rain;
  S.nightTarget = frame.tick % 2 === 1 ? 1 : 0;
  if (isStart) parkVan(frame.location);
  else if (frame.route.length > 1) startTrip(frame);
  else parkVan(frame.location);

  const ev = frame.events;
  if (!isStart) {
    if (ev.screened !== null) {
      const vil = S.villages[ev.screened];
      ripple(vil);
      const found = [[ev.early, 'early'], [ev.symptomatic, 'symptomatic'], [ev.severe, 'severe referred']]
        .filter(([n]) => n > 0).map(([n, what]) => `+${n} ${what}`).join(' · ');
      popup(vil.centre, found || 'screened: no cases found', 'popup', ev.screened);
    }
    if (ev.netted !== null) {
      const vil = S.villages[ev.netted];
      ripple(vil, 0x4fd6d0);
      popup(vil.centre, `Nets: ${S.data.constants.net_days} days`, 'popup nets', ev.netted);
    }
    if (ev.stranded) popup(S.van.position, 'Out of fuel: van stranded', 'popup bad', 'van');
    if (ev.invalid) popup(S.van.position, 'Invalid action', 'popup bad', 'van');
  }
  updateHud(frame, isStart);
}

// ------------------------------------------------------------------ HUD
function gauge(id, frac, text) {
  $(`g-${id}`).style.width = `${(clamp01(frac) * 100).toFixed(1)}%`;
  $(`v-${id}`).textContent = text;
}

const SHORT = (name) => name.replace('Travel to ', '→ ').replace('Prevent (bed nets)', 'Bed nets').replace('Return to hub', 'Restock (hub)');

function updateThinking(thinking) {
  const bars = $('bars');
  const actions = S.data.actions;
  if (!bars.children.length) {
    for (const name of actions) {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<span>${SHORT(name)}</span><div class="track"><div class="fill"></div></div><span class="val"></span>`;
      bars.appendChild(row);
    }
  }
  const kind = thinking?.kind;
  $('thinking-kind').textContent = kind === 'probability' ? 'policy probabilities' : kind === 'q_value' ? 'Q-values' : 'baseline rule';
  const values = thinking?.values || [];
  const valid = values.filter((v) => v !== null && v !== undefined);
  const lo = Math.min(...valid); const hi = Math.max(...valid);
  [...bars.children].forEach((row, a) => {
    const v = values[a];
    const masked = v === null || v === undefined;
    row.classList.toggle('masked', masked);
    row.classList.toggle('chosen', thinking?.chosen === a);
    const width = masked ? 0 : kind === 'probability' ? v : (hi > lo ? (v - lo) / (hi - lo) : 1);
    row.querySelector('.fill').style.width = `${(clamp01(width) * 100).toFixed(1)}%`;
    row.querySelector('.val').textContent = masked ? '—' : kind === 'probability' ? v.toFixed(2) : v.toFixed(3);
  });
  if (thinking?.rationale) $('rationale').textContent = thinking.rationale;
}

function drawLine(ctx, xs, ys, xMax, yMax, colour, w, h) {
  if (xs.length < 2) return;
  ctx.strokeStyle = colour; ctx.lineWidth = 2 * devicePixelRatio; ctx.beginPath();
  xs.forEach((x, i) => {
    const px = (x / xMax) * w; const py = h - (ys[i] / yMax) * (h - 8) - 4;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function prepCanvas(id) {
  const c = $(id);
  const w = c.clientWidth * devicePixelRatio; const h = c.clientHeight * devicePixelRatio;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

function drawCharts() {
  const hist = S.history;
  const maxDay = S.data?.constants ? S.data.constants.max_ticks / S.data.constants.ticks_per_day : 30;
  const [ctx, w, h] = prepCanvas('c-epi');
  const days = hist.map((p) => p.day);
  const truth = hist.map((p) => p.truth); const reported = hist.map((p) => p.reported);
  const yMax = Math.max(0.05, ...truth, ...reported) * 1.1;
  ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = `${10 * devicePixelRatio}px sans-serif`;
  ctx.fillText(`${(yMax * 100).toFixed(0)}% of population`, 4, 12 * devicePixelRatio);
  drawLine(ctx, days, truth, maxDay, yMax, '#ff5a5a', w, h);
  drawLine(ctx, days, reported, maxDay, yMax, '#ffb547', w, h);
  const [ctx2, w2, h2] = prepCanvas('c-reward');
  const rets = hist.map((p) => p.ret);
  const lo = Math.min(0, ...rets);
  drawLine(ctx2, days, rets.map((r) => r - lo), maxDay, Math.max(1, -lo), '#7fa3d6', w2, h2);
}

function updateHud(frame, isStart) {
  const c = S.data.constants;
  $('day').textContent = `Day ${frame.day.toFixed(1)}`;
  $('daynight').textContent = frame.tick % 2 === 1 ? '☾' : '☀';
  gauge('fuel', frame.stock.fuel_frac, `${frame.stock.fuel_km.toFixed(0)} km`);
  gauge('kits', frame.stock.kits / c.kit_capacity, `${frame.stock.kits} / ${c.kit_capacity}`);
  gauge('nets', frame.stock.nets / c.net_capacity, `${frame.stock.nets} / ${c.net_capacity}`);
  gauge('rain', frame.weather.rain, `${(frame.weather.rain * 100).toFixed(0)}%`);
  gauge('forecast', frame.weather.forecast, `${(frame.weather.forecast * 100).toFixed(0)}%`);
  gauge('trans', (frame.weather.rain_multiplier - 0.3) / 1.7, `×${frame.weather.rain_multiplier.toFixed(2)}`);
  $('v-return').textContent = frame.return.toFixed(0);
  $('t-inf').textContent = Math.round(frame.totals.infections);
  $('t-screens').textContent = frame.totals.screens;
  $('t-nets').textContent = frame.totals.nets_placed;
  $('t-km').textContent = Math.round(frame.totals.km);
  const reportedShare = frame.villages.reduce((a, v) => a + v.reported_count, 0) / S.totalPop;
  S.history.push({ day: frame.day, truth: frame.district.true_prevalence, reported: reportedShare, ret: frame.return });
  drawCharts();
  if (!isStart) {
    updateThinking(frame.thinking);
    const where = S.locs[frame.origin].name;
    const li = document.createElement('li');
    const extra = frame.events.screened !== null ? ` · +${frame.events.early} early` : '';
    li.innerHTML = `<b>Day ${frame.day.toFixed(1)}</b> ${SHORT(frame.action_name)}${frame.action < 8 ? '' : ` @ ${where}`}${extra}`;
    $('log').prepend(li);
    while ($('log').children.length > 10) $('log').lastChild.remove();
  }
}

function showSummary(msg) {
  const ref = msg.reference || {};
  const rows = [['This agent', msg.return, msg.infections], ...Object.entries(ref).map(([k, r]) => [k.replace('_', '-'), r.return, r.infections])];
  const best = Math.max(...rows.map((r) => r[1]));
  const averted = ref.do_nothing ? ref.do_nothing.infections - msg.infections : null;
  const ending = msg.termination === 'time' ? 'day 30 reached' : msg.termination === 'stranded' ? `van stranded on day ${msg.days.toFixed(1)}` : `outbreak escaped on day ${msg.days.toFixed(1)}`;
  $('s-title').textContent = ending;
  $('s-body').innerHTML = `<table><tr><th>Policy</th><th>Return</th><th>Infections</th></tr>${rows.map(([n, r, i]) =>
    `<tr><td>${n}</td><td class="${r === best ? 'win' : ''}">${r.toFixed(1)}</td><td>${Math.round(i)}</td></tr>`).join('')}</table>`
    + (averted !== null ? `<p>Infections averted vs doing nothing: <b>${Math.round(averted)}</b> (${((averted / ref.do_nothing.infections) * 100).toFixed(0)}%). `
      + `${msg.screens} screenings, ${msg.nets_placed} net distributions, ${msg.km.toFixed(0)} km.</p>` : '');
  $('summary').hidden = false;
  status('season over');
}

function status(text) { $('status-badge').textContent = text; }

// ------------------------------------------------------------------ connection and controls
function send(msg) { if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(msg)); }

function connect() {
  const q = new URLSearchParams();
  for (const k of ['algo', 'seed', 'scenario', 'speed']) if (params.get(k)) q.set(k, params.get(k));
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  S.ws = new WebSocket(`${proto}://${location.host}/ws/sim?${q}`);
  S.ws.onopen = () => status('running');
  S.ws.onclose = () => status('disconnected');
  S.ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'episode') {
      S.history = [];
      $('log').innerHTML = '';
      $('summary').hidden = true;
      $('model-badge').textContent = `${msg.model} · ${msg.policy_mode === 'sample' ? 'sampled policy' : 'greedy policy'} · seed ${msg.seed}`;
      status(S.paused ? 'paused' : 'running');
      applyFrame(msg.start, true);
    } else if (msg.type === 'frame') {
      applyFrame(msg);
    } else if (msg.type === 'end') {
      showSummary(msg);
    } else if (msg.type === 'error') {
      status(`error: ${msg.message}`);
    }
  };
}

function setSpeed(v) {
  S.speed = Math.min(20, Math.max(0.25, v));
  $('speed').textContent = `${S.speed.toFixed(S.speed < 1 ? 2 : 1).replace(/\.0+$/, '')}×`;
  send({ cmd: 'speed', value: S.speed });
}

window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  if (key === 'h') setView(S.view === 'agent' ? 'reality' : 'agent');
  else if (key === ' ') { e.preventDefault(); S.paused = !S.paused; send({ cmd: S.paused ? 'pause' : 'resume' }); status(S.paused ? 'paused' : 'running'); }
  else if (key === '+' || key === '=') setSpeed(S.speed * 1.5);
  else if (key === '-' || key === '_') setSpeed(S.speed / 1.5);
  else if (key === 'n') { $('summary').hidden = true; send({ cmd: 'next' }); }
  else if (key === 'f') S.follow = !S.follow;
});
$('s-next').addEventListener('click', () => { $('summary').hidden = true; send({ cmd: 'next' }); });
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labels.setSize(window.innerWidth, window.innerHeight);
  drawCharts();
});

// ------------------------------------------------------------------ animation loop
const clock = new THREE.Clock();
const tmpColour = new THREE.Color();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  const now = performance.now();
  const t = now / 1000;
  if (S.data) {
    S.night = lerp(S.night, S.nightTarget, clamp01(dt * 1.5));
    S.rain = lerp(S.rain, S.rainTarget, clamp01(dt * 2));
    const sky = SKY.day.clone().lerp(SKY.night, S.night).lerp(SKY.storm, S.rain * 0.45 * (1 - S.night));
    scene.background.copy(sky);
    scene.fog.color.copy(sky);
    sun.intensity = lerp(2.4, 0.18, S.night) * (1 - 0.45 * S.rain);
    sun.color.setHex(S.night > 0.5 ? 0x9fb7ff : 0xfff0d2);
    hemi.intensity = lerp(0.95, 0.3, S.night);

    const pos = S.rainPoints.geometry.attributes.position;
    for (let i = 0; i < S.rainN; i++) {
      let y = pos.getY(i) - dt * 24;
      if (y < 0) y += 20;
      pos.setY(i, y);
    }
    pos.needsUpdate = true;
    S.rainPoints.geometry.setDrawRange(0, S.rain > 0.08 ? Math.floor(S.rainN * S.rain) : 0);

    for (const r of S.roads) {
      r.flooded = lerp(r.flooded, r.floodedTarget || 0, clamp01(dt * 2));
      r.highlight = Math.max(0, r.highlight - dt * 0.5);
      r.mesh.material.color.setHex(0xb5793f).lerp(tmpColour.setHex(0x3f78b0), r.flooded * 0.85);
      r.mesh.material.emissive.setHex(0xffd25a).multiplyScalar(r.highlight * 0.6);
    }
    for (const vil of S.villages) {
      const p = vil.pillar;
      const hNow = lerp(p.scale.y, vil.pillarTarget, clamp01(dt * 3));
      p.scale.y = hNow;
      p.position.y = vil.centre.y + hNow / 2;
      p.material.emissiveIntensity = 0.4 + 0.5 * clamp01(vil.pillarTarget / 5);
      vil.glow.material.color.lerp(vil.glowColour, clamp01(dt * 3));
      const domeOpacity = lerp(vil.dome.material.opacity, vil.domeTarget, clamp01(dt * 2));
      vil.dome.material.opacity = domeOpacity + (domeOpacity > 0.02 ? 0.04 * Math.sin(t * 3 + vil.loc.id) : 0);
      vil.dome.visible = domeOpacity > 0.01;
      const n = Math.floor(vil.swarmParams.length * vil.swarmTarget);
      const sp = vil.swarm.geometry.attributes.position;
      for (let i = 0; i < n; i++) {
        const q = vil.swarmParams[i];
        const a = q.a + t * q.w;
        sp.setXYZ(i, vil.centre.x + Math.cos(a) * q.r, vil.centre.y + q.h + 0.25 * Math.sin(t * 3 + i), vil.centre.z + Math.sin(a) * q.r);
      }
      sp.needsUpdate = true;
      vil.swarm.geometry.setDrawRange(0, n);
    }
    updateVan(now);
    updateEffects(now);
    if (S.follow) controls.target.lerp(S.van.position, clamp01(dt * 2));
  }
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
}

// ------------------------------------------------------------------ start
fetch('/scene')
  .then((r) => r.json())
  .then((data) => {
    buildWorld(data);
    setView('agent');
    setSpeed(S.speed);
    connect();
  })
  .catch((err) => status(`failed to load scene: ${err}`));
animate();
