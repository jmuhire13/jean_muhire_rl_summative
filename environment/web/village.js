// Aid Impact Simulator: 3D village drawn with three.js
//
// The Python server runs the environment. After every step it sends the
// state of the village here, and this file updates the 3D scene and the
// panels around it.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
const NUM_HOUSEHOLDS = 12;
const VILLAGE_RADIUS = 15;          // houses sit on a circle around the office
const ACTION_ICONS = ["👁", "💵", "🐐", "🛠", "➕", "🏦", "📋", "🎓"];

const COLORS = {
  grass: new THREE.Color("#7fae55"),
  dryGrass: new THREE.Color("#c8a466"),
  sky: new THREE.Color("#a9d4f2"),
  drySky: new THREE.Color("#e8c48f"),
  red: new THREE.Color("#e0524d"),
  amber: new THREE.Color("#e9a23b"),
  green: new THREE.Color("#4cbf6b"),
  gold: new THREE.Color("#ffd45e"),
  thatch: new THREE.Color("#9c6b3c"),
  path: new THREE.Color("#b98f5e"),
};

// ---------------------------------------------------------------------------
// Basic scene: renderer, camera, lights
// ---------------------------------------------------------------------------
const container = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = COLORS.sky.clone();
scene.fog = new THREE.Fog(COLORS.sky.clone(), 45, 110);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 300);
camera.position.set(0, 30, 38);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI / 2.2;   // do not go under the ground
controls.minDistance = 15;
controls.maxDistance = 80;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.25;
controls.addEventListener("start", () => { controls.autoRotate = false; });

// Soft light from the sky and a sun that casts shadows
const hemiLight = new THREE.HemisphereLight("#ffffff", "#6b8f45", 1.1);
scene.add(hemiLight);

const sun = new THREE.DirectionalLight("#fff3dc", 2.2);
sun.position.set(25, 40, 15);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -35;
sun.shadow.camera.right = 35;
sun.shadow.camera.top = 35;
sun.shadow.camera.bottom = -35;
scene.add(sun);

// ---------------------------------------------------------------------------
// Ground, paths and trees
// ---------------------------------------------------------------------------
const groundMaterial = new THREE.MeshStandardMaterial({ color: COLORS.grass.clone(), roughness: 1 });
const ground = new THREE.Mesh(new THREE.CircleGeometry(70, 64), groundMaterial);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const pathMaterial = new THREE.MeshStandardMaterial({ color: COLORS.path, roughness: 1 });
const ringPath = new THREE.Mesh(new THREE.RingGeometry(10, 11.3, 64), pathMaterial);
ringPath.rotation.x = -Math.PI / 2;
ringPath.position.y = 0.01;
ringPath.receiveShadow = true;
scene.add(ringPath);

// Small paths from the ring to each house
for (let i = 0; i < NUM_HOUSEHOLDS; i++) {
  const angle = (i / NUM_HOUSEHOLDS) * Math.PI * 2;
  const spoke = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 3), pathMaterial);
  spoke.rotation.x = -Math.PI / 2;
  spoke.rotation.z = -angle + Math.PI / 2;
  spoke.position.set(Math.cos(angle) * 12.3, 0.011, Math.sin(angle) * 12.3);
  scene.add(spoke);
}

// A simple random number generator with a seed, so trees are always in the same place
function seededRandom(seed) {
  let value = seed;
  return () => {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}

const random = seededRandom(7);
const leafMaterial = new THREE.MeshStandardMaterial({ color: "#4f8a3a", roughness: 0.9 });
const trunkMaterial = new THREE.MeshStandardMaterial({ color: "#6e4b2a" });
const treeLeaves = [];
for (let i = 0; i < 60; i++) {
  const angle = random() * Math.PI * 2;
  const distance = 22 + random() * 30;
  const tree = new THREE.Group();
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.22, 1.4), trunkMaterial);
  trunk.position.y = 0.7;
  const leaves = new THREE.Mesh(new THREE.SphereGeometry(0.9 + random() * 0.6, 10, 8), leafMaterial.clone());
  leaves.position.y = 1.9;
  leaves.scale.y = 0.8;
  trunk.castShadow = leaves.castShadow = true;
  tree.add(trunk, leaves);
  tree.position.set(Math.cos(angle) * distance, 0, Math.sin(angle) * distance);
  scene.add(tree);
  treeLeaves.push(leaves);
}

// ---------------------------------------------------------------------------
// NGO office in the middle of the village
// ---------------------------------------------------------------------------
function buildOffice() {
  const office = new THREE.Group();
  const walls = new THREE.Mesh(
    new THREE.BoxGeometry(4.5, 2.4, 3.2),
    new THREE.MeshStandardMaterial({ color: "#f4f1ea" })
  );
  walls.position.y = 1.2;
  const roof = new THREE.Mesh(
    new THREE.BoxGeometry(5, 0.35, 3.7),
    new THREE.MeshStandardMaterial({ color: "#2f6fb5" })
  );
  roof.position.y = 2.55;
  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.05, 0.05, 4),
    new THREE.MeshStandardMaterial({ color: "#cccccc" })
  );
  pole.position.set(2.8, 2, 0);
  const flag = new THREE.Mesh(
    new THREE.PlaneGeometry(1.2, 0.7),
    new THREE.MeshStandardMaterial({ color: "#4fa3ff", side: THREE.DoubleSide })
  );
  flag.position.set(3.4, 3.6, 0);
  walls.castShadow = roof.castShadow = pole.castShadow = true;
  walls.receiveShadow = true;
  office.add(walls, roof, pole, flag);
  office.userData.flag = flag;
  scene.add(office);
  return office;
}
const office = buildOffice();

// ---------------------------------------------------------------------------
// Households
// ---------------------------------------------------------------------------
function buildGoat() {
  const goat = new THREE.Group();
  const white = new THREE.MeshStandardMaterial({ color: "#f2efe6" });
  const dark = new THREE.MeshStandardMaterial({ color: "#3b3b3b" });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.38, 0.34), white);
  body.position.y = 0.5;
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.26, 0.24), white);
  head.position.set(0.45, 0.72, 0);
  goat.add(body, head);
  for (const [x, z] of [[-0.25, -0.12], [-0.25, 0.12], [0.25, -0.12], [0.25, 0.12]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.32, 0.08), dark);
    leg.position.set(x, 0.16, z);
    goat.add(leg);
  }
  goat.traverse((part) => { part.castShadow = true; });
  return goat;
}

function buildHouse(index) {
  const angle = (index / NUM_HOUSEHOLDS) * Math.PI * 2;
  const group = new THREE.Group();
  group.position.set(Math.cos(angle) * VILLAGE_RADIUS, 0, Math.sin(angle) * VILLAGE_RADIUS);
  group.lookAt(0, 0, 0);  // the door faces the office

  // Coloured ring on the ground shows the household status
  const ringMaterial = new THREE.MeshBasicMaterial({ color: COLORS.amber, transparent: true, opacity: 0.55 });
  const ring = new THREE.Mesh(new THREE.RingGeometry(2.2, 2.9, 40), ringMaterial);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.03;

  // Walls change colour from red (crisis) to green (resilient)
  const wallMaterial = new THREE.MeshStandardMaterial({ color: COLORS.amber.clone(), roughness: 0.8 });
  const walls = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.6, 2.4), wallMaterial);
  walls.position.y = 0.8;
  walls.castShadow = walls.receiveShadow = true;
  walls.userData.household = index;

  const roof = new THREE.Mesh(
    new THREE.ConeGeometry(2.15, 1.4, 4),
    new THREE.MeshStandardMaterial({ color: COLORS.thatch, roughness: 1 })
  );
  roof.position.y = 2.3;
  roof.rotation.y = Math.PI / 4;
  roof.castShadow = true;
  roof.userData.household = index;

  const door = new THREE.Mesh(
    new THREE.BoxGeometry(0.55, 0.95, 0.06),
    new THREE.MeshStandardMaterial({ color: "#5a3a1e" })
  );
  door.position.set(0, 0.48, 1.22);

  // Productive asset: a goat
  const goat = buildGoat();
  goat.position.set(2.2, 0, 0.6);
  goat.rotation.y = -0.6;

  // Field behind the house: crops grow with income
  const crops = new THREE.Group();
  const cropMaterial = new THREE.MeshStandardMaterial({ color: "#9cc94f" });
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 4; col++) {
      const plant = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.8, 6), cropMaterial);
      plant.position.set(-0.9 + col * 0.6, 0.4, -2.2 - row * 0.55);
      plant.castShadow = true;
      crops.add(plant);
    }
  }
  const soil = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 1.8), new THREE.MeshStandardMaterial({ color: "#7a5532" }));
  soil.rotation.x = -Math.PI / 2;
  soil.position.set(0, 0.02, -2.75);
  soil.receiveShadow = true;

  // Savings: a stack of coins
  const coins = new THREE.Group();
  const coinMaterial = new THREE.MeshStandardMaterial({ color: COLORS.gold, metalness: 0.6, roughness: 0.3 });
  for (let i = 0; i < 6; i++) {
    const coin = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.08, 16), coinMaterial);
    coin.position.y = 0.05 + i * 0.09;
    coin.castShadow = true;
    coins.add(coin);
  }
  coins.position.set(-1.8, 0, 1.2);

  // Health cover: a small sign with a red cross
  const health = new THREE.Group();
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1), new THREE.MeshStandardMaterial({ color: "#dddddd" }));
  post.position.y = 0.5;
  const board = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.05), new THREE.MeshStandardMaterial({ color: "#ffffff" }));
  board.position.y = 1.1;
  const crossMaterial = new THREE.MeshStandardMaterial({ color: "#e0524d" });
  const crossA = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.1, 0.06), crossMaterial);
  const crossB = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.34, 0.06), crossMaterial);
  crossA.position.y = crossB.position.y = 1.1;
  crossA.position.z = crossB.position.z = 0.01;
  health.add(post, board, crossA, crossB);
  health.position.set(-1.7, 0, -0.9);

  // Skills training: a blue toolbox
  const toolbox = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.3, 0.3), new THREE.MeshStandardMaterial({ color: "#2f6fb5" }));
  toolbox.position.set(1.7, 0.15, -1.2);
  toolbox.castShadow = true;

  // Graduation: a golden star floating above the house
  const star = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.45),
    new THREE.MeshStandardMaterial({ color: COLORS.gold, emissive: COLORS.gold, emissiveIntensity: 0.6, metalness: 0.5 })
  );
  star.position.y = 3.7;

  // Illness: a purple pulse that grows and fades
  const pulse = new THREE.Mesh(
    new THREE.SphereGeometry(1, 20, 16),
    new THREE.MeshBasicMaterial({ color: "#a066ff", transparent: true, opacity: 0 })
  );
  pulse.position.y = 1;

  group.add(ring, walls, roof, door, goat, crops, soil, coins, health, toolbox, star, pulse);
  scene.add(group);

  // HTML label with the household number
  const label = document.createElement("div");
  label.className = "house-label";
  label.textContent = `H${index + 1}`;
  document.getElementById("labels").appendChild(label);

  return {
    index, group, ring, walls, roof, goat, crops, coins, health, toolbox, star, pulse, label,
    // values shown on screen, moved smoothly toward the latest data
    shown: { lasting: 0.35, income: 0.1 },
    target: { lasting: 0.35, income: 0.1 },
    data: null,
    pulseTime: 0,
  };
}

const houses = [];
for (let i = 0; i < NUM_HOUSEHOLDS; i++) houses.push(buildHouse(i));

// ---------------------------------------------------------------------------
// Program officer who walks to the household under review
// ---------------------------------------------------------------------------
function buildOfficer() {
  const officer = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.36, 1.1, 14), new THREE.MeshStandardMaterial({ color: "#2f6fb5" }));
  body.position.y = 0.75;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.26, 16, 12), new THREE.MeshStandardMaterial({ color: "#8d5a3b" }));
  head.position.y = 1.55;
  const clipboard = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.38, 0.04), new THREE.MeshStandardMaterial({ color: "#f4f1ea" }));
  clipboard.position.set(0.32, 0.95, 0.2);
  body.castShadow = head.castShadow = true;
  officer.add(body, head, clipboard);
  scene.add(officer);
  return officer;
}
const officer = buildOfficer();
const officerTarget = new THREE.Vector3(0, 0, 2.5);

// Light beam marking the household under review
const beam = new THREE.Mesh(
  new THREE.CylinderGeometry(2.4, 2.4, 14, 40, 1, true),
  new THREE.MeshBasicMaterial({ color: "#7cc4ff", transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false })
);
beam.position.y = 7;
scene.add(beam);

// Floating bubble that shows the chosen action
const bubble = document.createElement("div");
bubble.className = "action-bubble";
bubble.style.opacity = 0;
document.getElementById("labels").appendChild(bubble);
let bubbleHousehold = 0;
let bubbleUntil = 0;

// ---------------------------------------------------------------------------
// Drought dust
// ---------------------------------------------------------------------------
const dustCount = 600;
const dustPositions = new Float32Array(dustCount * 3);
for (let i = 0; i < dustCount; i++) {
  dustPositions[i * 3] = (random() - 0.5) * 80;
  dustPositions[i * 3 + 1] = random() * 10;
  dustPositions[i * 3 + 2] = (random() - 0.5) * 80;
}
const dustGeometry = new THREE.BufferGeometry();
dustGeometry.setAttribute("position", new THREE.BufferAttribute(dustPositions, 3));
const dust = new THREE.Points(dustGeometry, new THREE.PointsMaterial({ color: "#d9b37a", size: 0.25, transparent: true, opacity: 0 }));
scene.add(dust);
let droughtLevel = 0;      // shown on screen, from 0 to 1
let droughtTarget = 0;

// ---------------------------------------------------------------------------
// Turning a lasting score into a colour
// ---------------------------------------------------------------------------
function scoreColor(score, inCrisis) {
  if (inCrisis) return COLORS.red.clone();
  const t = THREE.MathUtils.clamp((score - 0.2) / 0.35, 0, 1);
  if (t < 0.5) return COLORS.red.clone().lerp(COLORS.amber, t * 2);
  return COLORS.amber.clone().lerp(COLORS.green, (t - 0.5) * 2);
}

// ---------------------------------------------------------------------------
// Applying new data from the server to the 3D scene
// ---------------------------------------------------------------------------
let currentHousehold = 0;
let selectedHousehold = null;   // null means "follow the household under review"
let lastSnapshot = null;
const history = { lasting: [], crisis: [] };

function applySnapshot(snap) {
  lastSnapshot = snap;
  currentHousehold = snap.current_household;

  for (const house of houses) {
    const d = snap.households[house.index];
    house.data = d;
    house.target.lasting = d.lasting_score;
    house.target.income = d.income;

    house.goat.visible = d.has_asset;
    house.toolbox.visible = d.has_training;
    house.health.visible = d.has_health_cover;
    house.star.visible = d.graduated;

    // Number of coins follows savings
    const coinCount = Math.round(d.savings * 6);
    house.coins.children.forEach((coin, i) => { coin.visible = i < coinCount; });

    // Ring colour: gold if graduated, red if in crisis, otherwise by score
    const ringColor = d.graduated ? COLORS.gold : scoreColor(d.lasting_score, d.in_crisis);
    house.ring.material.color.copy(ringColor);
  }

  // Illness events start a purple pulse on that house
  for (const event of snap.events) {
    const match = event.match(/Illness in household (\d+)/);
    if (match) houses[Number(match[1]) - 1].pulseTime = 1.0;
  }

  droughtTarget = snap.drought ? 1 : 0;

  // The officer walks to the reviewed household (or the next one at reset)
  const reviewed = snap.reviewed_household ?? snap.current_household;
  const angle = (reviewed / NUM_HOUSEHOLDS) * Math.PI * 2;
  officerTarget.set(Math.cos(angle) * (VILLAGE_RADIUS - 3.3), 0, Math.sin(angle) * (VILLAGE_RADIUS - 3.3));

  if (snap.chosen_action !== undefined) {
    bubbleHousehold = reviewed;
    bubble.textContent = `${ACTION_ICONS[snap.chosen_action]} ${snap.last_action_name}`;
    bubbleUntil = performance.now() + 1600;
  }

  updatePanels(snap);
}

// ---------------------------------------------------------------------------
// Panels around the scene
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function formatSigned(value) {
  return (value >= 0 ? "+" : "") + value.toFixed(2);
}

function updatePanels(snap) {
  $("week-text").textContent = `${snap.week} / ${snap.program_weeks}`;
  $("week-bar").style.width = `${(snap.week / snap.program_weeks) * 100}%`;
  $("budget-text").textContent = `${snap.budget.toFixed(1)} / ${snap.start_budget}`;
  $("budget-bar").style.width = `${(snap.budget / snap.start_budget) * 100}%`;
  $("total-reward").textContent = snap.total_reward.toFixed(2);
  $("step-reward").textContent = formatSigned(snap.reward);
  $("step-reward").className = "stat-value " + (snap.reward >= 0 ? "good" : "bad");
  $("lasting").textContent = snap.mean_lasting_score.toFixed(2);
  $("crisis").textContent = snap.households_in_crisis;
  $("graduations").textContent = snap.successful_graduations;
  $("relapses").textContent = snap.relapses;

  // Decision card
  if (snap.chosen_action !== undefined) {
    const h = snap.reviewed_household + 1;
    $("action-text").textContent = `${ACTION_ICONS[snap.chosen_action]}  Household ${h}: ${snap.last_action_name}`;
    $("action-sub").textContent = `week ${snap.week} · reward ${formatSigned(snap.reward)}${snap.drought ? " · drought" : ""}`;
    const card = $("action-card");
    card.classList.remove("flash");
    void card.offsetWidth;  // restart the flash animation
    card.classList.add("flash");
  } else {
    $("action-text").textContent = "New program ready. Press Play.";
    $("action-sub").textContent = "";
  }
  if (snap.done) {
    $("action-text").textContent = "Program finished";
    $("action-sub").textContent =
      `total reward ${snap.total_reward.toFixed(2)} · graduations held ${snap.successful_graduations} · in crisis ${snap.households_in_crisis}`;
  }

  // Chart history
  if (snap.week === 0) { history.lasting = []; history.crisis = []; }
  history.lasting[snap.week] = snap.mean_lasting_score;
  history.crisis[snap.week] = snap.households_in_crisis / NUM_HOUSEHOLDS;
  drawChart(snap.program_weeks);

  // Event log
  const log = $("event-log");
  for (const event of snap.events) {
    const item = document.createElement("li");
    // Illness happens often (each family can fall sick), so it is shown small and grey.
    // The important events (relapse, drought, graduation, refused action) stand out.
    let kind = "";
    if (/Illness/.test(event)) kind = "minor";
    else if (/relapsed|refused/.test(event)) kind = "bad";
    else if (/Drought/.test(event)) kind = "warn";
    else if (/graduat/.test(event)) kind = "good";
    item.className = kind;
    item.innerHTML = `<span class="wk">W${snap.week}</span>${event}`;
    log.prepend(item);
  }
  while (log.children.length > 14) log.removeChild(log.lastChild);

  updateHouseholdPanel();
}

function drawChart(totalWeeks) {
  const canvas = $("chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // light grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath();
    ctx.moveTo(0, (h * i) / 4);
    ctx.lineTo(w, (h * i) / 4);
    ctx.stroke();
  }

  function drawLine(values, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    let started = false;
    values.forEach((value, week) => {
      if (value === undefined) return;
      const x = (week / totalWeeks) * w;
      const y = h - value * h * 0.95 - 3;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  drawLine(history.crisis, "#ff6b6b");
  drawLine(history.lasting, "#5fd08a");
}

function miniBar(value, color) {
  const pct = Math.round(value * 100);
  return `<div class="mini"><i style="width:${pct}%;background:${color}"></i><b>${value.toFixed(2)}</b></div>`;
}

function updateHouseholdPanel() {
  if (!lastSnapshot) return;
  const index = selectedHousehold ?? currentHousehold;
  const d = lastSnapshot.households[index];

  $("hh-title").textContent = `${index + 1}${selectedHousehold === null ? "  (under review next)" : ""}`;
  const status = $("hh-status");
  if (d.graduated) { status.textContent = "Graduated"; status.className = "badge graduated"; }
  else if (d.in_crisis) { status.textContent = "Food crisis"; status.className = "badge crisis"; }
  else { status.textContent = "In program"; status.className = "badge"; }

  const rows = [
    ["Food", d.food, d.record_food, "#e9a23b"],
    ["Income", d.income, d.record_income, "#4fa3ff"],
    ["Savings", d.savings, d.record_savings, "#ffd45e"],
    ["Health", d.health, d.record_health, "#ff7d9c"],
  ];
  let html = rows.map(([name, real, record, color]) =>
    `<div class="hh-row"><span class="name">${name}</span>${miniBar(real, color)}${miniBar(record, color + "99")}</div>`
  ).join("");
  html += `<div class="hh-row"><span class="name">Dependency</span>${miniBar(d.dependency, "#b07cff")}<span class="name">record age ${d.weeks_since_visit} wk</span></div>`;
  html += `<div class="hh-row"><span class="name">Resilience</span>${miniBar(d.resilience, "#5fd08a")}<span></span></div>`;
  $("hh-rows").innerHTML = html;

  const chips = [
    ["Asset", d.has_asset], ["Asset paying", d.asset_ready], ["Training", d.has_training],
    ["Savings group", d.in_savings_group], ["Health cover", d.has_health_cover],
  ];
  $("hh-support").innerHTML = chips.map(([name, on]) => `<span class="chip ${on ? "on" : ""}">${name}</span>`).join("");
}

// Click a house to inspect it, click the ground to go back to following the review
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener("click", (event) => {
  pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const targets = houses.flatMap((house) => [house.walls, house.roof]);
  const hit = raycaster.intersectObjects(targets)[0];
  selectedHousehold = hit ? hit.object.userData.household : null;
  updateHouseholdPanel();
});

// ---------------------------------------------------------------------------
// Animation loop: runs every frame
// ---------------------------------------------------------------------------
const clock = new THREE.Clock();
const screenPosition = new THREE.Vector3();

function placeLabel(element, worldPosition) {
  screenPosition.copy(worldPosition).project(camera);
  const visible = screenPosition.z < 1;
  element.style.display = visible ? "block" : "none";
  element.style.left = `${(screenPosition.x * 0.5 + 0.5) * window.innerWidth}px`;
  element.style.top = `${(-screenPosition.y * 0.5 + 0.5) * window.innerHeight}px`;
}

function animate() {
  const delta = Math.min(clock.getDelta(), 0.1);
  const time = clock.elapsedTime;
  const smooth = 1 - Math.exp(-delta * 5);   // how fast values move toward their target

  for (const house of houses) {
    house.shown.lasting += (house.target.lasting - house.shown.lasting) * smooth;
    house.shown.income += (house.target.income - house.shown.income) * smooth;

    const inCrisis = house.data?.in_crisis ?? false;
    house.walls.material.color.copy(scoreColor(house.shown.lasting, inCrisis));

    // Crops grow taller with income
    const height = 0.25 + house.shown.income * 1.5;
    house.crops.children.forEach((plant) => {
      plant.scale.y = height;
      plant.position.y = 0.4 * height;
    });

    // Crisis: the ring blinks. Graduated: the star turns.
    house.ring.material.opacity = inCrisis ? 0.45 + 0.4 * Math.sin(time * 6) : 0.55;
    house.star.rotation.y += delta * 1.5;
    house.star.position.y = 3.7 + Math.sin(time * 2 + house.index) * 0.15;

    // Illness pulse grows and fades away
    if (house.pulseTime > 0) {
      house.pulseTime -= delta * 0.8;
      const t = 1 - house.pulseTime;
      house.pulse.scale.setScalar(1 + t * 2.5);
      house.pulse.material.opacity = 0.45 * Math.max(house.pulseTime, 0);
    } else {
      house.pulse.material.opacity = 0;
    }

    // Label above each house
    const labelPos = house.group.position.clone();
    labelPos.y = 4.6;
    house.label.classList.toggle("current", house.index === currentHousehold);
    placeLabel(house.label, labelPos);
  }

  // Officer walks smoothly and bobs a little while moving
  const distance = officer.position.distanceTo(officerTarget);
  officer.position.lerp(officerTarget, smooth * 1.4);
  officer.position.y = distance > 0.2 ? Math.abs(Math.sin(time * 10)) * 0.12 : 0;
  officer.lookAt(officerTarget.x * 2, 0, officerTarget.z * 2);

  // Beam over the household that will be reviewed next
  const next = houses[currentHousehold].group.position;
  beam.position.x += (next.x - beam.position.x) * smooth;
  beam.position.z += (next.z - beam.position.z) * smooth;
  beam.material.opacity = 0.10 + 0.05 * Math.sin(time * 3);

  // Action bubble above the reviewed house
  const bubblePos = houses[bubbleHousehold].group.position.clone();
  bubblePos.y = 5.6;
  placeLabel(bubble, bubblePos);
  bubble.style.opacity = performance.now() < bubbleUntil ? 1 : 0;

  // Drought: dry ground, warm sky and floating dust
  droughtLevel += (droughtTarget - droughtLevel) * smooth * 0.6;
  groundMaterial.color.copy(COLORS.grass).lerp(COLORS.dryGrass, droughtLevel);
  scene.background.copy(COLORS.sky).lerp(COLORS.drySky, droughtLevel);
  scene.fog.color.copy(scene.background);
  sun.color.set("#fff3dc").lerp(new THREE.Color("#ffb36b"), droughtLevel);
  dust.material.opacity = 0.7 * droughtLevel;
  dust.rotation.y += delta * 0.05;
  for (const leaves of treeLeaves) {
    leaves.material.color.set("#4f8a3a").lerp(new THREE.Color("#9b8a45"), droughtLevel);
  }

  // NGO flag waves
  office.userData.flag.rotation.y = Math.sin(time * 2) * 0.25;

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------------------
// Talking to the Python server
// ---------------------------------------------------------------------------
const socket = new WebSocket(`ws://${location.host}/ws`);
let waitingForReply = null;    // resolves when the next snapshot arrives
let playing = false;
let finished = false;

socket.addEventListener("open", async () => {
  $("connection").classList.add("ok");
  const info = await (await fetch("/api/info")).json();
  const select = $("policy-select");
  for (const name of info.policies) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    if (name === info.default_policy) option.selected = true;
    select.appendChild(option);
  }
  await restart();

  // Optional settings in the address, for example /?autoplay=1&speed=8
  const params = new URLSearchParams(location.search);
  if (params.has("speed")) $("speed-input").value = params.get("speed");
  if (params.get("autoplay") === "1") $("play-btn").click();
});

socket.addEventListener("close", () => {
  $("connection").classList.remove("ok");
  $("connection").textContent = "Disconnected. Is the Python server still running?";
  playing = false;
});

socket.addEventListener("message", (event) => {
  const snap = JSON.parse(event.data);
  finished = snap.done;
  applySnapshot(snap);
  if (waitingForReply) {
    waitingForReply();
    waitingForReply = null;
  }
});

function send(message) {
  return new Promise((resolve) => {
    waitingForReply = resolve;
    socket.send(JSON.stringify(message));
  });
}

function restart() {
  playing = false;
  $("play-btn").textContent = "Play";
  $("event-log").innerHTML = "";
  selectedHousehold = null;
  return send({ type: "reset", policy: $("policy-select").value, seed: Number($("seed-input").value) });
}

function stepDelay() {
  // speed 1 is slow (1 second per week), speed 10 is fast (0.1 seconds per week)
  return 1100 - Number($("speed-input").value) * 100;
}

async function playLoop() {
  while (playing && !finished) {
    await send({ type: "step" });
    await new Promise((resolve) => setTimeout(resolve, stepDelay()));
  }
  playing = false;
  $("play-btn").textContent = "Play";
}

$("play-btn").addEventListener("click", () => {
  if (finished) return;
  playing = !playing;
  $("play-btn").textContent = playing ? "Pause" : "Play";
  if (playing) playLoop();
});

$("step-btn").addEventListener("click", () => {
  if (!playing && !finished) send({ type: "step" });
});

$("restart-btn").addEventListener("click", restart);
$("policy-select").addEventListener("change", restart);
