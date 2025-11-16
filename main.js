import {
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  Scene,
  AmbientLight,
  DirectionalLight,
  GridHelper,
  AxesHelper,
  Raycaster,
  Vector2,
  Object3D,
  Bone,
  Vector3,
  Quaternion,
  Color,
  Group,
  Mesh,
  SphereGeometry,
  MeshStandardMaterial,
  BoxGeometry,
  Euler,
  MathUtils,
  Matrix4,
  Box3,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { SkeletonHelper } from './customSkeletonHelper.js';

const errorToast = (text, isError) =>
  Toastify({
    text: text,
    duration: 3000,
    close: true,
    gravity: "top", // `top` or `bottom`
    position: "left", // `left`, `center` or `right`
    stopOnFocus: true, // Prevents dismissing of toast on hover
    style: {
      background: isError ? "#ff5353" : "#53b3ff",
    },
    onClick: function () { }, // Callback after click
  }).showToast();

// ---------- Utilities ----------
const DEG = Math.PI / 180,
  RAD = 180 / Math.PI;
const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const byId = (id) => document.getElementById(id);
const setStatus = (t) => (byId("status").textContent = t);

// ---------- Scene ----------
const viewportEl = byId("viewport");
const renderer = new WebGLRenderer({
  antialias: true,
  alpha: true,
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(viewportEl.clientWidth, viewportEl.clientHeight);
renderer.setClearColor(0x000000, 0); // transparent to show reference image
viewportEl.appendChild(renderer.domElement);

// Perspective & Ortho cameras
const persp = new PerspectiveCamera(
  45,
  viewportEl.clientWidth / viewportEl.clientHeight,
  0.01,
  100
);
persp.position.set(2.2, 1.8, 2.6);

const orthoSize = 2.2;
const ortho = new OrthographicCamera(
  -orthoSize,
  orthoSize,
  orthoSize,
  -orthoSize,
  -10,
  100
);
ortho.position.set(0, 1.7, 3);
ortho.lookAt(0, 1, 0);

let camera = persp;
let needsRender = true;

const scene = new Scene();

// Lights
scene.add(new AmbientLight(0xffffff, 0.6));
const key = new DirectionalLight(0xffffff, 0.9);
key.position.set(3, 5, 3);
scene.add(key);
const rim = new DirectionalLight(0xffffff, 0.3);
rim.position.set(-3, 3, -3);
scene.add(rim);

// Grid & axes
const grid = new GridHelper(5, 20, 0x26406d, 0x1c2947);
grid.position.y = 0;
scene.add(grid);
let gridVisible = true;

const axes = new AxesHelper(0.25);
axes.position.set(0, 0, 0);
scene.add(axes);

// Orbit + Transform controls
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.target.set(0, 1.0, 0);

const tctrl = new TransformControls(camera, renderer.domElement);
tctrl.setMode("rotate");
tctrl.setSize(0.76);
tctrl.addEventListener("dragging-changed", (e) => (orbit.enabled = !e.value));
scene.add(tctrl.getHelper());

// Repaint whenever controls change camera or gizmo state
orbit.addEventListener("change", () => markPoseDirty());
tctrl.addEventListener("change", () => markPoseDirty());
renderer.domElement.addEventListener("wheel", () => markPoseDirty(), {
  passive: true,
});
renderer.domElement.style.touchAction = "none"; // better touch behavior

// Picking
const raycaster = new Raycaster();
const mouse = new Vector2();
renderer.domElement.addEventListener("pointerdown", onPointerDown);

// ---------- Skeleton definition ----------
/**
 * We model an OpenPose-like skeleton using Bone. All bones use local +Y as the "bone axis",
 * which makes IK simpler (we align local +Y to the child direction). Left/right bones are mirrored along world X.
 * Units are meters. Default pose is relaxed A-pose.
 */
const boneSpec = [
  // name, parent, [x,y,z]
  ["Root", null, [0, 1.0, 0]],
  // pelvis & spine
  ["Pelvis", "Root", [0, 0.05, 0]],
  ["Spine1", "Pelvis", [0, 0.12, 0]],
  ["Spine2", "Spine1", [0, 0.12, 0]],
  ["Neck", "Spine2", [0, 0.12, 0]],
  ["Head", "Neck", [0, 0.18, 0]],
  // clavicles / shoulders
  ["LeftClavicle", "Spine2", [0.12, 0.08, 0]],
  ["LeftShoulder", "LeftClavicle", [0.13, 0.0, 0]],
  ["LeftElbow", "LeftShoulder", [0.0, -0.3, 0]],
  ["LeftWrist", "LeftElbow", [0.0, -0.26, 0]],
  ["LeftHand", "LeftWrist", [0.0, -0.1, 0]],
  ["RightClavicle", "Spine2", [-0.12, 0.08, 0]],
  ["RightShoulder", "RightClavicle", [-0.13, 0.0, 0]],
  ["RightElbow", "RightShoulder", [0.0, -0.3, 0]],
  ["RightWrist", "RightElbow", [0.0, -0.26, 0]],
  ["RightHand", "RightWrist", [0.0, -0.1, 0]],
  // hips / legs
  ["LeftHip", "Pelvis", [0.1, -0.19, 0]],
  ["LeftKnee", "LeftHip", [0.0, -0.38, 0]],
  ["LeftAnkle", "LeftKnee", [0.0, -0.38, 0]],
  ["LeftFoot", "LeftAnkle", [0.08, 0.0, 0.12]], // small forward foot
  ["RightHip", "Pelvis", [-0.1, -0.19, 0]],
  ["RightKnee", "RightHip", [0.0, -0.38, 0]],
  ["RightAnkle", "RightKnee", [0.0, -0.38, 0]],
  ["RightFoot", "RightAnkle", [-0.08, 0.0, 0.12]],
];

const jointLimits = {
  // Cervical complex split across two bones (Neck does the heavy lifting; Head adds fine motion)
  Neck: { x: [-60, 45], y: [-80, 80], z: [-45, 45] }, // flex 45, ext 60, rot 80, side 45
  Head: { x: [-25, 25], y: [-40, 40], z: [-25, 25] },

  // Thoracic/Lumbar distributed over two segments
  Spine1: { x: [-30, 20], y: [-20, 20], z: [-15, 15] }, // per segment
  Spine2: { x: [-30, 20], y: [-25, 25], z: [-15, 15] },

  // Optional pelvis rotation (small but useful for posing). If you prefer none, remove this entry.
  Pelvis: { x: [-15, 15], y: [-30, 30], z: [-15, 15] },

  // Clavicle (SC/AC complex) - contributes to overhead reach
  LeftClavicle: { x: [-35, 20], y: [-25, 25], z: [-25, 25] },
  RightClavicle: { x: [-35, 20], y: [-25, 25], z: [-25, 25] },

  // Glenohumeral shoulder (use clavicle + shoulder together for full overhead)
  LeftShoulder: { x: [-140, 90], y: [-90, 90], z: [-90, 110] },
  RightShoulder: { x: [-140, 90], y: [-90, 90], z: [-90, 110] },

  // Elbow: primary hinge with tiny laxity; allow slight hyperextension for realism
  LeftElbow: { x: [-155, 5], y: [-10, 10], z: [-10, 10] },
  RightElbow: { x: [-155, 5], y: [-10, 10], z: [-10, 10] },

  // Wrist: flex/extend, radial/ulnar deviation, plus axial “twist” if you lack a forearm-twist bone
  LeftWrist: { x: [-70, 80], y: [-85, 85], z: [-15, 30] }, // y used for pron/sup surrogate
  RightWrist: { x: [-70, 80], y: [-85, 85], z: [-15, 30] },

  // Hip: ball-and-socket
  LeftHip: { x: [-140, 70], y: [-60, 45], z: [-30, 60] },
  RightHip: { x: [-140, 70], y: [-60, 45], z: [-60, 30] },

  // Knee: near-hinge; small axial/varus/valgus available (esp. when flexed)
  LeftKnee: { x: [-5, 155], y: [-10, 10], z: [-10, 10] },
  RightKnee: { x: [-5, 155], y: [-10, 10], z: [-10, 10] },

  // Ankle (talocrural) + subtalar approximation (more inversion than eversion)
  LeftAnkle: { x: [-50, 20], y: [-20, 20], z: [-10, 25] },
  RightAnkle: { x: [-50, 20], y: [-20, 20], z: [-10, 25] },

  // Include end effectors too (helpful if you rotate them directly)
  LeftFoot: { x: [-20, 20], y: [-20, 20], z: [-30, 30] },
  RightFoot: { x: [-20, 20], y: [-20, 20], z: [-30, 30] },
  LeftHand: { x: [-30, 30], y: [-45, 45], z: [-30, 30] },
  RightHand: { x: [-30, 30], y: [-45, 45], z: [-30, 30] },
};

// Build skeleton (Bones + helper spheres for picking)
const bonesByName = new Map();
const rootObj = new Object3D();
rootObj.name = "Armature";
scene.add(rootObj);

// Bone creation
for (const [name, parentName, pos] of boneSpec) {
  const b = new Bone();

  b.name = name;
  b.position.fromArray(pos);
  b.rotation.set(0, 0, 0, "XYZ");
  bonesByName.set(name, b);
  if (parentName) {
    bonesByName.get(parentName).add(b);
  } else {
    rootObj.add(b);
  }
}

// After building the skeleton and parenting bones:
for (const [name, b] of bonesByName) {
  // Default axis is +Y; if the bone has a child, use that offset direction.
  const childBone = b.children.find((c) => c.isBone);
  const axis = new Vector3(0, 1, 0);
  if (childBone) {
    axis.copy(childBone.position).normalize();
    if (axis.lengthSq() < 1e-9) axis.set(0, 1, 0);
  }
  b.userData.bindAxisLocal = axis; // e.g., arms/legs become (0,-1,0)
}

// --- Pole vector storage (parent-local so it moves with the torso/clavicle/pelvis) ---
const poleByLimb = {}; // limb -> { parent: Object3D, nLocal: Vector3 }

// Which bones form each limb is already defined by your chainForLimb(...)
function initPoleForLimb(limb) {
  const chain = chainForLimb(limb);
  const [b0, b1, b2] = chain.map((n) => bonesByName.get(n));
  if (!b0 || !b1 || !b2) return;

  // Make sure world matrices are current
  scene.updateMatrixWorld(true);

  const p0 = b0.getWorldPosition(new Vector3());
  const p1 = b1.getWorldPosition(new Vector3());
  const p2 = b2.getWorldPosition(new Vector3());

  // Bind-plane normal (points which way the elbow/knee "likes" to bend)
  const nWorld = new Vector3().crossVectors(
    new Vector3().subVectors(p1, p0),
    new Vector3().subVectors(p2, p1)
  );

  if (nWorld.lengthSq() < 1e-8) nWorld.set(0, 0, 1); // fallback "forward"
  nWorld.normalize();

  // Store it in the parent’s local space so it follows the torso/hips naturally
  const parent = b0.parent;
  const parentWQInv = parent
    .getWorldQuaternion(new Quaternion())
    .invert();
  const nLocal = nWorld.clone().applyQuaternion(parentWQInv);

  poleByLimb[limb] = { parent, nLocal };
}

// Initialize for all 4 limbs
["LeftArm", "RightArm", "LeftLeg", "RightLeg"].forEach(initPoleForLimb);

// Helper: recover the pole direction in WORLD space at solve time
function getPoleDirWorld(limb) {
  const rec = poleByLimb[limb];
  if (!rec) return new Vector3(0, 0, 1);
  const wq = rec.parent.getWorldQuaternion(new Quaternion());
  return rec.nLocal.clone().applyQuaternion(wq).normalize();
}

// Skeleton visual helper (lines)
const skelHelper = new SkeletonHelper(rootObj, {
  linewidth: 6,
  color: new Color(0xff0000)
});
scene.add(skelHelper);

// Joints as clickable spheres
const jointGroup = new Group();
jointGroup.name = "JointSpheres";
scene.add(jointGroup);

const jointSpheres = new Map();
for (const [name, b] of bonesByName) {
  const sphere = new Mesh(
    new SphereGeometry(0.018, 16, 12),
    new MeshStandardMaterial({
      color: name.includes("Left")
        ? 0x6be28c
        : name.includes("Right")
          ? 0xe26b9e
          : 0x8aa3ff,
      metalness: 0.1,
      roughness: 0.6,
    })
  );
  sphere.userData.isJoint = true;
  sphere.userData.boneName = name;
  // Attach to bone so it inherits transform
  b.add(sphere);
  jointSpheres.set(name, sphere);
}

function addIKHandle(endBoneName) {
  const m = new Mesh(
    new BoxGeometry(0.04, 0.04, 0.04),
    new MeshStandardMaterial({
      color: 0x2ed17a,
      emissive: 0x0,
      metalness: 0.2,
      roughness: 0.4,
    })
  );
  m.visible = false; // shown only in IK mode
  m.userData.ikFor = endBoneName;
  scene.add(m);
  // place initially at end-effector
  const eb = bonesByName.get(endBoneName);
  const p = eb.getWorldPosition(new Vector3());
  m.position.copy(p);
  return m;
}

// IK handles (green cubes) at hands & feet
const ikHandles = {
  LeftArm: addIKHandle("LeftWrist"),
  RightArm: addIKHandle("RightWrist"),
  LeftLeg: addIKHandle("LeftAnkle"),
  RightLeg: addIKHandle("RightAnkle"),
};

const bindLocalPos = new Map();
for (const [name, b] of bonesByName) bindLocalPos.set(name, b.position.clone());

// IK state (FK by default)
const ikMode = {
  LeftArm: "FK",
  RightArm: "FK",
  LeftLeg: "FK",
  RightLeg: "FK",
};

// Transform controls attach/detach
let selectedBone = null;
function selectBoneByName(name) {
  const b = bonesByName.get(name) || null;
  selectedBone = b;

  // update joint sphere highlight
  for (const [n, mesh] of jointSpheres) {
    mesh.material.emissive = new Color(n === name ? 0x3856ff : 0);
  }
  // Attach gizmo to the selected bone (unless null)
  if (b) {
    tctrl.attach(b);
    updateInspectorUI();
    setStatus(`Selected: ${name}`);
  } else {
    tctrl.detach();
    setStatus("No joint selected");
  }
  updateTreeSelection(name);
}

function alignMidToPole(p0, p1, p2, poleDirWorld) {
  const axis = new Vector3().subVectors(p2, p0);
  const len = axis.length();
  if (len < 1e-6) return p1.clone();
  axis.multiplyScalar(1 / len);

  // Desired plane normal = pole projected onto plane ⟂ axis
  const nDes = poleDirWorld
    .clone()
    .sub(axis.clone().multiplyScalar(poleDirWorld.dot(axis)));
  if (nDes.lengthSq() < 1e-10) return p1.clone();
  nDes.normalize();

  // Current bend normal from the triangle (p0, p1, p2)
  const v0 = new Vector3().subVectors(p1, p0);
  const v1 = new Vector3().subVectors(p2, p1);
  let nCur = new Vector3().crossVectors(v0, v1);
  if (nCur.lengthSq() < 1e-12) {
    // Place p1 on the pole side at the same axial distance & radius as current
    const to1 = new Vector3().subVectors(p1, p0);
    const axial = axis.clone().multiplyScalar(to1.dot(axis));
    const radial = to1.clone().sub(axial);
    const r = radial.length();
    if (r < 1e-9) return p1.clone();
    const nDes = poleDirWorld
      .clone()
      .sub(axis.clone().multiplyScalar(poleDirWorld.dot(axis)))
      .normalize();
    // Build an orthonormal basis (u = nDes × axis, v = nDes)
    const u = new Vector3().crossVectors(nDes, axis).normalize();
    const v = nDes; // already ⟂ axis
    const radialOnPole = u.multiplyScalar(radial.length()); // rotate to pole side
    return new Vector3().addVectors(p0, axial).add(radialOnPole);
  }
  nCur.sub(axis.clone().multiplyScalar(nCur.dot(axis))).normalize();

  // Signed angle from current to desired normal around axis
  const dot = MathUtils.clamp(nCur.dot(nDes), -1, 1);
  let ang = Math.acos(dot);
  const sgn =
    Math.sign(new Vector3().crossVectors(nCur, nDes).dot(axis)) || 1;
  ang *= sgn;

  // Rotate p1 around the axis (through p0)
  const q = new Quaternion().setFromAxisAngle(axis, ang);
  const p1r = new Vector3().subVectors(p1, p0).applyQuaternion(q).add(p0);
  return p1r;
}

// Transform controls mode + space
function setMode(mode) {
  tctrl.setMode(mode);
  updateToolbarMode(mode);
}
function updateToolbarMode(mode) {
  byId("mode-translate").dataset.state = mode === "translate" ? "active" : "";
  byId("mode-rotate").dataset.state = mode === "rotate" ? "active" : "";
}
function setSpace(space) {
  tctrl.setSpace(space);
  byId("space-local").dataset.state = space === "local" ? "active" : "";
  byId("space-world").dataset.state = space === "world" ? "active" : "";
}

// ---------- Hierarchy UI ----------
// Existing helper
// function byId(id) { ... }
// bonesByName: Map
// selectBoneByName(name): your existing function

function buildTree() {
  const treeEl = byId("tree");
  treeEl.innerHTML = "";

  function buildNode(name) {
    const b = bonesByName.get(name);
    const li = document.createElement("li");
    li.classList.add("tree-item");

    // --- Label container (icon + button) ---
    const label = document.createElement("div");
    label.classList.add("tree-item-label");

    // Collapse/expand icon
    const icon = document.createElement("span");
    icon.classList.add("tree-toggle");
    icon.textContent = ""; // set later depending on children

    // Your existing button
    const btn = document.createElement("button");
    btn.textContent = name;
    btn.onclick = () => selectBoneByName(name);
    btn.id = `tree-${name}`;
    btn.classList = "button"

    label.appendChild(icon);
    label.appendChild(btn);
    li.appendChild(label);

    // --- Children (if any) ---
    const kids = b.children.filter((c) => c.isBone);
    if (kids.length) {
      icon.textContent = "▼"; // collapsed by default

      const ul = document.createElement("ul");
      ul.classList.add("tree-children"); // start collapsed

      for (const k of kids) {
        ul.appendChild(buildNode(k.name));
      }
      li.appendChild(ul);
    } else {
      // No children => no toggle icon
      li.classList.add("tree-no-children");
    }

    return li;
  }

  const rootBone = bonesByName.get("Root");
  const ul = document.createElement("ul");
  ul.appendChild(buildNode(rootBone.name));
  treeEl.appendChild(ul);
}

// Your existing selection update stays the same
function updateTreeSelection(name) {
  for (const [n, _] of bonesByName) {
    const el = byId(`tree-${n}`);
    if (el) {
      el.classList.toggle("sel", n === name);
    }
  }
}

// --- NEW: one-time event delegation to handle collapsing ---
const treeContainer = byId("tree");
treeContainer.addEventListener("click", (e) => {
  // Only toggle when clicking on the icon, not the button
  const toggle = e.target.closest(".tree-toggle");
  if (!toggle) return;

  const li = toggle.closest("li");
  if (!li) return;

  const childrenContainer = li.querySelector(":scope > .tree-children");
  if (!childrenContainer) return; // leaf node

  const isCollapsed = childrenContainer.classList.contains("tree-collapsed");

  if (isCollapsed) {
    childrenContainer.classList.remove("tree-collapsed");
    toggle.textContent = "▼";
  } else {
    childrenContainer.classList.add("tree-collapsed");
    toggle.textContent = "▶";
  }
});

// Build initially
buildTree();

// ---------- Inspector ----------
const rotX = byId("rot-x"),
  rotY = byId("rot-y"),
  rotZ = byId("rot-z");
const posX = byId("pos-x"),
  posY = byId("pos-y"),
  posZ = byId("pos-z");

const toNum = (v) => (Number.isFinite(parseFloat(v)) ? parseFloat(v) : 0);

rotX.addEventListener(
  "change",
  () =>
    selectedBone &&
    applyEulerDeg(
      selectedBone,
      parseFloat(rotX.value),
      parseFloat(rotY.value),
      parseFloat(rotZ.value)
    )
);
rotY.addEventListener(
  "change",
  () =>
    selectedBone &&
    applyEulerDeg(
      selectedBone,
      parseFloat(rotX.value),
      parseFloat(rotY.value),
      parseFloat(rotZ.value)
    )
);
rotZ.addEventListener(
  "change",
  () =>
    selectedBone &&
    applyEulerDeg(
      selectedBone,
      parseFloat(rotX.value),
      parseFloat(rotY.value),
      parseFloat(rotZ.value)
    )
);
posX.addEventListener(
  "change",
  () =>
    selectedBone &&
    selectedBone.name === "Root" &&
    ((selectedBone.position.x = parseFloat(posX.value)), markPoseDirty())
);
posY.addEventListener(
  "change",
  () =>
    selectedBone &&
    selectedBone.name === "Root" &&
    ((selectedBone.position.y = parseFloat(posY.value)), markPoseDirty())
);
posZ.addEventListener(
  "change",
  () =>
    selectedBone &&
    selectedBone.name === "Root" &&
    ((selectedBone.position.z = parseFloat(posZ.value)), markPoseDirty())
);

function eulerDeg(bone) {
  const e = new Euler().setFromQuaternion(bone.quaternion, "XYZ");
  return { x: e.x * RAD, y: e.y * RAD, z: e.z * RAD };
}

function applyEulerDeg(bone, x, y, z, clampToLimits = true) {
  const lim = jointLimits[bone.name] || {};
  const cx = clampToLimits && lim.x ? clamp(x, lim.x[0], lim.x[1]) : x;
  const cy = clampToLimits && lim.y ? clamp(y, lim.y[0], lim.y[1]) : y;
  const cz = clampToLimits && lim.z ? clamp(z, lim.z[0], lim.z[1]) : z;
  const e = new Euler(cx * DEG, cy * DEG, cz * DEG, "XYZ");
  bone.quaternion.setFromEuler(e);
  if (symmetryEnabled) mirrorPartnerOnEdit(bone);
  markPoseDirty();
  updateInspectorUIValuesOnly(); // keep UI in sync after clamping
}

function updateInspectorUI() {
  const name = selectedBone ? selectedBone.name : "Root";
  byId("inspector-joint").textContent = name;
  const lim = jointLimits[name] || {};
  byId("lim-x").textContent = `limits: ${lim.x ? `${lim.x[0]}, ${lim.x[1]}` : "*"
    }`;
  byId("lim-y").textContent = `limits: ${lim.y ? `${lim.y[0]}, ${lim.y[1]}` : "*"
    }`;
  byId("lim-z").textContent = `limits: ${lim.z ? `${lim.z[0]}, ${lim.z[1]}` : "*"
    }`;
  // Position: only Root is editable
  const rootEditable = name === "Root";
  for (const el of [posX, posY, posZ]) {
    el.disabled = !rootEditable;
    el.title = rootEditable ? "" : "Only Root is translatable";
  }
  updateInspectorUIValuesOnly();
}

function updateInspectorUIValuesOnly() {
  if (!selectedBone) return;
  const e = eulerDeg(selectedBone);
  rotX.value = e.x.toFixed(1);
  rotY.value = e.y.toFixed(1);
  rotZ.value = e.z.toFixed(1);
  const p = selectedBone.position;
  posX.value = p.x.toFixed(2);
  posY.value = p.y.toFixed(2);
  posZ.value = p.z.toFixed(2);
}

queueMicrotask(() => {
  selectBoneByName("Root");
});

// ---------- Picking ----------
function onPointerDown(e) {
  // Avoid picking while dragging transform controls
  if (tctrl.dragging) return;
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const meshes = [...jointSpheres.values()];
  const hit = raycaster.intersectObjects(meshes, false)[0];
  if (hit) {
    selectBoneByName(hit.object.userData.boneName);
    // If this bone is an IK end effector and that limb is IK, attach tctrl to the handle instead
    const limb = limbFromEnd(selectedBone.name);
    if (limb && ikMode[limb] === "IK") {
      attachToIKHandle(limb);
    } else {
      tctrl.attach(selectedBone);
    }
  }
}

// ---------- Transform controls change -> clamp + sync ----------
tctrl.addEventListener("objectChange", () => {
  if (!tctrl.object) return;
  const isBone = tctrl.object.isBone === true;
  const isActiveTranslate = tctrl.mode === "translate"; // TransformControls exposes `.mode`
  if (isBone && isActiveTranslate && selectedBone.name !== "Root") {
    const bind = bindLocalPos.get(tctrl.object.name);
    if (bind) {
      tctrl.object.position.copy(bind);
    }
  }
  // If attached to IK handle, solve IK; else if attached to a bone, clamp to limits.
  if (tctrl.object.userData && tctrl.object.userData.isIKHandle) {
    const limb = tctrl.object.userData.limb;
    solveIKForLimb(limb);
    markPoseDirty();
    updateInspectorUIValuesOnly();
  } else if (selectedBone) {
    // Clamp bone to limits
    const lim = jointLimits[selectedBone.name];
    if (lim) {
      const e = eulerDeg(selectedBone);
      applyEulerDeg(selectedBone, e.x, e.y, e.z, true); // reapply to clamp
    }
    markPoseDirty();
    updateInspectorUIValuesOnly();
  }
});

// ---------- FK/IK ----------
function limbFromEnd(endName) {
  switch (endName) {
    case "LeftWrist":
      return "LeftArm";
    case "RightWrist":
      return "RightArm";
    case "LeftAnkle":
      return "LeftLeg";
    case "RightAnkle":
      return "RightLeg";
    default:
      return null;
  }
}
function chainForLimb(limb) {
  if (limb === "LeftArm") return ["LeftShoulder", "LeftElbow", "LeftWrist"];
  if (limb === "RightArm") return ["RightShoulder", "RightElbow", "RightWrist"];
  if (limb === "LeftLeg") return ["LeftHip", "LeftKnee", "LeftAnkle"];
  if (limb === "RightLeg") return ["RightHip", "RightKnee", "RightAnkle"];
  return [];
}
// Attach transform control to IK handle
function attachToIKHandle(limb) {
  const h = ikHandles[limb];
  if (!h) return;
  h.userData.isIKHandle = true;
  h.userData.limb = limb;
  tctrl.attach(h);
  setSpace("world");
  setMode("translate");
}

// Simple FABRIK solver (positions only), then orient bones so local +Y aims to child vector.
function solveIKForLimb(limb, fromMirror = false) {
  const names = chainForLimb(limb);
  if (names.length < 3) return;
  const bones = names.map((n) => bonesByName.get(n));
  const target = ikHandles[limb].position.clone();

  // Collect world positions and lengths
  const pts = bones.map((b) => b.getWorldPosition(new Vector3()));
  const lens = [];
  for (let i = 0; i < pts.length - 1; i++)
    lens.push(pts[i].distanceTo(pts[i + 1]));
  const totalLen = lens.reduce((a, b) => a + b, 0);
  const base = pts[0].clone();

  // If target too far, just stretch toward target
  const dist = base.distanceTo(target);
  const iters = 12;
  const eps = 1e-3;

  if (dist > totalLen) {
    // forward pass only
    for (let i = 0; i < pts.length - 1; i++) {
      const r = target.clone().sub(pts[i]).normalize().multiplyScalar(lens[i]);
      pts[i + 1] = pts[i].clone().add(r);
    }
  } else {
    // FABRIK: forward & backward
    let b = base.clone();
    let diff = pts[pts.length - 1].distanceTo(target);
    let k = 0;
    while (diff > eps && k++ < iters) {
      // Forward: set end to target
      pts[pts.length - 1] = target.clone();
      for (let i = pts.length - 2; i >= 0; i--) {
        const dir = pts[i]
          .clone()
          .sub(pts[i + 1])
          .normalize();
        pts[i] = pts[i + 1].clone().add(dir.multiplyScalar(lens[i]));
      }
      // Backward: set base to original
      pts[0] = b.clone();
      for (let i = 0; i < pts.length - 1; i++) {
        const dir = pts[i + 1].clone().sub(pts[i]).normalize();
        pts[i + 1] = pts[i].clone().add(dir.multiplyScalar(lens[i]));
      }
      diff = pts[pts.length - 1].distanceTo(target);
    }
  }

  // Align the mid-joint to the limb’s pole plane (keeps elbow/knee from flipping)
  const poleDir = getPoleDirWorld(limb);
  const p0 = pts[0],
    p1 = pts[1],
    p2 = pts[2];
  const p1Aligned = alignMidToPole(p0, p1, p2, poleDir);
  pts[1] = p1Aligned; // keep endpoints fixed; rotate mid around the axis

  // Update bone orientations to aim +Y to next segment and positions (only for base if chain root != parent)
  for (let i = 0; i < bones.length; i++) {
    const bone = bones[i];
    // Desired direction in WORLD:
    if (i < bones.length - 1) {
      const nextPos = pts[i + 1];
      const dirWorld = nextPos.clone().sub(pts[i]).normalize();
      // Express that direction in the PARENT’S LOCAL space:
      const parentWQ = bone.parent.getWorldQuaternion(new Quaternion());
      const parentWQInv = parentWQ.clone().invert();
      const dirLocal = dirWorld
        .clone()
        .applyQuaternion(parentWQInv)
        .normalize();

      // Map bind axis -> desired local direction
      const bindAxis =
        bone.userData.bindAxisLocal || new Vector3(0, 1, 0);
      const qLocal = new Quaternion();
      const dot = bindAxis.dot(dirLocal);
      if (dot < -0.9995) {
        // Stable 180° rotation around an axis orthogonal to bindAxis
        const ortho =
          Math.abs(bindAxis.x) < 0.9
            ? new Vector3(1, 0, 0)
            : new Vector3(0, 0, 1);
        const axis = ortho
          .clone()
          .sub(bindAxis.clone().multiplyScalar(ortho.dot(bindAxis)))
          .normalize();
        qLocal.setFromAxisAngle(axis, Math.PI);
      } else if (dot > 0.9995) {
        qLocal.identity(); // already aligned
      } else {
        qLocal.setFromUnitVectors(bindAxis, dirLocal);
      }
      bone.quaternion.copy(qLocal);
      // Clamp after solve (softly)
      const lim = jointLimits[bone.name];
      if (lim) {
        const e = new Euler().setFromQuaternion(bone.quaternion, "XYZ");
        const ex = lim.x ? clamp(e.x * RAD, lim.x[0], lim.x[1]) : e.x * RAD;
        const ey = lim.y ? clamp(e.y * RAD, lim.y[0], lim.y[1]) : e.y * RAD;
        const ez = lim.z ? clamp(e.z * RAD, lim.z[0], lim.z[1]) : e.z * RAD;
        bone.quaternion.setFromEuler(
          new Euler(ex * DEG, ey * DEG, ez * DEG, "XYZ")
        );
      }
    }
  }
  if (symmetryEnabled && !fromMirror) {
    // Mirror once without bouncing back recursively
    mirrorPartnerLimb(limb);
  }
  // Bones changed -> refresh world matrices for consistent reads elsewhere
  scene.updateMatrixWorld(true);
}

// Keep IK handle positions in sync with current end-effectors
function syncIKHandles() {
  for (const [limb, handle] of Object.entries(ikHandles)) {
    const end = bonesByName.get(chainForLimb(limb).slice(-1)[0]);
    const p = end.getWorldPosition(new Vector3());
    if (tctrl.object !== handle) {
      handle.position.copy(p);
    }
    handle.userData.isIKHandle = true;
    handle.userData.limb = limb;
    handle.visible = ikMode[limb] === "IK";
  }
}

// ---------- Symmetry & Mirror ----------
const mirrorMap = new Map([
  ["LeftClavicle", "RightClavicle"],
  ["LeftShoulder", "RightShoulder"],
  ["LeftElbow", "RightElbow"],
  ["LeftWrist", "RightWrist"],
  ["LeftHand", "RightHand"],
  ["LeftHip", "RightHip"],
  ["LeftKnee", "RightKnee"],
  ["LeftAnkle", "RightAnkle"],
  ["LeftFoot", "RightFoot"],
]);
function partnerName(name) {
  for (const [L, R] of mirrorMap) {
    if (name === L) return R;
    if (name === R) return L;
  }
  return null;
}
let symmetryEnabled = false;
function mirrorQuaternionAcrossX(q) {
  // M R M where M = diag(-1,1,1) (mirror across X). Convert to matrix, apply, then back to quaternion.
  const m = new Matrix4().makeRotationFromQuaternion(q);
  const M = new Matrix4().makeScale(-1, 1, 1);
  const out = new Matrix4().copy(M).multiply(m).multiply(M);
  const q2 = new Quaternion().setFromRotationMatrix(out);
  return q2;
}
function mirrorBonePose(srcBone, dstBone) {
  // Mirror local rotation across world X and negate local X position (for clavicle/hip spacing)
  dstBone.quaternion.copy(mirrorQuaternionAcrossX(srcBone.quaternion));
  if (srcBone.parent === dstBone.parent) {
    dstBone.position.set(
      -srcBone.position.x,
      srcBone.position.y,
      srcBone.position.z
    );
  }
}
function mirrorPoseAll() {
  for (const [L, R] of mirrorMap) {
    const l = bonesByName.get(L),
      r = bonesByName.get(R);
    // decide direction based on current selection (left→right or right→left); default left→right
    if (selectedBone && selectedBone.name.startsWith("Right")) {
      mirrorBonePose(r, l);
    } else {
      mirrorBonePose(l, r);
    }
  }
  markPoseDirty();
  setStatus("Pose mirrored across X");
}
function mirrorPartnerOnEdit(bone) {
  const p = partnerName(bone.name);
  if (!p) return;
  const partner = bonesByName.get(p);
  if (!partner) return;
  mirrorBonePose(bone, partner);
}
function mirrorPartnerLimb(limb) {
  const other = limb.startsWith("Left")
    ? limb.replace("Left", "Right")
    : limb.replace("Right", "Left");
  const srcEnd = chainForLimb(limb).slice(-1)[0];
  const dstEnd = chainForLimb(other).slice(-1)[0];
  const src = bonesByName.get(srcEnd).getWorldPosition(new Vector3());
  const dst = new Vector3(-src.x, src.y, src.z); // mirror end-effector target across X
  ikHandles[other].position.copy(dst);
  solveIKForLimb(other, /*fromMirror=*/ true);
}

// ---------- Pose Library (localStorage) ----------
const LIBKEY = "openpose3d-library-v1";
function loadLibrary() {
  try {
    const raw = localStorage.getItem(LIBKEY);
    return raw ? JSON.parse(raw) : { poses: [] };
  } catch {
    return { poses: [] };
  }
}
function saveLibrary(lib) {
  localStorage.setItem(LIBKEY, JSON.stringify(lib));
}
function captureThumbnail(w = 200, h = 140) {
  // Draw current renderer image into a smaller canvas to create a lightweight dataURL
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  const ctx = tmp.getContext("2d");
  ctx.drawImage(renderer.domElement, 0, 0, w, h);
  return tmp.toDataURL("image/png");
}
function currentPoseAsJSON() {
  const o = {};
  for (const [name, b] of bonesByName) {
    const e = eulerDeg(b);
    o[name] = {
      r: [e.x, e.y, e.z],
      p: [b.position.x, b.position.y, b.position.z],
    };
  }
  return o;
}
function applyPose(json) {
  for (const [name, data] of Object.entries(json)) {
    const b = bonesByName.get(name);
    if (!b) continue;
    const [rx, ry, rz] = data.r || [0, 0, 0];
    applyEulerDeg(b, rx, ry, rz, true);
    if (name === "Root" && data.p) {
      b.position.set(data.p[0], data.p[1], data.p[2]);
    }
  }
  scene.updateMatrixWorld(true);
  syncIKHandles();
  markPoseDirty(); // schedule a render so the viewport updates immediately
}
function refreshThumbs() {
  const { poses } = loadLibrary();
  const box = byId("thumbs");
  box.innerHTML = "";
  poses.forEach((p, idx) => {
    const card = document.createElement("div");
    card.className = "thumb";
    const img = document.createElement("img");
    img.src = p.thumb || "";
    const cap = document.createElement("div");
    cap.className = "cap";
    const name = document.createElement("p");
    name.textContent = p.name;
    const butContainer = document.createElement("div");
    butContainer.className = "button-container";
    const del = document.createElement("button");
    del.textContent = "Delete";
    del.className = "smaller button";
    // const over = document.createElement("button");
    // over.textContent = "Overwrite";
    // over.className = "smaller button";
    card.appendChild(img);
    cap.appendChild(name);
    cap.appendChild(butContainer);
    // butContainer.appendChild(over);
    butContainer.appendChild(del);
    card.appendChild(cap);
    card.onclick = (ev) => {
      if (
        // ev.target === over ||
        ev.target === del
      ) {
        return;
      }
      applyPose(p.pose);
      setStatus(`Applied pose: ${p.name}`);
    };
    // over.onclick = (ev) => {
    //   ev.stopPropagation();
    //   const lib = loadLibrary();
    //   lib.poses[idx] = {
    //     name: p.name,
    //     pose: currentPoseAsJSON(),
    //     thumb: captureThumbnail(),
    //   };
    //   saveLibrary(lib);
    //   refreshThumbs();
    //   setStatus(`Overwrote: ${p.name}`);
    // };
    del.onclick = (ev) => {
      ev.stopPropagation();
      const lib = loadLibrary();
      lib.poses.splice(idx, 1);
      saveLibrary(lib);
      refreshThumbs();
      setStatus(`Deleted: ${p.name}`);
    };
    box.appendChild(card);
  });
}
// ---------- OpenPose (BODY_25) Export ----------
// Loosely based on openpose BODY_25 format, draw things seems to have
// missing joints so we adapt accordingly in an odd format so we adapt accordingly.
// BODY_25 indices and how we source them from your rig.
// For missing parts (face, toes, heel), we export (0,0,0) per OpenPose convention.
const OP_BODY25_ORDER = [
  "REye",
  "Neck",
  "RShoulder",
  "RElbow",
  "RWrist",
  "LShoulder",
  "LElbow",
  "LWrist",
  "RHip",
  "RKnee",
  "RAnkle",
  "LHip",
  "LKnee",
  "LAnkle",
  "REye",
  "LEye",
  "REar",
  "LEar",
];

// Map OpenPose names to your skeleton bones (or a function)
const OP_TO_BONE = {
  Neck: "Neck",
  RShoulder: "RightShoulder",
  RElbow: "RightElbow",
  RWrist: "RightWrist",
  LShoulder: "LeftShoulder",
  LElbow: "LeftElbow",
  LWrist: "LeftWrist",
  RHip: "RightHip",
  RKnee: "RightKnee",
  RAnkle: "RightAnkle",
  LHip: "LeftHip",
  LKnee: "LeftKnee",
  LAnkle: "LeftAnkle",
};

// Project a world position to normalized screen coordinates [0,1]^2.
// Returns {x,y,c} with c=1 if inside clip range, else {0,0,0}.
function projectWorldTo01(worldVec, camera) {
  if (!worldVec) return { x: 0, y: 0, c: 0 };
  const v = worldVec.clone().project(camera); // NDC [-1,1]
  const inClip = v.z >= -1 && v.z <= 1;
  if (!inClip) return { x: 0, y: 0, c: 0 };
  const x = (v.x + 1) * 0.5;
  const y = (1 - v.y) * 0.5; // top-left origin like OpenPose
  // Clamp defensively
  return {
    x: Math.min(1, Math.max(0, x)),
    y: Math.min(1, Math.max(0, y)),
    c: 1.0,
  };
}

// For pixel coordinates instead of normalized: convert [0,1] → pixels
function toPixels01(pt01) {
  const rect = renderer.domElement.getBoundingClientRect();
  return { x: pt01.x * rect.width, y: pt01.y * rect.height, c: pt01.c };
}

function buildOpenPoseBody25({ normalize = true } = {}) {
  scene.updateMatrixWorld(true);

  const triples = [];
  for (const label of OP_BODY25_ORDER) {
    let worldPos = null;

    if (OP_TO_BONE[label]) {
      if (typeof OP_TO_BONE[label] === "function") {
        worldPos = OP_TO_BONE[label](bonesByName); // e.g., MidHip midpoint
      } else {
        const b = bonesByName.get(OP_TO_BONE[label]);
        if (b) worldPos = b.getWorldPosition(new Vector3());
      }
    }
    // Missing labels (Nose/Eyes/Ears/Toes/Heels) → null → 0,0,0

    let pt = projectWorldTo01(worldPos, camera);
    if (!normalize && pt.c > 0) {
      pt = toPixels01(pt);
    }
    triples.push(pt.x, pt.y, pt.c);
  }
  return triples;
}

function exportOpenPoseJSON({ normalize = true } = {}) {
  // Single-person export compatible with OpenPose JSON
  const pose2d = buildOpenPoseBody25({ normalize });

  const payload = {
    people: [
      {
        pose_keypoints_2d: pose2d,
      },
    ],
  };

  const blob = new Blob([JSON.stringify(payload)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  a.href = URL.createObjectURL(blob);
  a.download = `openpose_body25_${normalize ? "norm01" : "px"}_${ts}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportOpenPoseClipboard({ normalize = true } = {}) {
  // Single-person export compatible with OpenPose JSON
  const pose2d = buildOpenPoseBody25({ normalize });

  const payload = {
    people: [
      {
        pose_keypoints_2d: pose2d,
      },
    ],
  };
  navigator.clipboard.writeText(JSON.stringify(payload)).then(
    () => {
      errorToast("Copied OpenPose JSON to clipboard!", false);
    },
    () => {
      errorToast("Failed to copy OpenPose JSON to clipboard.", true);
    }
  );
}

// Hook up the button
document.getElementById("export-openpose").onclick = () => {
  const normalize = !!document.getElementById("openpose-normalize")?.checked;
  exportOpenPoseJSON({ normalize });
};
document.getElementById("copy-openpose").onclick = () => {
  const normalize = !!document.getElementById("openpose-normalize")?.checked;
  exportOpenPoseClipboard({ normalize });
};
byId("save-pose").onclick = () => {
  const name =
    byId("pose-name").value.trim() ||
    `${new Date()
      .toLocaleString([], {
        hour: "numeric",
        hour12: false,
        day: "numeric",
        month: "numeric",
        minute: "numeric",
        second: "numeric",
      })
      .replace(", ", "-")}`;
  const pose = currentPoseAsJSON();
  const thumb = captureThumbnail();
  const lib = loadLibrary();
  // If name exists, replace (explicit overwrite is in UI too)
  const idx = lib.poses.findIndex((p) => p.name === name);
  if (idx >= 0) lib.poses[idx] = { name, pose, thumb };
  else lib.poses.unshift({ name, pose, thumb });
  saveLibrary(lib);
  refreshThumbs();
  setStatus(`Saved pose: ${name}`);
};
byId("export-library").onclick = () => {
  const lib = loadLibrary();
  const blob = new Blob([JSON.stringify(lib)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "pose-library.json";
  a.click();
  URL.revokeObjectURL(a.href);
};
byId("import-library").onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      if (!data || !Array.isArray(data.poses)) throw new Error("Invalid file");
      saveLibrary(data);
      refreshThumbs();
      setStatus(`Imported ${data.poses.length} poses`);
    } catch (err) {
      alert("Import failed: " + err.message);
    }
  };
  reader.readAsText(file);
};

refreshThumbs();

// ---------- Reference image overlay ----------
const refImg = byId("ref-img");
const refFile = byId("ref-file");
const refOpacity = byId("ref-opacity");
const refScale = byId("ref-scale");
const refOX = byId("ref-ox");
const refOY = byId("ref-oy");

refFile.onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  refImg.src = url;
  refImg.onload = () => {
    refImg.classList.remove("hidden");
    setStatus("Reference image loaded");
  };
};
function applyRefControls() {
  refImg.style.opacity = refOpacity.value;
  refImg.style.transform = `translate(${refOX.value}px, ${refOY.value}px) scale(${refScale.value})`;
}
[refOpacity, refScale, refOX, refOY].forEach((el) =>
  el.addEventListener("input", applyRefControls)
);

// ---------- Camera buttons ----------
function usePersp() {
  camera = persp;
  byId("cam-persp").dataset.state = "active";
  byId("cam-ortho-front").dataset.state = "";
  byId("cam-ortho-side").dataset.state = "";
  orbit.object = camera;
  tctrl.camera = camera;
  orbit.enableRotate = true;
  orbit.enableZoom = true;
  orbit.zoomSpeed = 15;
  orbit.enablePan = true;
  setStatus("Camera: Perspective");
}
function useOrthoFront() {
  camera = ortho;
  byId("cam-persp").dataset.state = "";
  byId("cam-ortho-front").dataset.state = "active";
  byId("cam-ortho-side").dataset.state = "";
  orbit.object = camera;
  tctrl.camera = camera;
  // lock to exact front view
  camera.position.set(0, 1.4, 5);
  camera.lookAt(0, 1, 0);
  orbit.target.set(0, 1, 0);
  orbit.enableRotate = false;
  orbit.enableZoom = true;
  orbit.enablePan = true;
  setStatus("Camera: Ortho Front (locked)");
}
function useOrthoSide() {
  camera = ortho;
  byId("cam-persp").dataset.state = "";
  byId("cam-ortho-front").dataset.state = "";
  byId("cam-ortho-side").dataset.state = "active";
  orbit.object = camera;
  tctrl.camera = camera;
  camera.position.set(5, 1.4, 0);
  camera.lookAt(0, 1, 0);
  orbit.target.set(0, 1, 0);
  orbit.enableRotate = false;
  orbit.enableZoom = true;
  orbit.enablePan = true;
  setStatus("Camera: Ortho Side (locked)");
}
// Buttons
byId("cam-persp").onclick = () => {
  orbit.enableRotate = true;
  usePersp();
};
byId("cam-ortho-front").onclick = useOrthoFront;
byId("cam-ortho-side").onclick = useOrthoSide;

// ---------- Toolbar buttons ----------
byId("mode-translate").onclick = () => setMode("translate");
byId("mode-rotate").onclick = () => setMode("rotate");
byId("space-local").onclick = () => setSpace("local");
byId("space-world").onclick = () => setSpace("world");
byId("toggle-grid").onclick = () => {
  gridVisible = !gridVisible;
  grid.visible = gridVisible;
  byId("toggle-grid").dataset.state = gridVisible ? "active" : "";
};
byId("reset-pose").onclick = resetPose;
byId("symmetry").onclick = (e) => {
  symmetryEnabled = !symmetryEnabled;
  byId("symmetry").dataset.state = symmetryEnabled ? "active" : "";
  mirrorPoseAll();
};
byId("screenshot").onclick = () => {
  const a = document.createElement("a");
  a.href = renderer.domElement.toDataURL("image/png");
  a.download = "openpose3d.png";
  a.click();
};

// ---------- FK/IK toggles UI ----------
byId("ik-left-arm").onchange = (e) => setIKMode("LeftArm", e.target.value);
byId("ik-right-arm").onchange = (e) => setIKMode("RightArm", e.target.value);
byId("ik-left-leg").onchange = (e) => setIKMode("LeftLeg", e.target.value);
byId("ik-right-leg").onchange = (e) => setIKMode("RightLeg", e.target.value);

function setIKMode(limb, mode) {
  ikMode[limb] = mode;
  ikHandles[limb].visible = mode === "IK";
  if (mode === "IK") {
    // place handle at end-effector world pos
    const endName = chainForLimb(limb).slice(-1)[0];
    const end = bonesByName.get(endName);
    scene.updateMatrixWorld(true);
    ikHandles[limb].position.copy(end.getWorldPosition(new Vector3()));
    ikHandles[limb].userData.isIKHandle = true;
    ikHandles[limb].userData.limb = limb;
  } else {
    // if we were attached to this handle, detach
    if (tctrl.object === ikHandles[limb]) tctrl.detach();
  }
}

// ---------- Keyboard shortcuts ----------
window.addEventListener("keydown", (e) => {
  if (e.key === "w" || e.key === "W") {
    setMode("translate");
  }
  if (e.key === "e" || e.key === "E") {
    setMode("rotate");
  }
  if (e.key === "g" || e.key === "G") {
    gridVisible = !gridVisible;
    grid.visible = gridVisible;

    byId("toggle-grid").dataset.state = gridVisible ? "active" : "";
  }
  if (e.key === "f" || e.key === "F") {
    frameSelected();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "s") {
    e.preventDefault();
    byId("save-pose").click();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "c") {
    e.preventDefault();
    byId("copy-openpose").click();
  }
  markPoseDirty();
});

function frameSelected() {
  const obj = tctrl.object || bonesByName.get("Root");
  const box = new Box3().setFromObject(obj);
  const size = new Vector3();
  box.getSize(size);
  const center = new Vector3();
  box.getCenter(center);
  orbit.target.copy(center);
  const dist = Math.max(size.x, size.y, size.z) * 2 + 0.5;
  if (camera.isPerspectiveCamera) {
    const dir = new Vector3()
      .subVectors(camera.position, orbit.target)
      .normalize();
    camera.position.copy(orbit.target).addScaledVector(dir, dist);
  } else {
    // Ortho: adjust zoom by changing view size
    // Keep it simple: just move back a bit
    camera.position.set(center.x + 2, center.y + 2, center.z + 2);
    camera.lookAt(center);
  }
}

// ---------- Pose reset ----------
function resetPose() {
  for (const [name, b] of bonesByName) {
    b.quaternion.identity();
    // Reset local position to bind pose per spec
    const spec = boneSpec.find((s) => s[0] === name);
    if (spec) b.position.fromArray(spec[2]);
  }
  scene.updateMatrixWorld(true);
  syncIKHandles();
  markPoseDirty(); // render now so users see the reset
  setStatus("Pose reset");
}

// ---------- Tree selection auto-update at start ----------
updateInspectorUI();

// ---------- Layout / Resize ----------
function onResize() {
  renderer.setSize(viewportEl.clientWidth, viewportEl.clientHeight);
  persp.aspect = viewportEl.clientWidth / viewportEl.clientHeight;
  persp.updateProjectionMatrix();
  // Match ortho frustum to viewport
  const s = orthoSize;
  const aspect = viewportEl.clientWidth / viewportEl.clientHeight;
  if (aspect >= 1) {
    ortho.left = -s * aspect;
    ortho.right = s * aspect;
    ortho.top = s;
    ortho.bottom = -s;
  } else {
    ortho.left = -s;
    ortho.right = s;
    ortho.top = s / aspect;
    ortho.bottom = -s / aspect;
  }
  ortho.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);

// ---------- Toolbar initial state ----------
setMode("rotate");
setSpace("local");
usePersp();

// ---------- Render loop ----------
function markPoseDirty() {
  needsRender = true;
}
function render() {
  orbit.update();
  // Ensure world matrices are fresh before any getWorldPosition/getWorldQuaternion calls
  scene.updateMatrixWorld(true);
  syncIKHandles(); // this reads end-effector world positions
  renderer.render(scene, camera);
}
function tick() {
  if (needsRender) {
    render();
    needsRender = false;
  } else {
    orbit.update();
  }
  requestAnimationFrame(tick);
}
tick();

// Ensure we render on any input/change
const ro = new ResizeObserver(() => {
  onResize();
  markPoseDirty();
});
ro.observe(viewportEl);
renderer.domElement.addEventListener("pointermove", () => markPoseDirty());
document.addEventListener("input", () => markPoseDirty());
document.addEventListener("change", () => markPoseDirty());

// ---------- Toolbar button states ----------
function toggleState(el, active) {
  el.dataset.state = active ? "active" : "";
}

// Activate gizmo buttons & space buttons
byId("mode-translate").addEventListener("click", () => setMode("translate"));
byId("mode-rotate").addEventListener("click", () => setMode("rotate"));
byId("space-local").addEventListener("click", () => setSpace("local"));
byId("space-world").addEventListener("click", () => setSpace("world"));

document.addEventListener('DOMContentLoaded', () => {
  MicroModal.init()
});

// ---------- Final small touches ----------
setStatus("Ready. Click a joint sphere to select. Use E/W to switch gizmos.");
