// Three.js model viewer: loads the mesh/point-cloud GLBs Model Forge writes, with orbit controls,
// colour/clay/wireframe shading and optional camera markers showing where each photo was taken.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const ACCENT = 0xe8793b;

export function createViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 100);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.2;

  scene.add(new THREE.HemisphereLight(0xf2f4f5, 0x3a3228, 1.6));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
  keyLight.position.set(1, 1.4, 1.2);
  camera.add(keyLight);
  scene.add(camera);

  const grid = new THREE.GridHelper(2, 20, 0x5b666c, 0x3a4247);
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  scene.add(grid);

  const root = new THREE.Group();
  scene.add(root);

  let mesh = null;
  let points = null;
  let cameraMarkers = null;
  let show = "mesh";
  let shade = "color";
  let home = { target: new THREE.Vector3(), position: new THREE.Vector3(0, 0.4, 2) };

  const materials = {
    color: new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }),
    clay: new THREE.MeshStandardMaterial({ color: 0xd8d0c4, roughness: 0.85, metalness: 0, side: THREE.DoubleSide }),
    wire: new THREE.MeshBasicMaterial({ color: ACCENT, wireframe: true, transparent: true, opacity: 0.55 }),
  };

  function resize() {
    const { clientWidth: w, clientHeight: h } = canvas;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);

  let running = true;
  function loop() {
    if (!running) return;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  function firstDrawable(gltf, type) {
    let found = null;
    gltf.scene.traverse((obj) => {
      if (!found && obj[type]) found = obj;
    });
    return found;
  }

  async function loadGlb(url) {
    const gltf = await new GLTFLoader().loadAsync(url);
    return gltf;
  }

  function clear() {
    for (const obj of [mesh, points, cameraMarkers]) {
      if (!obj) continue;
      root.remove(obj);
      obj.geometry?.dispose();
    }
    mesh = points = cameraMarkers = null;
  }

  function buildCameraMarkers(cams) {
    if (!cams || !cams.length) return null;
    const size = 0.035;
    const verts = [];
    const up = new THREE.Vector3(0, 1, 0);
    for (const c of cams) {
      const pos = new THREE.Vector3(c[0], c[1], c[2]);
      const fwd = new THREE.Vector3(c[3], c[4], c[5]).normalize();
      let right = new THREE.Vector3().crossVectors(fwd, up);
      if (right.lengthSq() < 1e-6) right = new THREE.Vector3(1, 0, 0);
      right.normalize().multiplyScalar(size * 0.66);
      const camUp = new THREE.Vector3().crossVectors(right, fwd).normalize().multiplyScalar(size * 0.5);
      const base = pos.clone().addScaledVector(fwd, size);
      const corners = [
        base.clone().add(right).add(camUp),
        base.clone().sub(right).add(camUp),
        base.clone().sub(right).sub(camUp),
        base.clone().add(right).sub(camUp),
      ];
      for (let i = 0; i < 4; i++) {
        verts.push(pos, corners[i], corners[i], corners[(i + 1) % 4]);
      }
    }
    const geom = new THREE.BufferGeometry().setFromPoints(verts);
    const lines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.8 }));
    lines.visible = false;
    return lines;
  }

  function frame() {
    const target = mesh && show === "mesh" ? mesh : points || mesh;
    if (!target) return;
    const box = new THREE.Box3().setFromObject(target);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const r = sphere.radius || 1;
    grid.position.y = box.min.y;
    grid.scale.setScalar(Math.max(1, r * 1.6));
    const dist = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
    home = {
      target: sphere.center.clone(),
      position: sphere.center.clone().add(new THREE.Vector3(0, r * 0.45, dist)),
    };
    camera.near = dist / 200;
    camera.far = dist * 20;
    camera.updateProjectionMatrix();
    reset();
  }

  function reset() {
    controls.target.copy(home.target);
    camera.position.copy(home.position);
    controls.update();
  }

  function applyState() {
    if (mesh) {
      mesh.visible = show === "mesh";
      mesh.material = materials[shade];
    }
    if (points) points.visible = show === "points" || !mesh;
  }

  return {
    async load({ meshUrl, pointsUrl, cameras }) {
      clear();
      const [meshGltf, pointsGltf] = await Promise.all([
        meshUrl ? loadGlb(meshUrl) : null,
        pointsUrl ? loadGlb(pointsUrl) : null,
      ]);
      if (meshGltf) {
        mesh = firstDrawable(meshGltf, "isMesh");
        if (mesh) {
          mesh.removeFromParent();
          root.add(mesh);
        }
      }
      if (pointsGltf) {
        points = firstDrawable(pointsGltf, "isPoints");
        if (points) {
          points.removeFromParent();
          const count = points.geometry.attributes.position.count;
          points.material = new THREE.PointsMaterial({
            size: count > 200000 ? 0.0035 : count > 50000 ? 0.006 : 0.01,
            vertexColors: true,
            sizeAttenuation: true,
          });
          root.add(points);
        }
      }
      cameraMarkers = buildCameraMarkers(cameras);
      if (cameraMarkers) root.add(cameraMarkers);
      if (!mesh) show = "points";
      applyState();
      frame();
      return { hasMesh: !!mesh, hasPoints: !!points };
    },
    setShow(value) {
      show = value;
      applyState();
    },
    setShade(value) {
      shade = value;
      applyState();
    },
    setCameras(visible) {
      if (cameraMarkers) cameraMarkers.visible = visible;
    },
    setSpin(on) {
      controls.autoRotate = on;
    },
    reset,
    dispose() {
      running = false;
      clear();
      renderer.dispose();
    },
  };
}
