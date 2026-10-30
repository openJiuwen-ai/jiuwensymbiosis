import {ViewportNavigation} from './navigation.mjs';
import {contentRect, pickCoordinates, boxPath} from './selection.mjs';
import {CounterRate, BackendMetrics} from './metrics.mjs';
const byId = (id) => document.getElementById(id);
const video = byId('video');
const observing = () => (viewState?.settings?.camera_name || 'observer') === 'observer';
let peer = null, session = null, generation = 0, connecting = false, busy = false;
let backendState = 'unknown', pendingCamera = null, sendingCamera = false;
let lastHeartbeat = 0, polling = false;
const backendMetrics = new BackendMetrics(), decodedRate = new CounterRate(), presentedRate = new CounterRate();
let presentedFrames = null, videoFrameCallback = null;
let decodedFps = null, presentedFps = null;
let observedConfig = -1;
let sceneCatalog = [];
let configGeneration = 0;
let viewState = null, selection = null, selecting = false, selectionGeneration = 0;
let selectionTask = Promise.resolve(), cameraRequest = Promise.resolve();

async function post(path, action, body = {}, keepalive = false) {
  const response = await fetch(`/genesis/${path}`, {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Genesis-Action': action},
    body: JSON.stringify(body), keepalive,
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : '请求参数无效。');
  return result;
}
function showError(error) { byId('notice').textContent = error.message || String(error); }
function buttons() {
  if (busy || backendState !== 'ready' || !session || peer?.connectionState !== 'connected') navigation.stop();
  document.body.dataset.connected = String(peer?.connectionState === 'connected');
  for (const id of ['resolution', 'fps', 'scene', 'scene-search', 'robot-mode', 'robot', 'render-quality']) byId(id).disabled = busy || ['starting', 'reconfiguring', 'stopping'].includes(backendState);
  byId('start').disabled = busy || ['starting', 'ready', 'reconfiguring', 'stopping'].includes(backendState);
  byId('stop').disabled = busy || ['stopped', 'unknown', 'unavailable'].includes(backendState);
  byId('connect').disabled = busy || backendState !== 'ready' || !!peer || connecting;
  byId('disconnect').disabled = !peer && !connecting;
  byId('start').hidden = backendState !== 'stopped';
  byId('stop').hidden = ['stopped', 'unknown', 'unavailable'].includes(backendState);
  byId('connect').hidden = backendState !== 'ready' || !!peer || connecting;
  byId('disconnect').hidden = !peer && !connecting;
  const controlsUnavailable = busy || backendState !== 'ready' || !session || peer?.connectionState !== 'connected';
  byId('camera-source').disabled = busy || backendState !== 'ready';
  byId('resolution').disabled ||= !observing();
  byId('reset').disabled = controlsUnavailable || selecting || !observing();
  byId('navigation-hint').hidden = !observing();
  byId('focus').disabled = byId('reset').disabled;
  byId('clear-selection').disabled = byId('reset').disabled;
  for (const button of document.querySelectorAll('[data-camera]')) button.disabled = byId('reset').disabled;
  const mobile = viewState?.robot?.control_mode === 'wheel_velocity_physics';
  byId('base-controls').hidden = !mobile;
  byId('robot-stop').hidden = !mobile;
  for (const button of document.querySelectorAll('[data-base]')) {
    button.disabled = controlsUnavailable || !mobile || (selecting && button.dataset.base !== 'stop');
  }
  drawSelection();
}
function showBackend(state) {
  const mode = state.scene?.background_id ? 'MOZ1 室内物理 · 固定家具碰撞' : {static_background: '静态背景', robot_preview: '机器人检视 · 固定姿态', robot_mobile: 'MOZ1 底盘物理 · 上身锁定'}[state.scene?.mode] || '物理仿真';
  byId('backend').textContent = state.state === 'ready'
    ? `后台运行中 · ${mode}`
    : state.message;
  byId('backend').title = state.message || '';
}
async function disconnect(keepalive = false) {
  navigation.stop();
  generation += 1;
  const oldSession = session, oldPeer = peer;
  session = peer = pendingCamera = null;
  connecting = false;
  if (oldPeer) { oldPeer.onconnectionstatechange = null; oldPeer.close(); }
  video.srcObject = null;
  byId('placeholder').hidden = false;
  byId('connection').textContent = '视频未连接';
  resetVideoMetrics();
  showMetrics();
  buttons();
  if (oldSession) await post('disconnect', 'disconnect', {session_id: oldSession}, keepalive);
}
function waitIce(pc) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error('收集本机网络候选超时，请重连。')); }, 8000);
    const changed = () => { if (pc.iceGatheringState === 'complete') { cleanup(); resolve(); } };
    const cleanup = () => { clearTimeout(timer); pc.removeEventListener('icegatheringstatechange', changed); };
    pc.addEventListener('icegatheringstatechange', changed);
  });
}
async function connect() {
  if (peer || connecting) return;
  const token = ++generation;
  connecting = true;
  byId('notice').textContent = '';
  const pc = new RTCPeerConnection({iceServers: []});
  peer = pc;
  lastHeartbeat = 0;
  resetVideoMetrics();
  byId('connection').textContent = '正在建立视频连接…';
  buttons();
  pc.addTransceiver('video', {direction: 'recvonly'});
  pc.ontrack = (event) => {
    if (token !== generation) return;
    video.srcObject = new MediaStream([event.track]);
    watchPresentedFrames();
    video.play().catch(showError);
  };
  pc.onconnectionstatechange = () => {
    if (token !== generation) return;
    byId('connection').textContent = ({connected: '视频已连接', connecting: '视频协商中…'})[pc.connectionState] || pc.connectionState;
    buttons();
    if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
      disconnect().catch(showError);
      showError(new Error('视频连接已结束，可以重新连接。'));
    }
  };
  try {
    await pc.setLocalDescription(await pc.createOffer());
    await waitIce(pc);
    if (token !== generation) return;
    const answer = await post('offer', 'offer', {sdp: pc.localDescription.sdp, type: 'offer'});
    if (token !== generation) {
      await post('disconnect', 'disconnect', {session_id: answer.session_id});
      return;
    }
    session = answer.session_id;
    await pc.setRemoteDescription({sdp: answer.sdp, type: answer.type});
  } catch (error) {
    if (token === generation) { await disconnect().catch(showError); showError(error); }
  } finally { if (token === generation) connecting = false; buttons(); }
}
async function backendAction(action) {
  configGeneration += 1;
  busy = true; buttons(); byId('notice').textContent = '';
  byId('backend').textContent = action === 'start' ? '正在启动 Genesis / Nyx…' : action === 'configure' ? '正在切换，视频将自动恢复…' : '正在停止 Genesis…';
  try {
    if (action === 'stop') await disconnect();
    const options = action !== 'stop' ? {resolution: byId('resolution').value, fps: Number(byId('fps').value), scene_id: byId('scene').value, robot_mode: byId('robot-mode').value, robot_id: byId('robot').value, render_quality: byId('render-quality').value, camera_name: byId('camera-source').value} : {};
    const state = await post(`backend/${action}`, action, options);
    backendState = state.state;
    showBackend(state);
  } catch (error) { showError(error); }
  finally { busy = false; buttons(); }
}
const clamp = (v) => Math.max(-1, Math.min(1, v));
let cameraTimer = null;
function camera(action, dx = 0, dy = 0, delta = 0, forward = 0, strafe = 0, lift = 0) {
  if (!observing() || busy || selecting || backendState !== 'ready' || !session || peer?.connectionState !== 'connected') return;
  if (pendingCamera?.action !== action) pendingCamera = {action};
  for (const [key, value] of Object.entries({dx, dy, delta, forward, strafe, lift})) {
    pendingCamera[key] = clamp((pendingCamera[key] || 0) + value);
  }
  scheduleCamera();
}
function scheduleCamera() {
  if (cameraTimer !== null || sendingCamera) return;
  cameraTimer = setTimeout(() => { cameraTimer = null; flushCamera(); }, 33);
}
async function flushCamera() {
  if (sendingCamera || !pendingCamera || !session) return;
  sendingCamera = true;
  const body = pendingCamera, id = session;
  pendingCamera = null;
  try {
    cameraRequest = post('camera', 'camera', {session_id: id, ...body});
    const result = await cameraRequest;
    if (id === session && viewState) { viewState.camera = result.requested_camera; drawSelection(); }
  }
  catch (error) { if (id === session) showError(error); }
  finally { sendingCamera = false; if (pendingCamera) scheduleCamera(); }
}
const navigation = new ViewportNavigation(video, {
  ready: () => observing() && !busy && !selecting && backendState === 'ready' && !!session && peer?.connectionState === 'connected',
  send: ({action, dx, dy, delta, forward, strafe, lift}) => camera(action, dx, dy, delta, forward, strafe, lift),
  cancel: () => { pendingCamera = null; clearTimeout(cameraTimer); cameraTimer = null; },
  onSpeed: speed => {
    byId('navigation-speed').value = String(speed);
    byId('navigation-hint').textContent = `右键环顾 · WASD 移动 · 速度 ${speed}×`;
  },
  onPick: (x, y) => {
    const pixel = pickCoordinates(video.getBoundingClientRect(), video.videoWidth, video.videoHeight, x, y);
    selectObject(pixel ? 'pick' : 'clear', pixel || {});
  },
  onFocus: () => focusSelection(),
  onClear: () => selectObject('clear'),
});

function drawSelection() {
  const visible = observing() && selection && viewState?.camera && backendState === 'ready' && !busy && peer?.connectionState === 'connected'
    && selection.scene_instance_id === viewState.scene?.instance_id;
  byId('selection-hud').hidden = !visible;
  byId('selection-box').toggleAttribute('hidden', !visible);
  if (!visible || !video.videoWidth || !video.videoHeight) return;
  byId('selection-name').textContent = selection.name;
  byId('selection-name').title = selection.path;
  const rect = video.getBoundingClientRect();
  const image = contentRect(rect, video.videoWidth, video.videoHeight);
  Object.assign(byId('selection-box').style, {left: `${image.left - rect.left}px`, top: `${image.top - rect.top}px`,
    width: `${image.width}px`, height: `${image.height}px`});
  byId('selection-path').setAttribute('d', boxPath(selection.bounds_m, viewState.camera, viewState.camera_fov_y,
    video.videoWidth / video.videoHeight));
}

function selectObject(action, pixel = {}) {
  if (busy || selecting || !session || backendState !== 'ready' || !viewState?.scene) return;
  const id = session, token = ++selectionGeneration;
  selecting = true; navigation.stop(); buttons();
  selectionTask = (async () => {
    try {
      await cameraRequest;
      if (id !== session || busy) return;
      const result = await post('selection', 'selection', {session_id: id, action, ...pixel,
        scene_instance_id: viewState.scene.instance_id, config_revision: viewState.config_revision,
        camera_revision: viewState.camera.revision});
      if (id !== session || token !== selectionGeneration) return;
      selection = result.selection;
      viewState.camera = result.camera;
      byId('notice').textContent = '';
    } catch (error) { if (id === session) showError(error); }
    finally { selecting = false; buttons(); }
  })();
}

async function focusSelection() {
  const id = session;
  await selectionTask;
  if (id === session) { navigation.stop(); camera('focus'); }
}
new ResizeObserver(drawSelection).observe(video);
video.addEventListener('resize', drawSelection);
byId('focus').onclick = focusSelection;
byId('clear-selection').onclick = () => selectObject('clear');
byId('navigation-speed').onchange = () => navigation.setSpeed(Number(byId('navigation-speed').value));
video.addEventListener('playing', () => { byId('placeholder').hidden = true; });
byId('start').onclick = () => backendAction('start');
for (const id of ['scene', 'resolution', 'fps', 'robot-mode', 'robot', 'render-quality', 'camera-source']) byId(id).onchange = () => {
  if (['scene', 'robot', 'robot-mode'].includes(id)) byId('camera-source').value = 'observer';
  navigation.stop();
  const mode = sceneCatalog.find(asset => asset.id === byId('scene').value)?.preview_mode;
  if (id === 'scene' && mode !== 'static_usd') {
    byId('robot').value = '';
    if (mode !== 'urdf') byId('robot-mode').value = 'inspection';
  }
  if (id === 'scene' && mode === 'static_usd' && !byId('robot').value) byId('robot-mode').value = 'inspection';
  if (id === 'robot') byId('robot-mode').value = byId('robot').value ? 'moz1_wheels' : 'inspection';
  if (id === 'robot-mode' && byId('robot-mode').value === 'inspection') byId('robot').value = '';
  showQuality();
  if (backendState === 'ready') backendAction('configure');
};
byId('stop').onclick = () => backendAction('stop');
byId('connect').onclick = () => connect().catch(showError);
byId('disconnect').onclick = () => disconnect().catch(showError);
byId('reset').onclick = () => { navigation.stop(); camera('reset'); };
byId('settings').onclick = () => { showQuality(); byId('settings-dialog').showModal(); };
byId('manual').onclick = () => byId('manual-dialog').showModal();
byId('help').onclick = () => byId('help-dialog').showModal();
byId('diagnostics').onclick = () => byId('diagnostics-dialog').showModal();
for (const button of document.querySelectorAll('[data-close]')) {
  button.onclick = () => byId(button.dataset.close).close();
}
byId('fullscreen').disabled = !document.fullscreenEnabled;
byId('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch (error) { showError(new Error('当前浏览器无法进入全屏，可使用工作台的“专注画面”。')); }
};
document.addEventListener('fullscreenchange', () => {
  const label = document.fullscreenElement ? '退出全屏' : '全屏画面';
  byId('fullscreen').title = label;
  byId('fullscreen').setAttribute('aria-label', label);
  byId('fullscreen-label').textContent = document.fullscreenElement ? '退出全屏' : '全屏';
});
const cameraSteps = {
  left: ['orbit', -0.04, 0], right: ['orbit', 0.04, 0],
  up: ['orbit', 0, 0.04], down: ['orbit', 0, -0.04],
  in: ['zoom', 0, 0, -0.12], out: ['zoom', 0, 0, 0.12],
};
for (const button of document.querySelectorAll('[data-camera]')) {
  button.onclick = () => camera(...cameraSteps[button.dataset.camera]);
}
const baseSteps = {forward: [.15, 0, 0], backward: [-.15, 0, 0], left: [0, .15, 0],
  right: [0, -.15, 0], ccw: [0, 0, .3], cw: [0, 0, -.3], stop: [0, 0, 0]};
for (const button of document.querySelectorAll('[data-base]')) {
  button.onclick = async () => {
    if (button.disabled || !session || !viewState?.scene) return;
    const [vx, vy, wz] = baseSteps[button.dataset.base];
    try {
      await post('base', 'base', {session_id: session, vx, vy, wz,
        scene_instance_id: viewState.scene.instance_id, config_revision: viewState.config_revision});
    } catch (error) { showError(error); }
  };
}
window.addEventListener('pagehide', () => { disconnect(true).catch(() => {}); });
window.addEventListener('message', (event) => {
  if (event.origin === location.origin && event.source === window.parent && event.data?.type === 'genesis-disconnect') disconnect().catch(showError);
});
async function poll() {
  if (polling) return;
  polling = true;
  const configToken = configGeneration;
  const selectionToken = selectionGeneration;
  try {
    const response = await fetch('/genesis/backend', {cache: 'no-store'});
    if (!response.ok) throw new Error('无法读取后台状态。');
    const state = await response.json();
    if (busy || configToken !== configGeneration) return;
    backendState = state.state;
    if (state.state === 'ready') {
      const sameScene = viewState?.scene?.instance_id === state.scene?.instance_id;
      const camera = sameScene && viewState.camera?.revision > state.camera?.revision ? viewState.camera : state.camera;
      viewState = {...state, camera};
      if (!sameScene || (!selecting && selectionToken === selectionGeneration)) selection = state.selection;
    }
    if (state.state === 'reconfiguring') {
      byId('placeholder').hidden = false;
    } else if (state.state === 'ready' && peer?.connectionState === 'connected') {
      byId('placeholder').hidden = video.readyState >= 2;
    }
    if (state.state === 'ready' && state.config_revision !== observedConfig) {
      observedConfig = state.config_revision;
      decodedRate.reset(); presentedRate.reset();
      decodedFps = presentedFps = null;
      byId('notice').textContent = state.reconfigure_error || '';
    }
    if (state.settings && ['ready', 'starting', 'reconfiguring'].includes(backendState)) {
      const selected = backendState === 'reconfiguring' ? state.requested_settings : state.settings;
      const cameraSelect = byId('camera-source');
      if (state.cameras) {
        const names = ['observer', ...state.cameras.map(camera => camera.name)];
        if (JSON.stringify([...cameraSelect.options].map(option => option.value)) !== JSON.stringify(names)) {
          const labels = {observer: '观察视角', head_cam: '头部相机', left_gripper_cam: '左手相机', right_gripper_cam: '右手相机'};
          cameraSelect.replaceChildren(...names.map(name => new Option(labels[name] || name, name)));
        }
      }
      const name = selected.camera_name || 'observer';
      if ([...cameraSelect.options].some(option => option.value === name)) cameraSelect.value = name;
      byId('resolution').value = `${selected.width}x${selected.height}`;
      byId('fps').value = String(selected.fps);
      byId('render-quality').value = 'reference';
      byId('robot-mode').value = selected.robot_mode || 'inspection';
      byId('robot').value = selected.robot_id || '';
      if ([...byId('scene').options].some(option => option.value === (selected.scene_id || 'demo'))) byId('scene').value = selected.scene_id || 'demo';
    }
    backendMetrics.update(state, performance.now());
    showQuality();
    if (!busy) showBackend(state);
    byId('placeholder').textContent = state.state === 'reconfiguring' ? '正在切换，视频将自动恢复…' : state.state === 'ready' ? '后台已就绪，点击“连接视频”' : state.state === 'starting' ? '后台初始化中，请稍候…' : state.state === 'failed' ? '后台启动或运行失败，请查看下方提示' : '启动后台，等待就绪后连接视频';
    byId('details').textContent = JSON.stringify(state, null, 2);
    if (session && Date.now() - lastHeartbeat > 2000) {
      lastHeartbeat = Date.now();
      try { await post('heartbeat', 'heartbeat', {session_id: session}); }
      catch (error) { await disconnect(); throw error; }
    }
  } catch (error) {
    backendState = 'unavailable';
    backendMetrics.reset();
    byId('backend').textContent = '服务不可达 · 后台运行状态未知';
    if (!peer || peer.connectionState !== 'connected') byId('placeholder').textContent = '暂时无法读取服务状态，请等待连接恢复';
    showError(error);
  } finally {
    await sampleVideoMetrics();
    showMetrics();
    polling = false; buttons();
  }
}
poll(); setInterval(poll, 1000);

fetch('/genesis/scenes').then(async r => {
  if (!r.ok) {
    const error = await r.json();
    throw new Error(error.detail || '读取场景列表失败');
  }
  return r.json();
}).then(scenes => {
  sceneCatalog = scenes;
  const select = byId('scene');
  fillSceneOptions(select, scenes, select.value);
  filterScenes();
  byId('robot').replaceChildren(new Option('无', ''), ...scenes.filter(scene => scene.preview_mode === 'urdf').map(scene => new Option(scene.label, scene.id)));
}).catch(showError);

function resetVideoMetrics() {
  if (videoFrameCallback !== null) video.cancelVideoFrameCallback?.(videoFrameCallback);
  videoFrameCallback = null;
  presentedFrames = null;
  decodedRate.reset(); presentedRate.reset();
  decodedFps = presentedFps = null;
}

function watchPresentedFrames() {
  if (!video.requestVideoFrameCallback) return;
  const token = generation;
  const callback = (now, metadata) => {
    videoFrameCallback = null;
    if (token !== generation) return;
    if (document.visibilityState === 'visible' && Number.isFinite(metadata.presentedFrames)) {
      if (presentedFrames === null) presentedRate.sample(metadata.presentedFrames, now);
      presentedFrames = metadata.presentedFrames;
    }
    videoFrameCallback = video.requestVideoFrameCallback(callback);
  };
  videoFrameCallback = video.requestVideoFrameCallback(callback);
}

async function sampleVideoMetrics() {
  const pc = peer;
  if (pc?.connectionState !== 'connected') { decodedFps = presentedFps = null; return; }
  try {
    const stats = await pc.getStats();
    if (pc !== peer) return;
    const inbound = [...stats.values()].find(s => s.type === 'inbound-rtp' && s.kind === 'video');
    decodedFps = decodedRate.sample(inbound?.framesDecoded, inbound?.timestamp);
    presentedFps = document.visibilityState === 'visible' && presentedFrames !== null
      ? presentedRate.sample(presentedFrames, performance.now()) : null;
  } catch { decodedFps = presentedFps = null; }
}

document.addEventListener('visibilitychange', () => {
  presentedFrames = null;
  presentedRate.reset();
  presentedFps = null;
  showMetrics();
});

const number = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—';
function showMetrics() {
  const metrics = backendMetrics.current(performance.now());
  const element = byId('metrics');
  const values = {decodedFps, presentedFps, ...metrics};
  for (const key of ['decodedFps', 'presentedFps', 'physicsHz', 'realTimeFactor', 'renderFps', 'renderP95Ms']) {
    element.dataset[key] = Number.isFinite(values[key]) ? values[key].toFixed(3) : '';
  }
  element.dataset.sampledAt = String(performance.now());
  element.dataset.backendState = backendState;
  element.dataset.backendFresh = String(metrics.fresh);
  byId('video-metrics').textContent = `呈现 ${number(presentedFps)} · 解码 ${number(decodedFps)} FPS`;
  byId('physics-metrics').textContent = `物理 ${number(metrics.physicsHz)} Hz · RTF ${number(metrics.realTimeFactor, 2)}`;
  byId('render-metrics').textContent = `视频渲染 P95 ${number(metrics.renderP95Ms)} ms`;
  element.title = `${video.videoWidth || '—'} × ${video.videoHeight || '—'} · 后台渲染 ${number(metrics.renderFps)} FPS · RTF 1.00 为实时；呈现为浏览器合成帧率`;
  byId('actual-performance').textContent = `呈现 ${number(presentedFps)} FPS · 物理 ${number(metrics.physicsHz)} Hz · RTF ${number(metrics.realTimeFactor, 2)}`;
}

function showQuality() {
  byId('quality-description').textContent = 'Reference：64 采样，呈现间接光与反射；实际刷新率取决于场景。';
  const label = 'Reference / 64';
  const sensor = viewState?.cameras?.find(camera => camera.name === byId('camera-source').value);
  const size = sensor ? `${sensor.width} × ${sensor.height}` : byId('resolution').value.replace('x', ' × ');
  byId('requested-profile').textContent = `${size} · ${label} · ${byId('fps').value} FPS 上限`;
}

function fillSceneOptions(select, scenes, selected) {
  const groups = new Map();
  for (const scene of scenes) {
    const label = scene.preview_mode === 'static_usd' ? '室内与静态场景' : scene.preview_mode === 'urdf' ? '机器人姿态检视' : '物理演示';
    if (!groups.has(label)) {
      const group = document.createElement('optgroup');
      group.label = label;
      groups.set(label, group);
    }
    groups.get(label).append(new Option(scene.label, scene.id));
  }
  select.replaceChildren(...groups.values());
  if (scenes.some(scene => scene.id === selected)) select.value = selected;
}
function filterScenes() {
  const query = byId('scene-filter').value.trim().toLocaleLowerCase();
  const scenes = sceneCatalog.filter(scene => `${scene.label} ${scene.id}`.toLocaleLowerCase().includes(query));
  fillSceneOptions(byId('scene-results'), scenes, byId('scene-results').value || byId('scene').value);
  if (scenes.length && byId('scene-results').selectedIndex < 0) byId('scene-results').selectedIndex = 0;
  byId('scene-count').textContent = scenes.length ? `${scenes.length} 个匹配场景` : '未找到匹配场景';
  byId('apply-scene').disabled = !scenes.length;
}
byId('scene-search').onclick = () => {
  byId('scene-filter').value = '';
  filterScenes();
  byId('scene-results').value = byId('scene').value;
  byId('scene-dialog').showModal();
  byId('scene-filter').focus();
};
byId('scene-filter').oninput = filterScenes;
byId('apply-scene').onclick = () => {
  const selected = byId('scene-results').value;
  if (!selected || byId('scene').disabled) return;
  byId('scene-dialog').close();
  if (selected !== byId('scene').value) {
    byId('scene').value = selected;
    byId('scene').dispatchEvent(new Event('change'));
  }
};
byId('scene-results').ondblclick = () => byId('apply-scene').click();
showQuality(); showMetrics();
