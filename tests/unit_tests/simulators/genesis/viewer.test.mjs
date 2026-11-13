import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../../../jiuwensymbiosis/simulators/common/web/assets/viewer.js', import.meta.url), 'utf8');

// Execute the real status/request handlers; only DOM, network and display are stubbed.
function viewer() {
  class Option {
    constructor(text, value) { this.text = text; this.value = value; this.dataset = {}; }
  }
  class Select {
    constructor(names) { this.replaceChildren(...names.map(name => new Option(name, name))); }
    get value() { return this.selected; }
    set value(value) { this.selected = this.options.some(option => option.value === value) ? value : ''; }
    replaceChildren(...options) { this.options = options.flatMap(option => option.children || [option]); this.selected = options[0]?.value || ''; }
  }
  const elements = new Map([
    ['camera-source', new Select(['observer', 'head_cam', 'left_gripper_cam'])],
    ['scene', new Select(['room'])],
    ['engine', new Select(['genesis', 'mujoco'])],
    ['robot-mode', new Select(['inspection', 'moz1_wheels', 'jumper', 'microduck'])],
    ['robot', new Select(['', 'moz1'])],
    ['scene-results', new Select([])],
    ['render-quality', new Select(['reference'])],
  ]);
  const byId = id => {
    if (!elements.has(id)) elements.set(id, {value: '', options: []});
    return elements.get(id);
  };
  byId('camera-source').value = 'head_cam';
  const settings = {camera_name: 'head_cam', width: 1280, height: 720, fps: 60,
    render_mode: 'fast', spp: 32, robot_mode: 'moz1_wheels', robot_id: 'moz1', scene_id: 'room'};
  const noop = () => {};
  const context = vm.createContext({
    byId, Option, settings, sceneCatalog: [], document: {createElement() {return {children: [], append(option) {this.children.push(option);}}}}, polling: false, busy: false, configGeneration: 0, selectionGeneration: 0,
    backendState: 'ready', viewState: null, peer: null, session: null, selection: null, selecting: false,
    observedConfig: -1, decodedFps: null, presentedFps: null, performance,
    backendMetrics: {update: noop, reset: noop}, decodedRate: {reset: noop}, presentedRate: {reset: noop},
    showQuality: noop, showBackend: noop, showLoading: noop, showMetrics: noop, buttons: noop, sampleVideoMetrics: async () => {},
    showError: error => { throw error; },
    fetch: async () => ({ok: true, json: async () => context.state}),
    post: async (path, action, options) => { context.sent = options; return {state: 'starting'}; },
  });
  vm.runInContext(source.slice(source.indexOf('async function poll()'), source.indexOf('poll(); setInterval')), context);
  vm.runInContext(source.slice(source.indexOf('async function backendAction('), source.indexOf('const clamp =')), context);
  vm.runInContext(source.slice(source.indexOf('function fillSceneOptions('), source.indexOf("byId('scene-search').onclick")), context);
  vm.runInContext(source.slice(source.indexOf('function availableScenes()')), context);
  return context;
}

test('manual movement uses the live robot rates and keeps the default rates for other robots', async () => {
  const ui = viewer();
  const controls = ['forward', 'backward', 'left', 'right', 'ccw', 'cw', 'stop'].map(direction => ({dataset: {base: direction}}));
  ui.document.querySelectorAll = () => controls;
  ui.session = 'session';
  vm.runInContext(source.slice(source.indexOf('const baseSteps ='), source.indexOf("window.addEventListener('pagehide'")), ui);
  for (const limits of [undefined, {linear_m_s: .4, angular_rad_s: 1.5}, undefined]) {
    ui.viewState = {robot: {command_limits: limits}, scene: {instance_id: 'scene'}, config_revision: 3};
    const linear = limits?.linear_m_s ?? .15, angular = limits?.angular_rad_s ?? .3;
    const expected = [[linear, 0, 0], [-linear, 0, 0], [0, linear, 0], [0, -linear, 0], [0, 0, angular], [0, 0, -angular], [0, 0, 0]];
    for (const [index, button] of controls.entries()) {
      await button.onclick();
      assert.deepEqual([ui.sent.vx, ui.sent.vy, ui.sent.wz], expected[index]);
      assert.equal(ui.sent.scene_instance_id, 'scene');
      assert.equal(ui.sent.config_revision, 3);
    }
  }
});

for (const mode of ['jumper', 'microduck']) {
  test(`selecting ${mode} chooses its physics mode and clears an attached robot`, () => {
    const ui = viewer();
    ui.sceneCatalog = [{id: 'room', preview_mode: mode, robot_mode: mode}];
    ui.navigation = {stop() {}};
    ui.backendState = 'stopped';
    ui.byId('robot').value = 'old-robot';
    const start = source.indexOf("for (const id of ['engine', 'scene', 'resolution'");
    vm.runInContext(source.slice(start, source.indexOf("byId('stop').onclick", start)), ui);
    ui.byId('scene').onchange();
    assert.equal(ui.byId('robot-mode').value, mode);
    assert.equal(ui.byId('robot').value, '');
    assert.equal(ui.byId('camera-source').value, 'observer');
  });
}

for (const pending of ['starting', 'reconfiguring']) {
  for (const ended of ['stopped', 'failed']) {
    test(`${pending} then ${ended} keeps a valid camera for restart`, async () => {
      const ui = viewer();
      const requested = {...ui.settings, camera_name: 'left_gripper_cam'};
      ui.state = {state: pending, settings: ui.settings, requested_settings: requested};
      await ui.poll();
      const expected = pending === 'reconfiguring' ? 'left_gripper_cam' : 'head_cam';
      assert.equal(ui.byId('camera-source').value, expected);
      ui.state = {state: ended, settings: ui.settings};
      await ui.poll();
      await ui.backendAction('start');
      assert.equal(ui.sent.camera_name, expected);
      assert.equal(ui.sent.render_quality, 'reference');
    });
  }
}

test('a ready scene replaces the camera catalog and restores the accepted selection', async () => {
  const ui = viewer();
  ui.state = {state: 'ready', settings: {...ui.settings, camera_name: 'observer'}, cameras: [],
    scene: {instance_id: 'new-scene'}, camera: {revision: 1}, config_revision: 1};
  await ui.poll();
  assert.deepEqual(ui.byId('camera-source').options.map(option => option.value), ['observer']);
  assert.equal(ui.byId('camera-source').value, 'observer');
});

test('engine selection filters scenes and clears the previous controller and camera', () => {
  const ui = viewer();
  ui.sceneCatalog = [
    {id: 'demo', label: 'Demo', backends: ['genesis', 'mujoco']},
    {id: 'usd', label: 'USD', backends: ['genesis']},
    {id: 'jumper', label: 'Jumper', backends: ['mujoco'], robot_mode: 'jumper'},
  ];
  ui.navigation = {stop() {}};
  ui.backendState = 'stopped';
  ui.byId('engine').value = 'mujoco';
  const start = source.indexOf("for (const id of ['engine', 'scene', 'resolution'");
  vm.runInContext(source.slice(start, source.indexOf("byId('stop').onclick", start)), ui);
  ui.byId('engine').onchange();
  assert.deepEqual(ui.byId('scene').options.map(option => option.value), ['demo', 'jumper']);
  assert.equal(ui.byId('robot-mode').value, 'inspection');
  assert.equal(ui.byId('camera-source').value, 'observer');
});
