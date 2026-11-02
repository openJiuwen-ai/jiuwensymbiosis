import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../../../jiuwensymbiosis/simulators/genesis/web/assets/viewer.js', import.meta.url), 'utf8');

// Execute the real status/request handlers; only DOM, network and display are stubbed.
function viewer() {
  class Option {
    constructor(text, value) { this.text = text; this.value = value; }
  }
  class Select {
    constructor(names) { this.replaceChildren(...names.map(name => new Option(name, name))); }
    get value() { return this.selected; }
    set value(value) { this.selected = this.options.some(option => option.value === value) ? value : ''; }
    replaceChildren(...options) { this.options = options; this.selected = options[0]?.value || ''; }
  }
  const elements = new Map([
    ['camera-source', new Select(['observer', 'head_cam', 'left_gripper_cam'])],
    ['scene', new Select(['room'])],
  ]);
  const byId = id => {
    if (!elements.has(id)) elements.set(id, {});
    return elements.get(id);
  };
  byId('camera-source').value = 'head_cam';
  const settings = {camera_name: 'head_cam', width: 1280, height: 720, fps: 60,
    render_mode: 'fast', spp: 32, robot_mode: 'moz1_wheels', robot_id: 'moz1', scene_id: 'room'};
  const noop = () => {};
  const context = vm.createContext({
    byId, Option, settings, polling: false, busy: false, configGeneration: 0, selectionGeneration: 0,
    backendState: 'ready', viewState: null, peer: null, session: null, selection: null, selecting: false,
    observedConfig: -1, decodedFps: null, presentedFps: null, performance,
    backendMetrics: {update: noop, reset: noop}, decodedRate: {reset: noop}, presentedRate: {reset: noop},
    showQuality: noop, showBackend: noop, showMetrics: noop, buttons: noop, sampleVideoMetrics: async () => {},
    showError: error => { throw error; },
    fetch: async () => ({ok: true, json: async () => context.state}),
    post: async (path, action, options) => { context.sent = options; return {state: 'starting'}; },
  });
  vm.runInContext(source.slice(source.indexOf('async function poll()'), source.indexOf('poll(); setInterval')), context);
  vm.runInContext(source.slice(source.indexOf('async function backendAction('), source.indexOf('const clamp =')), context);
  return context;
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
