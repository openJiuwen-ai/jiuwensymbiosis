import assert from 'node:assert/strict';
import test from 'node:test';
import {CounterRate, BackendMetrics} from '../../../../jiuwensymbiosis/simulators/genesis/web/assets/metrics.mjs';

const state = (overrides = {}) => ({state: 'ready', scene: {instance_id: 'room-a'}, config_revision: 1,
  measured_at: 10, steps: 1200, sim_time: 10, rendered_frames: 300,
  timings_ms: {render: {p95: 21.5}}, ...overrides});

test('counter rate measures actual deltas, including a stalled stream', () => {
  const rate = new CounterRate();
  assert.equal(rate.sample(60, 1000), null);
  assert.equal(rate.sample(105, 2000), 45);
  assert.equal(rate.sample(105, 3000), 0);
});

test('counter resets and invalid timestamps never produce invented rates', () => {
  const rate = new CounterRate();
  rate.sample(60, 1000);
  assert.equal(rate.sample(1, 2000), null);
  assert.equal(rate.sample(61, 2000), null);
  assert.equal(rate.sample(undefined, 3000), null);
  assert.equal(rate.sample(121, 4000), null);
});

test('physics, real time, and rendered rate use worker time rather than polling delay', () => {
  const metrics = new BackendMetrics();
  assert.equal(metrics.update(state(), 0).physicsHz, null);
  const result = metrics.update(state({measured_at: 12, steps: 1380, sim_time: 11.5, rendered_frames: 370}), 4000);
  assert.equal(result.physicsHz, 90);
  assert.equal(result.realTimeFactor, .75);
  assert.equal(result.renderFps, 35);
  assert.equal(result.renderP95Ms, 21.5);
});

test('cached samples age out and stop showing old performance', () => {
  const metrics = new BackendMetrics();
  metrics.update(state(), 0);
  metrics.update(state({measured_at: 11, steps: 1320, sim_time: 11}), 1000);
  assert.equal(metrics.update(state({measured_at: 11, steps: 1320, sim_time: 11}), 2000).physicsHz, 120);
  const stale = metrics.update(state({measured_at: 11, steps: 1320, sim_time: 11}), 4000);
  assert.equal(stale.fresh, false);
  assert.equal(stale.physicsHz, null);
  assert.equal(stale.renderP95Ms, null);
});

test('new scene, renderer, worker restart, and unavailable backend reset baselines', () => {
  for (const change of [{scene: {instance_id: 'room-b'}}, {config_revision: 2}, {measured_at: 1}]) {
    const metrics = new BackendMetrics();
    metrics.update(state(), 0);
    assert.equal(metrics.update(state({measured_at: 11, steps: 1320, ...change}), 1000).physicsHz, null);
  }
  const metrics = new BackendMetrics();
  metrics.update(state(), 0);
  const missing = metrics.update({state: 'unavailable'}, 1000);
  assert.equal(missing.fresh, false);
  assert.equal(missing.renderP95Ms, null);
});

test('missing timing and counters stay unknown', () => {
  const metrics = new BackendMetrics();
  const result = metrics.update(state({steps: undefined, sim_time: undefined, timings_ms: {}}), 0);
  assert.equal(result.physicsHz, null);
  assert.equal(result.realTimeFactor, null);
  assert.equal(result.renderP95Ms, null);
});
