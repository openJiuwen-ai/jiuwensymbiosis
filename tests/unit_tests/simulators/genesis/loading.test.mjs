import test from 'node:test';
import assert from 'node:assert/strict';
import {loadingView} from '../../../../jiuwensymbiosis/simulators/genesis/web/assets/loading.mjs';

test('opaque kernel build shows its phase and elapsed time without an invented percentage', () => {
  const view = loadingView({state: 'running', current_phase: 'scene_build', total_ms: 12400});
  assert.equal(view.value, null);
  assert.equal(view.label, '构建物理世界与计算内核');
  assert.equal(view.elapsed, '已用时 12.4 秒');
});

test('collision progress uses the completed entity count, not a whole-load estimate', () => {
  const view = loadingView({state: 'running', current_phase: 'collision_entities', progress: {completed: 12, total: 33}});
  assert.equal(view.value, 12 / 33);
  assert.equal(view.label, '构建场景碰撞体 · 12/33');
});

test('completion hides progress and failure retains the failing phase', () => {
  assert.equal(loadingView({state: 'complete'}).visible, false);
  assert.equal(loadingView({state: 'failed', failed_phase: 'reset_stop'}).label, '失败：确认机器人停稳');
});
