import test from 'node:test';
import assert from 'node:assert/strict';
import {loadingView} from '../../../../jiuwensymbiosis/simulators/common/web/assets/loading.mjs';

test('opaque kernel build shows its phase and elapsed time without an invented percentage', () => {
  const view = loadingView({state: 'running', current_phase: 'scene_build', total_ms: 12400});
  assert.equal(view.value, null);
  assert.equal(view.label, '构建物理世界与计算内核');
  assert.equal(view.elapsed, '已用时 12.4 秒');
});

test('single floor loading does not report removed house collision entities', () => {
  const view = loadingView({state: 'running', current_phase: 'single_floor'});
  assert.equal(view.value, null);
  assert.equal(view.label, '创建单层地面');
});

test('completion hides progress and failure retains the failing phase', () => {
  assert.equal(loadingView({state: 'complete'}).visible, false);
  assert.equal(loadingView({state: 'failed', failed_phase: 'reset_stop'}).label, '失败：确认机器人停稳');
});
