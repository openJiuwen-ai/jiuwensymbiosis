import test from 'node:test';
import assert from 'node:assert/strict';
import {contentRect, pickCoordinates, boxPath} from '../../../../jiuwensymbiosis/simulators/genesis/web/assets/selection.mjs';

test('object-fit contain mapping rejects black bars and preserves framebuffer coordinates', () => {
  const rect = {left: 30, top: 200, width: 900, height: 300};
  const image = contentRect(rect, 1280, 720);
  assert.ok(Math.abs(image.width - 533.333333) < .001);
  assert.deepEqual(pickCoordinates(rect, 1280, 720, 480, 350), {u: .5, v: .5});
  assert.equal(pickCoordinates(rect, 1280, 720, 40, 350), null);
  assert.ok(Math.abs(pickCoordinates(rect, 1280, 720, image.left + image.width * .75, 350).u - .75) < 1e-12);
  const tall = {left: 0, top: 0, width: 400, height: 600};
  assert.equal(pickCoordinates(tall, 1920, 1080, 200, 20), null);
  assert.deepEqual(pickCoordinates(tall, 1920, 1080, 200, 300), {u: .5, v: .5});
});

test('selection projects twelve edges and clips geometry behind the camera', () => {
  const camera = {pos: [3, 0, 0], lookat: [0, 0, 0], yaw: 0, pitch: 0};
  const path = boxPath([[-.5, -.5, -.5], [.5, .5, .5]], camera, 60, 16 / 9);
  assert.equal((path.match(/M/g) || []).length, 12);
  assert.ok(!path.includes('NaN') && !path.includes('Infinity'));
  assert.equal(boxPath([[4, -1, -1], [5, 1, 1]], camera, 60, 16 / 9), '');
  const crossing = boxPath([[2, -1, -1], [5, 1, 1]], camera, 60, 16 / 9);
  assert.ok(crossing && !crossing.includes('NaN') && !crossing.includes('Infinity'));
});
