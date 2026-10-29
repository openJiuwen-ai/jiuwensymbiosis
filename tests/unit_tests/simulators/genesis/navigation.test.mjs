// Run with node --test tests/unit_tests/simulators/genesis/navigation.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {ViewportNavigation} from '../../../../jiuwensymbiosis/simulators/genesis/web/assets/navigation.mjs';

function harness() {
  const win = new EventTarget(), doc = new EventTarget(), video = new EventTarget();
  const frames = new Map(), commands = [], speeds = [], selections = [];
  let time = 0, next = 0, captured = null, ready = true, cancelled = 0;
  win.performance = {now: () => time};
  win.requestAnimationFrame = fn => { frames.set(++next, fn); return next; };
  win.cancelAnimationFrame = id => frames.delete(id);
  doc.defaultView = win; doc.hidden = false;
  video.ownerDocument = doc;
  video.focus = () => {};
  video.getBoundingClientRect = () => ({width: 1000, height: 500});
  video.setPointerCapture = id => { captured = id; };
  video.hasPointerCapture = id => captured === id;
  video.releasePointerCapture = () => { captured = null; };
  const navigation = new ViewportNavigation(video, {
    ready: () => ready, send: cmd => commands.push(cmd),
    cancel: () => { cancelled++; }, onSpeed: speed => speeds.push(speed),
    onPick: (x, y) => selections.push({x, y}), onFocus: () => selections.push('focus'), onClear: () => selections.push('clear'),
  });
  const emit = (target, type, fields = {}) => {
    const event = new Event(type, {cancelable: true});
    Object.assign(event, fields); target.dispatchEvent(event); return event;
  };
  return {navigation, video, win, doc, commands, speeds, selections,
    emit: (type, fields) => emit(video, type, fields),
    global: (type, fields) => emit(win, type, fields),
    down: (button = 2, fields = {}) => emit(video, 'pointerdown', {
      button, pointerId: 1, clientX: 200, clientY: 200, pointerType: 'mouse', ...fields,
    }),
    tick(ms = 40) {
      time += ms;
      const due = [...frames.values()]; frames.clear(); due.forEach(fn => fn(time));
    },
    unavailable() { ready = false; },
    hide() { doc.hidden = true; emit(doc, 'visibilitychange'); },
    get pending() { return frames.size; }, get cancelled() { return cancelled; },
  };
}

test('Viewport mouse mapping: Alt orbit, middle pan, right look; plain left does not navigate', () => {
  for (const [button, modifiers, expected] of [[0, {}, null], [0, {altKey: true}, 'orbit'],
    [1, {}, 'pan'], [0, {shiftKey: true}, 'pan'], [2, {}, 'fly']]) {
    const h = harness(); h.down(button, modifiers);
    h.emit('pointermove', {pointerId: 1, buttons: [1, 4, 2][button], clientX: 250, clientY: 220});
    h.tick();
    assert.equal(h.commands.at(-1)?.action ?? null, expected);
    if (expected) { assert.equal(h.commands[0].dx, 0.05); assert.equal(h.commands[0].dy, 0.04); }
  }
});

test('simultaneous look and flight share one command; diagonal speed is normalized', () => {
  const h = harness(); h.down();
  for (const code of ['KeyW', 'KeyD', 'KeyE']) h.emit('keydown', {code});
  h.emit('pointermove', {pointerId: 1, buttons: 2, clientX: 250, clientY: 220});
  h.tick();
  const c = h.commands[0];
  assert.equal(c.action, 'fly'); assert.equal(c.dx, 0.05); assert.equal(c.dy, 0.04);
  assert.ok(Math.abs(Math.hypot(c.forward, c.strafe, c.lift) - 0.04) < 1e-9);
  h.global('keyup', {code: 'KeyW'}); h.global('keyup', {code: 'KeyD'}); h.global('keyup', {code: 'KeyE'});
  h.tick(); assert.equal(h.commands.length, 1);
});

test('movement is elapsed-time based, with no catch-up after a stall', () => {
  const travel = step => {
    const h = harness(); h.down(); h.emit('keydown', {code: 'KeyW'});
    for (let t = 0; t < 1000; t += step) h.tick(step);
    return h.commands.reduce((sum, c) => sum + c.forward, 0);
  };
  assert.ok(Math.abs(travel(10) - travel(40)) < 1e-9);
  const h = harness(); h.down(); h.emit('keydown', {code: 'KeyW'}); h.tick(2000);
  assert.equal(h.commands[0].forward, 0.05);
});

test('release, cancel, lost capture, blur, hidden tab and unavailable backend stop movement', () => {
  for (const end of [h => h.emit('pointerup', {pointerId: 1}),
    h => h.emit('pointercancel', {pointerId: 1}), h => h.emit('lostpointercapture', {pointerId: 1}),
    h => h.emit('blur'), h => h.global('blur'), h => h.hide(), h => h.emit('keydown', {code: 'Escape'}),
    h => h.unavailable()]) {
    const h = harness(); h.down(); h.emit('keydown', {code: 'KeyW'}); h.tick();
    end(h); h.tick(); h.tick();
    assert.equal(h.commands.length, 1); assert.equal(h.pending, 0);
    assert.equal(h.navigation.keys.size, 0); assert.ok(h.cancelled > 1);
  }
});

test('keys outside the viewport and WASD without right button do not move camera', () => {
  const h = harness(); h.global('keydown', {code: 'KeyW'}); h.down(); h.tick();
  assert.equal(h.commands.length, 0);
  h.emit('pointerup', {pointerId: 1}); h.emit('keydown', {code: 'KeyW'}); h.tick();
  assert.equal(h.commands.length, 0);
  h.down(); h.tick(); assert.equal(h.commands[0].forward, 0.04); // Key held before RMB works.
});

test('Q descends, E ascends; shift, speed selector and RMB wheel change fly speed', () => {
  const h = harness(); h.down(2, {shiftKey: true}); h.emit('keydown', {code: 'KeyQ'}); h.tick();
  assert.equal(h.commands[0].lift, -0.12);
  h.global('keyup', {code: 'ShiftLeft'}); h.global('keyup', {code: 'KeyQ'});
  h.navigation.setSpeed(0.25); h.emit('keydown', {code: 'KeyE'}); h.tick();
  assert.equal(h.commands[1].lift, 0.01);
  h.emit('wheel', {deltaY: -100}); assert.equal(h.navigation.speed, 1);
  assert.deepEqual(h.speeds, [0.25, 1]);
});

test('Home resets, arrows remain accessible, and ordinary wheel zooms', () => {
  const h = harness(); h.emit('keydown', {code: 'Home'});
  h.emit('keydown', {code: 'ArrowRight'}); h.emit('keydown', {code: 'ArrowRight', shiftKey: true});
  h.emit('wheel', {deltaY: 120, deltaMode: 0});
  assert.deepEqual(h.commands.map(c => c.action), ['reset', 'orbit', 'pan', 'zoom']);
});

test('left click selects on release; drag, cancel and orbit do not select', () => {
  const h = harness(); h.down(0);
  assert.equal(h.selections.length, 0);
  h.emit('pointerup', {pointerId: 1, clientX: 200, clientY: 200});
  assert.deepEqual(h.selections, [{x: 200, y: 200}]);
  for (const fields of [{}, {altKey: true}, {shiftKey: true}]) {
    h.down(0, fields);
    h.emit('pointermove', {pointerId: 1, buttons: 1, clientX: 250, clientY: 220});
    h.emit('pointerup', {pointerId: 1, clientX: 250, clientY: 220});
  }
  h.down(0); h.emit('pointercancel', {pointerId: 1});
  assert.equal(h.selections.length, 1);
});

test('F focuses once, Escape clears, and F outside video does not focus', () => {
  const h = harness(); h.emit('keydown', {code: 'KeyF'});
  h.emit('keydown', {code: 'KeyF', repeat: true});
  h.global('keydown', {code: 'KeyF'});
  h.emit('keydown', {code: 'Escape'});
  assert.deepEqual(h.selections, ['focus', 'clear']);
});
