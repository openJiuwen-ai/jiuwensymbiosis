// Viewport input gestures, independent of WebRTC and camera state.
export class ViewportNavigation {
  constructor(video, {ready, send, cancel, onSpeed, onPick = () => {}, onFocus = () => {}, onClear = () => {}}) {
    this.video = video;
    this.ready = ready;
    this.send = send;
    this.cancel = cancel;
    this.onSpeed = onSpeed;
    this.onPick = onPick;
    this.onFocus = onFocus;
    this.onClear = onClear;
    this.window = video.ownerDocument.defaultView;
    this.keys = new Set();
    this.pointer = null;
    this.frame = null;
    this.speed = 1;
    this.dx = this.dy = 0;
    video.addEventListener('pointerdown', e => this.down(e));
    video.addEventListener('pointermove', e => this.move(e));
    video.addEventListener('pointerup', e => {
      const p = this.pointer;
      if (!p || p.id !== e.pointerId) return;
      const pick = !p.moved && (p.mode === 'select' || p.touch) && this.ready();
      this.stop();
      if (pick) this.onPick(e.clientX, e.clientY);
    });
    for (const name of ['pointercancel', 'lostpointercapture']) {
      video.addEventListener(name, e => { if (e.pointerId === this.pointer?.id) this.stop(); });
    }
    video.addEventListener('contextmenu', e => e.preventDefault());
    video.addEventListener('wheel', e => this.wheel(e), {passive: false});
    video.addEventListener('keydown', e => this.keydown(e));
    this.window.addEventListener('keyup', e => this.keys.delete(e.code));
    video.addEventListener('blur', () => this.stop());
    this.window.addEventListener('blur', () => this.stop());
    video.ownerDocument.addEventListener('visibilitychange', () => {
      if (video.ownerDocument.hidden) this.stop();
    });
  }

  stop() {
    const pointer = this.pointer;
    this.pointer = null;
    this.keys.clear();
    this.dx = this.dy = 0;
    if (this.frame !== null) this.window.cancelAnimationFrame(this.frame);
    this.frame = null;
    this.cancel();
    if (pointer && this.video.hasPointerCapture(pointer.id)) this.video.releasePointerCapture(pointer.id);
  }

  setSpeed(speed) {
    this.speed = Math.max(0.25, Math.min(4, speed));
    this.onSpeed(this.speed);
  }

  down(e) {
    if (!this.ready() || ![0, 1, 2].includes(e.button)) return;
    const held = new Set(this.keys);
    this.stop();
    this.video.focus({preventScroll: true});
    const mode = e.button === 2 ? 'fly' : e.button === 1 || e.shiftKey ? 'pan'
      : e.altKey || e.pointerType === 'touch' ? 'orbit' : 'select';
    this.pointer = {id: e.pointerId, button: e.button, mode, x: e.clientX, y: e.clientY,
      startX: e.clientX, startY: e.clientY, moved: false, touch: e.pointerType === 'touch'};
    this.video.setPointerCapture(e.pointerId);
    e.preventDefault();
    if (mode === 'fly') {
      this.keys = held;
      if (e.shiftKey) this.keys.add('ShiftLeft');
      this.lastTick = this.window.performance.now();
      this.frame = this.window.requestAnimationFrame(time => this.tick(time));
    }
  }

  move(e) {
    const p = this.pointer;
    if (!p || e.pointerId !== p.id) return;
    const mask = [1, 4, 2][p.button];
    if (!(e.buttons & mask) || !this.ready()) { this.stop(); return; }
    p.moved ||= Math.hypot(e.clientX - p.startX, e.clientY - p.startY) > 4;
    if (p.mode === 'select' || (p.touch && !p.moved)) return;
    const rect = this.video.getBoundingClientRect();
    const dx = (e.clientX - p.x) / Math.max(1, rect.width);
    const dy = (e.clientY - p.y) / Math.max(1, rect.height);
    p.x = e.clientX; p.y = e.clientY;
    if (p.mode === 'fly') { this.dx += dx; this.dy += dy; }
    else this.send({action: p.mode, dx, dy});
  }

  tick(time) {
    if (this.pointer?.mode !== 'fly' || !this.ready()) { this.stop(); return; }
    const elapsed = time - this.lastTick;
    if (elapsed >= 1000 / 30) {
      // Never catch up a suspended tab with a large jump.
      const dt = Math.min(elapsed / 1000, 0.05);
      this.lastTick = time;
      const held = code => Number(this.keys.has(code));
      const forward = held('KeyW') - held('KeyS');
      const strafe = held('KeyD') - held('KeyA');
      const lift = held('KeyE') - held('KeyQ');
      const modifier = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 3 : 1;
      const amount = dt * this.speed * modifier / Math.max(1, Math.hypot(forward, strafe, lift));
      if (this.dx || this.dy || forward || strafe || lift) {
        this.send({action: 'fly', dx: this.dx, dy: this.dy,
          forward: forward * amount, strafe: strafe * amount, lift: lift * amount});
        this.dx = this.dy = 0;
      }
    }
    this.frame = this.window.requestAnimationFrame(next => this.tick(next));
  }

  keydown(e) {
    if (e.code === 'Escape') { this.stop(); this.onClear(); return; }
    if (e.code === 'KeyF' && !e.isComposing && !e.metaKey && !e.ctrlKey && !e.altKey && this.pointer?.mode !== 'fly') {
      e.preventDefault();
      if (!e.repeat) { this.stop(); this.onFocus(); }
      return;
    }
    if (!this.ready() || e.isComposing || e.metaKey || e.ctrlKey) return;
    const movementKey = /^(Key[WASDQE]|ShiftLeft|ShiftRight)$/.test(e.code);
    if (movementKey) this.keys.add(e.code);
    if (this.pointer?.mode === 'fly') {
      if (movementKey) e.preventDefault();
      return;
    }
    if (e.ctrlKey || e.altKey) return;
    if (e.code === 'KeyH' || e.code === 'Home') {
      e.preventDefault(); this.stop(); this.send({action: 'reset'}); return;
    }
    const arrows = {ArrowLeft: [-0.04, 0], ArrowRight: [0.04, 0], ArrowUp: [0, -0.04], ArrowDown: [0, 0.04]};
    if (arrows[e.code]) {
      e.preventDefault();
      const [dx, dy] = arrows[e.code];
      this.send({action: e.shiftKey ? 'pan' : 'orbit', dx, dy});
    }
  }

  wheel(e) {
    e.preventDefault();
    if (!this.ready() || !e.deltaY) return;
    if (this.pointer?.mode === 'fly') {
      const speeds = [0.25, 1, 4];
      const index = speeds.indexOf(this.speed);
      this.setSpeed(speeds[Math.max(0, Math.min(2, index + (e.deltaY < 0 ? 1 : -1)))]);
    } else {
      this.send({action: 'zoom', delta: e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 540 : 1) / 600});
    }
  }
}
