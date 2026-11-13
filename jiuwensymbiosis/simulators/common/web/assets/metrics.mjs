/** Rates from measured counter deltas. Settings are deliberately not inputs. */
export class CounterRate {
  constructor() { this.reset(); }
  reset() { this.previous = null; this.value = null; }
  sample(count, timeMs) {
    if (!Number.isFinite(count) || !Number.isFinite(timeMs) || count < 0) {
      this.reset();
      return null;
    }
    const previous = this.previous;
    this.previous = {count, timeMs};
    this.value = previous && count >= previous.count && timeMs > previous.timeMs
      ? (count - previous.count) * 1000 / (timeMs - previous.timeMs) : null;
    return this.value;
  }
}

export class BackendMetrics {
  constructor() { this.reset(); }
  reset() {
    this.physics = new CounterRate();
    this.simulation = new CounterRate();
    this.rendered = new CounterRate();
    this.identity = null;
    this.measuredAt = null;
    this.receivedAt = null;
    this.renderP95 = null;
  }
  update(state, nowMs) {
    if (state.state !== 'ready' || !Number.isFinite(state.measured_at)) {
      this.reset();
      return this.current(nowMs);
    }
    const identity = `${state.scene?.instance_id ?? ''}:${state.config_revision ?? ''}`;
    if (identity !== this.identity || state.measured_at < this.measuredAt) {
      this.reset();
      this.identity = identity;
    }
    // A cached worker sample does not extend freshness or create a new rate.
    if (state.measured_at !== this.measuredAt) {
      this.measuredAt = state.measured_at;
      this.receivedAt = nowMs;
      const timeMs = state.measured_at * 1000;
      this.physics.sample(state.steps, timeMs);
      this.simulation.sample(state.sim_time, timeMs);
      this.rendered.sample(state.rendered_frames, timeMs);
      const p95 = state.timings_ms?.render?.p95;
      this.renderP95 = Number.isFinite(p95) && p95 >= 0 ? p95 : null;
    }
    return this.current(nowMs);
  }
  current(nowMs) {
    const fresh = this.receivedAt !== null && nowMs - this.receivedAt <= 2500;
    return {
      physicsHz: fresh ? this.physics.value : null,
      realTimeFactor: fresh ? this.simulation.value : null,
      renderFps: fresh ? this.rendered.value : null,
      renderP95Ms: fresh ? this.renderP95 : null,
      fresh,
    };
  }
}
