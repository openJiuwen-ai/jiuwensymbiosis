// Build the form from the same parameter contract the worker validates.
export function policyControls(form, submit) {
  const action = form.elements.namedItem('kind');
  const fields = form.querySelector('[data-fields]');
  const run = form.querySelector('[type="submit"]');
  const status = form.querySelector('output');
  let controls = [], binding = '', pending = false, disabled = true;

  function parameters() {
    fields.replaceChildren();
    const current = controls.find(item => item.name === action.value);
    for (const [name, spec] of Object.entries(current?.parameters.properties || {})) {
      const label = document.createElement('label');
      label.textContent = spec.description || name;
      let input;
      if (spec.enum || spec.type === 'boolean') {
        input = document.createElement('select');
        for (const value of spec.enum || [false, true]) {
          const text = spec.type === 'boolean' ? (value ? '闭合' : '张开') : value;
          input.add(new Option(text, String(value)));
        }
      } else {
        input = document.createElement('input');
        input.type = 'number';
        input.step = spec.type === 'integer' ? '1' : 'any';
        if (spec.minimum !== undefined) input.min = spec.minimum;
        if (spec.exclusiveMinimum !== undefined) input.min = spec.exclusiveMinimum;
        if (spec.maximum !== undefined) input.max = spec.maximum;
        input.required = true;
      }
      input.name = name;
      input.dataset.type = spec.type;
      if (spec.default !== undefined) input.value = String(spec.default);
      label.append(input);
      fields.append(label);
    }
  }
  action.onchange = parameters;
  form.onsubmit = async event => {
    event.preventDefault();
    if (pending || disabled || !action.value) return;
    pending = true;
    const started = binding;
    run.disabled = true;
    const params = {};
    for (const input of fields.querySelectorAll('input, select')) {
      params[input.name] = input.type === 'number' ? input.valueAsNumber :
        input.dataset.type === 'boolean' ? input.value === 'true' : input.value;
    }
    const label = action.selectedOptions[0].textContent;
    status.textContent = label + '执行中…';
    try {
      const result = await submit(action.value, params);
      if (started === binding) status.textContent = result.ok ? label + '已完成' : label + ' · ' + result.reason;
    } catch (error) {
      if (started === binding) status.textContent = error.message;
    } finally {
      pending = false;
      run.disabled = disabled;
    }
  };
  return (robot, sceneId, unavailable) => {
    const next = robot?.policy_controls || [];
    form.hidden = !next.length;
    if (sceneId !== binding || next.map(c => c.name).join() !== controls.map(c => c.name).join()) {
      binding = sceneId;
      controls = next;
      action.replaceChildren(...controls.map(c => new Option(c.label, c.name)));
      status.textContent = '';
      parameters();
    }
    disabled = unavailable || !next.length;
    run.disabled = disabled || pending;
    action.disabled = disabled || pending;
    fields.disabled = disabled || pending;
  };
}
