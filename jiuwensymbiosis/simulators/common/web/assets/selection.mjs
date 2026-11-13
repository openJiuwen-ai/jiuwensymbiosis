// Pixel mapping and a projected world-space selection box, independent of transport.
export function contentRect(rect, width, height) {
  const scale = Math.min(rect.width / width, rect.height / height);
  return {left: rect.left + (rect.width - width * scale) / 2,
    top: rect.top + (rect.height - height * scale) / 2, width: width * scale, height: height * scale};
}

export function pickCoordinates(rect, width, height, x, y) {
  if (!width || !height) return null;
  const image = contentRect(rect, width, height);
  const u = (x - image.left) / image.width, v = (y - image.top) / image.height;
  return u >= 0 && u < 1 && v >= 0 && v < 1 ? {u, v} : null;
}

export function boxPath(bounds, camera, fovY, aspect) {
  const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
  const forward = camera.lookat.map((v, i) => v - camera.pos[i]);
  const length = Math.hypot(...forward);
  for (let i = 0; i < 3; i++) forward[i] /= length;
  const right = [-Math.sin(camera.yaw), Math.cos(camera.yaw), 0];
  const up = [-Math.cos(camera.yaw) * Math.sin(camera.pitch),
    -Math.sin(camera.yaw) * Math.sin(camera.pitch), Math.cos(camera.pitch)];
  const tanY = Math.tan(fovY * Math.PI / 360);
  const vertices = Array.from({length: 8}, (_, bits) => {
    const relative = [0, 1, 2].map(i => bounds[(bits >> i) & 1][i] - camera.pos[i]);
    return [dot(relative, right), dot(relative, up), dot(relative, forward)];
  });
  const project = p => [500 * (1 + p[0] / (p[2] * tanY * aspect)), 500 * (1 - p[1] / (p[2] * tanY))];
  const parts = [], near = 0.01;
  for (let i = 0; i < 8; i++) for (const bit of [1, 2, 4]) {
    if (i & bit) continue;
    let a = [...vertices[i]], b = [...vertices[i | bit]];
    if (a[2] < near && b[2] < near) continue;
    if (a[2] < near || b[2] < near) {
      if (a[2] > b[2]) [a, b] = [b, a];
      const t = (near - a[2]) / (b[2] - a[2]);
      a = a.map((v, k) => v + t * (b[k] - v));
    }
    const [x1, y1] = project(a), [x2, y2] = project(b);
    parts.push(`M${x1.toFixed(2)},${y1.toFixed(2)}L${x2.toFixed(2)},${y2.toFixed(2)}`);
  }
  return parts.join(' ');
}
