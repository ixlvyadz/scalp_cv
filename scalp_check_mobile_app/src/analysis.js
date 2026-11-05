import { runTensor, info } from './model.js';

// ---------- Photo quality (brightness and sharpness) ----------
// Simple checks used only to explain results ("too dark", "blurry").
export function checkQuality(canvas) {
  const S = 128;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(canvas, 0, 0, S, S);
  const d = ctx.getImageData(0, 0, S, S).data;
  const g = new Float32Array(S * S);
  let sum = 0;
  for (let i = 0; i < S * S; i++) {
    g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    sum += g[i];
  }
  const brightness = sum / (S * S);
  // Variance of the Laplacian: low = blurry.
  let m = 0, m2 = 0, n = 0;
  for (let y = 1; y < S - 1; y++) {
    for (let x = 1; x < S - 1; x++) {
      const i = y * S + x;
      const l = g[i - S] + g[i + S] + g[i - 1] + g[i + 1] - 4 * g[i];
      m += l; m2 += l * l; n++;
    }
  }
  const sharpness = m2 / n - (m / n) ** 2;
  return {
    brightness,
    sharpness,
    bright: brightness >= 60, // 0-255 average
    clear: sharpness >= 20,
  };
}

// ---------- Where the app looked ----------
// Occlusion sensitivity: hide one part of the photo at a time and measure how
// much the app's confidence in its answer drops. Bigger drop = the app relied
// on that area more. Runs on the phone with the same model (about 25 passes).
export async function lookedMap(data, classIndex, baseProb, onProgress) {
  const S = info.pre.size;
  const N = 6;            // 6 x 6 grid of cells
  const W = 2;            // the hidden window covers 2 x 2 cells
  const cell = S / N;
  const heat = new Float32Array(N * N);
  const hits = new Float32Array(N * N);
  const nhwc = info.layout === 'NHWC';
  const n = S * S;
  const total = (N - W + 1) ** 2;
  let done = 0;
  for (let gy = 0; gy <= N - W; gy++) {
    for (let gx = 0; gx <= N - W; gx++) {
      const x0 = Math.round(gx * cell), x1 = Math.round((gx + W) * cell);
      const y0 = Math.round(gy * cell), y1 = Math.round((gy + W) * cell);
      const copy = data.slice();
      // 0 after normalisation = the average (mean) colour.
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = y * S + x;
          if (nhwc) { copy[i * 3] = 0; copy[i * 3 + 1] = 0; copy[i * 3 + 2] = 0; }
          else { copy[i] = 0; copy[n + i] = 0; copy[2 * n + i] = 0; }
        }
      }
      const p = (await runTensor(copy))[classIndex];
      const drop = Math.max(0, baseProb - p);
      for (let cy = gy; cy < gy + W; cy++) {
        for (let cx = gx; cx < gx + W; cx++) { heat[cy * N + cx] += drop; hits[cy * N + cx]++; }
      }
      done++;
      onProgress?.(done / total);
      await new Promise((r) => setTimeout(r, 0)); // keep the screen responsive
    }
  }
  for (let i = 0; i < heat.length; i++) heat[i] /= hits[i] || 1;
  const max = Math.max(...heat);
  return { grid: N, values: Array.from(heat, (v) => (max > 0 ? v / max : 0)) };
}

// Blue -> teal -> yellow ramp from the design kit.
const STOPS = [[0x1b, 0x3a, 0x8a], [0x1b, 0x8a, 0x7a], [0xf2, 0xc9, 0x4c]];
function ramp(t) {
  const x = Math.min(1, Math.max(0, t)) * 2;
  const i = Math.min(1, Math.floor(x));
  const f = x - i;
  return STOPS[i].map((a, k) => Math.round(a + (STOPS[i + 1][k] - a) * f));
}

// Draws the photo with the heat map on top, smoothly upscaled.
export function drawOverlay(photoCanvas, map) {
  const S = photoCanvas.width;
  const out = document.createElement('canvas');
  out.width = out.height = S;
  const ctx = out.getContext('2d');
  ctx.drawImage(photoCanvas, 0, 0);
  const small = document.createElement('canvas');
  small.width = small.height = map.grid;
  const sctx = small.getContext('2d');
  const img = sctx.createImageData(map.grid, map.grid);
  map.values.forEach((v, i) => {
    const [r, g, b] = ramp(v);
    img.data.set([r, g, b, Math.round(70 + 150 * v)], i * 4);
  });
  sctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, S, S);
  return out;
}
