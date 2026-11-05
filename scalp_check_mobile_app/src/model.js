// Runs the trained ONNX model on the phone (no internet, no server).
// Reads the repo's artifacts (copied into public/models/ by scripts/sync-model.mjs):
//   the model named by "model_file" in preprocess_config.json, labels.json,
//   preprocess_config.json, calibration.json
import * as ort from 'onnxruntime-web/wasm';

// If the JSON files use names this code does not recognise, or src/predict.py
// preprocesses differently, set the exact values here. null = read from JSON.
export const OVERRIDES = {
  size: null,          // final square size fed to the model, e.g. 224
  resizeShorter: null, // shorter side before centre crop, e.g. 256. 0 = squash to size x size
  mean: null,          // e.g. [0.485, 0.456, 0.406]
  std: null,           // e.g. [0.229, 0.224, 0.225]
  temperature: null,   // e.g. 1.37
  threshold: null,     // reject threshold on the top probability, e.g. 0.6
};
const DEFAULT_THRESHOLD = 0.5; // used only if no threshold is found anywhere

// The WebAssembly file is bundled by Vite into dist/assets, so it works offline.
ort.env.wasm.numThreads = 1; // the Android WebView cannot use multi-threaded WASM
ort.env.logLevel = 'error';

let session = null;
let loading = null;
export const info = { modelFile: null, labels: [], pre: null, temperature: 1, threshold: DEFAULT_THRESHOLD, warnings: [], layout: 'NCHW' };

const url = (f) => new URL('models/' + f, document.baseURI).href;
async function getJson(f) {
  const r = await fetch(url(f));
  if (!r.ok) throw new Error(`Missing model file: ${f}. Run "npm run sync-model".`);
  return r.json();
}

// Search an object (any nesting) for the first value under one of the keys.
function find(obj, keys, test = () => true) {
  if (!obj || typeof obj !== 'object') return undefined;
  for (const k of Object.keys(obj)) {
    if (keys.includes(k.toLowerCase()) && test(obj[k])) return obj[k];
  }
  for (const v of Object.values(obj)) {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      const r = find(v, keys, test);
      if (r !== undefined) return r;
    }
  }
  return undefined;
}
const isNum = (v) => typeof v === 'number' && isFinite(v);
const isTriple = (v) => Array.isArray(v) && v.length === 3 && v.every(isNum);

function parseLabels(j) {
  if (Array.isArray(j)) return j.map(String);
  const list = find(j, ['labels', 'classes', 'class_names', 'names'], Array.isArray);
  if (list) return list.map(String);
  const map = find(j, ['idx_to_class', 'index_to_label', 'id2label'], (v) => v && typeof v === 'object');
  const src = map || j;
  const entries = Object.entries(src);
  if (entries.every(([k]) => /^\d+$/.test(k))) return entries.sort((a, b) => a[0] - b[0]).map(([, v]) => String(v));
  if (entries.every(([, v]) => Number.isInteger(v))) return entries.sort((a, b) => a[1] - b[1]).map(([k]) => k);
  throw new Error('Could not read labels.json');
}

function parsePre(j) {
  let size = OVERRIDES.size ?? find(j, ['input_size', 'img_size', 'image_size', 'crop_size', 'size', 'crop'], (v) => isNum(v) || Array.isArray(v));
  if (Array.isArray(size)) size = size[size.length - 1];
  if (!isNum(size)) { size = 224; info.warnings.push('Image size not found; using 224.'); }

  let resize = OVERRIDES.resizeShorter ?? find(j, ['resize', 'resize_size', 'resize_shorter', 'shorter_side'], (v) => isNum(v) || Array.isArray(v));
  if (Array.isArray(resize)) resize = 0; // fixed (h, w) resize means squash
  if (!isNum(resize)) {
    const pct = find(j, ['crop_pct', 'crop_ratio'], isNum);
    resize = isNum(pct) ? Math.floor(size / pct) : 0;
  }

  let mean = OVERRIDES.mean ?? find(j, ['mean', 'norm_mean', 'normalize_mean'], isTriple);
  let std = OVERRIDES.std ?? find(j, ['std', 'norm_std', 'normalize_std'], isTriple);
  if (!mean || !std) { mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]; info.warnings.push('Mean/std not found; using ImageNet values.'); }
  if (mean.some((m) => m > 1)) { mean = mean.map((m) => m / 255); std = std.map((s) => s / 255); }
  const bgr = String(find(j, ['channel_order', 'color_order', 'channels'], (v) => typeof v === 'string') || 'RGB').toUpperCase() === 'BGR';
  return { size, resize, mean, std, bgr };
}

function parseCalib(j) {
  const T = OVERRIDES.temperature ?? find(j, ['temperature', 't', 'temp', 'optimal_temperature', 'best_t'], isNum);
  const thr = OVERRIDES.threshold ?? find(j, ['threshold', 'reject_threshold', 'tau', 'conf_threshold', 'confidence_threshold', 'reject_tau'], isNum);
  if (!isNum(T)) info.warnings.push('Temperature not found; using 1 (no calibration).');
  if (!isNum(thr)) info.warnings.push(`Reject threshold not found; using ${DEFAULT_THRESHOLD}.`);
  return { temperature: isNum(T) && T > 0 ? T : 1, threshold: isNum(thr) ? thr : DEFAULT_THRESHOLD };
}

export function loadModel() {
  if (loading) return loading;
  loading = (async () => {
    const [labels, pre, calib] = await Promise.all([getJson('labels.json'), getJson('preprocess_config.json'), getJson('calibration.json')]);
    info.labels = parseLabels(labels);
    info.pre = parsePre(pre);
    Object.assign(info, parseCalib(calib));
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    // Use the file the export step chose ("model_file" in preprocess_config.json).
    // In this repo that is model_fp32.onnx: INT8 lost about 12 macro-F1 points.
    const chosen = typeof pre.model_file === 'string' ? pre.model_file : 'model_fp32.onnx';
    const files = [chosen, ...['model_fp32.onnx', 'model_int8.onnx'].filter((f) => f !== chosen)];
    for (const f of files) {
      try {
        session = await ort.InferenceSession.create(url(f), opts);
        info.modelFile = f;
        break;
      } catch (e) {
        if (f === chosen) info.warnings.push(`${f} could not load: ${e.message || e}`);
      }
    }
    if (!session) throw new Error('No model could be loaded. ' + info.warnings.join(' '));
    const meta = session.inputMetadata?.[0];
    const dims = meta?.shape || meta?.dimensions;
    if (Array.isArray(dims) && dims.length === 4 && dims[3] === 3) info.layout = 'NHWC';
    return info;
  })();
  loading.catch(() => { loading = null; });
  return loading;
}

// Crop/resize the photo exactly like the training evaluation transform.
// Returns the square canvas the model sees (also used for display).
export function prepareCanvas(source) {
  const { size, resize } = info.pre;
  const w = source.width, h = source.height;
  let sx = 0, sy = 0, sw = w, sh = h;
  if (resize > 0) {
    // Same as src/predict.py: resize the shorter side to `resize` (other side
    // truncated to an integer), then crop size x size at rounded centre offsets.
    const nw = w <= h ? resize : Math.floor((resize * w) / h);
    const nh = w <= h ? Math.floor((resize * h) / w) : resize;
    const left = Math.round((nw - size) / 2), top = Math.round((nh - size) / 2);
    const fx = w / nw, fy = h / nh;
    sx = left * fx; sy = top * fy; sw = size * fx; sh = size * fy;
  }
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, size, size);
  return c;
}

export function toTensorData(canvas) {
  const { size, mean, std, bgr } = info.pre;
  const px = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, size, size).data;
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = (px[i * 4 + (bgr ? 2 - c : c)] / 255 - mean[c]) / std[c];
      if (info.layout === 'NHWC') out[i * 3 + c] = v; else out[c * n + i] = v;
    }
  }
  return out;
}

function softmax(logits, T) {
  // If the model already outputs probabilities, take logs so temperature still applies.
  const sum = logits.reduce((a, b) => a + b, 0);
  const isProb = logits.every((v) => v >= 0 && v <= 1) && Math.abs(sum - 1) < 1e-3;
  const z = isProb ? logits.map((p) => Math.log(Math.max(p, 1e-12))) : logits;
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp((v - m) / T));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export async function runTensor(data) {
  const { size } = info.pre;
  const shape = info.layout === 'NHWC' ? [1, size, size, 3] : [1, 3, size, size];
  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', data, shape) };
  const out = await session.run(feeds);
  const logits = Array.from(out[session.outputNames[0]].data);
  return softmax(logits, info.temperature);
}

// Main entry: photo (ImageBitmap / image / canvas) -> result.
export async function predict(source) {
  await loadModel();
  const canvas = prepareCanvas(source);
  const data = toTensorData(canvas);
  const probs = await runTensor(data);
  const ranked = probs.map((p, i) => ({ label: info.labels[i] ?? `Class ${i}`, index: i, p })).sort((a, b) => b.p - a.p);
  return { canvas, data, probs, top: ranked.slice(0, 3), sure: ranked[0].p >= info.threshold, threshold: info.threshold };
}
