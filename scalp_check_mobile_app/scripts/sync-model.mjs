// Copies the trained model files from the repository's artifacts/ folder
// into public/models/ so they are packaged inside the app.
// It only READS from ../artifacts. It never changes the repository's files.
import { cpSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const artifacts = process.env.SCALP_ARTIFACTS || join(root, '..', 'artifacts');
const dest = join(root, 'public', 'models');

import { readFileSync } from 'node:fs';
const CONFIGS = ['labels.json', 'preprocess_config.json', 'calibration.json'];

if (!existsSync(artifacts)) {
  console.warn(
    `[sync-model] No artifacts folder at ${artifacts}.\n` +
      '  Put mobile_app/ inside the scalp_cv repo, or set SCALP_ARTIFACTS to the folder path.\n' +
      '  Keeping whatever is already in public/models/.'
  );
  process.exit(0);
}

// Ship only the model file the export step chose ("model_file" in preprocess_config.json).
let chosen = 'model_fp32.onnx';
try {
  chosen = JSON.parse(readFileSync(join(artifacts, 'preprocess_config.json'), 'utf8')).model_file || chosen;
} catch {}
const FILES = [chosen, ...CONFIGS];

mkdirSync(dest, { recursive: true });
let copied = 0;
for (const f of FILES) {
  const from = join(artifacts, f);
  if (!existsSync(from)) {
    console.warn(`[sync-model] missing: ${f}`);
    continue;
  }
  cpSync(from, join(dest, f));
  copied++;
  const mb = (statSync(from).size / 1e6).toFixed(1);
  console.log(`[sync-model] ${f} (${mb} MB)`);
}
console.log(`[sync-model] copied ${copied}/${FILES.length} file(s) to public/models/`);
