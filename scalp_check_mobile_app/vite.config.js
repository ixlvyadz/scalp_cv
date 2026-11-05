import { defineConfig } from 'vite';

// base './' makes every asset path relative, which Capacitor needs.
export default defineConfig({
  base: './',
  build: { outDir: 'dist', target: 'es2020', assetsInlineLimit: 0 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
