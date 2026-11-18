import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies /api to the backend (python -m srl, port 3199) so the browser sees
// one origin, the same as in production where the backend serves the built files.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5199,
    proxy: { "/api": { target: "http://127.0.0.1:3199", changeOrigin: false } },
  },
  // One chunk is fine for an app served from localhost. Raised from 1600 when the UI gained a
  // second language (i18next and both catalogs, 2026-10-29).
  build: { outDir: "dist", sourcemap: true, chunkSizeWarningLimit: 1900 },
});
