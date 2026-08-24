import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite rather than Next.js: this is an internal research dashboard, so there is no SSR or SEO
// requirement to pay for.
export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173, proxy: { "/api": "http://gateway:8000" } },
});
