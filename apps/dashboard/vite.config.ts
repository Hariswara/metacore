import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite rather than Next.js: this is an internal research dashboard, so there is no SSR or SEO
// requirement to pay for.
//
// Local `pnpm dev` talks to uvicorn on this machine. Compose overrides GATEWAY_URL so the
// dashboard container can reach the `gateway` service by Docker DNS.
const gatewayUrl = process.env.GATEWAY_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173, proxy: { "/api": gatewayUrl } },
});
