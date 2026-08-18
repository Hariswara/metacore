import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Vite rather than Next.js: this is an internal research dashboard, so there is no SSR or SEO
// requirement to pay for.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // Under compose the gateway is reachable by service name; running `pnpm dev` on a laptop it is
  // not. Default to localhost so a bare checkout works, and let compose override.
  const gateway = env.METACORE_GATEWAY_URL ?? "http://localhost:8000";

  return {
    plugins: [react()],
    server: { host: true, port: 5173, proxy: { "/api": gateway } },
  };
});
