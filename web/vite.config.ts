import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8001",
      "/health": "http://127.0.0.1:8001",
      "/models": "http://127.0.0.1:8001",
      "/acquisitions": "http://127.0.0.1:8001"
    }
  }
});
