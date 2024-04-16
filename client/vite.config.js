// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    assetsDir: "static",
  },
  server: {
    port: 3000,
    cors: true,
    proxy: {
      "/api": {
        target: "http://localhost:5000/", // Replace with your Flask backend URL
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
