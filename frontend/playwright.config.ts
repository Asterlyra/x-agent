import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:41731", trace: "retain-on-failure" },
  webServer: {
    command: "pnpm dev --host 127.0.0.1 --port 41731 --strictPort",
    url: "http://127.0.0.1:41731",
    reuseExistingServer: false,
  },
})
