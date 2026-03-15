import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/predict': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/history': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/stats': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/settings': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/static': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
    }
  }
})
