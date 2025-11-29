import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,   //frontend port
    proxy: {
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      },
      '/upload_audio': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})