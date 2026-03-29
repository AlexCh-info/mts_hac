import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  envPrefix: 'VITE_',
  define: {
    'process.env': {}
  },
  resolve: {
    alias: {
      'mapbox-gl': 'mapbox-gl'
    }
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'redux', 'react-redux', 'styled-components'],
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          kepler: ['@kepler.gl/components', '@kepler.gl/reducers', '@kepler.gl/actions']
        }
      }
    }
  }
})