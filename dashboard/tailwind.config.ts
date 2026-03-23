import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: '#050505',
          card: '#0a0a0a',
          border: '#1a1f2e',
          text: '#e2e8f0',
          muted: '#8b949e',
        },
        primary: '#00f3ff', // Neon Cyan
        primaryHover: '#00c3cc',
        secondary: '#bf00ff', // Neon Purple
        success: '#00ff66',
        warning: '#ffb000',
        danger: '#ff003c',
        info: '#00aeff',
      },
      boxShadow: {
        'neon-primary': '0 0 10px rgba(0, 243, 255, 0.5), inset 0 0 10px rgba(0, 243, 255, 0.1)',
        'neon-primary-strong': '0 0 20px rgba(0, 243, 255, 0.8), inset 0 0 15px rgba(0, 243, 255, 0.2)',
        'neon-danger': '0 0 10px rgba(255, 0, 60, 0.5), inset 0 0 10px rgba(255, 0, 60, 0.1)',
        'neon-success': '0 0 10px rgba(0, 255, 102, 0.5), inset 0 0 10px rgba(0, 255, 102, 0.1)',
      },
      borderRadius: {
        'cyber': '2px', // Sharp, tactical corners
      },
      backgroundImage: {
        'cyber-grid': 'linear-gradient(rgba(0, 243, 255, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 243, 255, 0.05) 1px, transparent 1px)',
        'cyber-scanline': 'linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0,0,0,0.2) 50%, rgba(0,0,0,0.2))',
      },
      fontFamily: {
        sans: ['var(--font-chakra)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-fira-code)', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glitch': 'glitch 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94) both infinite',
        'scan': 'scan 8s linear infinite',
      },
      keyframes: {
        glitch: {
          '0%': { transform: 'translate(0)' },
          '20%': { transform: 'translate(-2px, 2px)' },
          '40%': { transform: 'translate(-2px, -2px)' },
          '60%': { transform: 'translate(2px, 2px)' },
          '80%': { transform: 'translate(2px, -2px)' },
          '100%': { transform: 'translate(0)' }
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' }
        }
      }
    },
  },
  plugins: [],
}

export default config
