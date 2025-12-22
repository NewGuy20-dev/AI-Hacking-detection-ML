import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      boxShadow: {
        'clay': '6px 6px 12px rgba(0,0,0,0.08), -4px -4px 10px rgba(255,255,255,0.9), inset 1px 1px 2px rgba(255,255,255,0.5)',
        'clay-inset': 'inset 3px 3px 6px rgba(0,0,0,0.08), inset -2px -2px 5px rgba(255,255,255,0.9)',
        'clay-hover': '8px 8px 16px rgba(0,0,0,0.1), -6px -6px 14px rgba(255,255,255,0.95)',
        'clay-pressed': 'inset 4px 4px 8px rgba(0,0,0,0.1), inset -2px -2px 6px rgba(255,255,255,0.7)',
        'clay-dark': '6px 6px 12px rgba(0,0,0,0.4), -4px -4px 10px rgba(255,255,255,0.05), inset 1px 1px 2px rgba(255,255,255,0.05)',
        'clay-dark-inset': 'inset 3px 3px 6px rgba(0,0,0,0.4), inset -2px -2px 5px rgba(255,255,255,0.05)',
      },
      borderRadius: {
        'clay': '20px',
        'clay-sm': '12px',
        'clay-lg': '28px',
      },
      colors: {
        clay: {
          bg: '#e8eef5',
          card: '#f0f4f8',
          'bg-dark': '#1a202c',
          'card-dark': '#2d3748',
          danger: '#feb2b2',
          'danger-dark': '#fc8181',
          success: '#9ae6b4',
          warning: '#fbd38d',
          info: '#90cdf4',
          critical: '#e53e3e',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

export default config
