/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'Plus Jakarta Sans'", "Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      colors: {
        brand: {
          dark: "#030712",
          slate: "#0b0f19",
          card: "rgba(15, 23, 42, 0.4)",
        },
        accent: {
          DEFAULT: "#8b5cf6",
          soft: "#a78bfa",
          glow: "#c084fc",
        },
        cyan: {
          glow: "#22d3ee",
        },
        status: {
          success: "#10b981",
          warning: "#f59e0b",
          error: "#f43f5e",
        }
      },
      boxShadow: {
        panel: "0 25px 70px -10px rgba(0, 0, 0, 0.7)",
        glow: "0 0 50px -5px rgba(139, 92, 246, 0.25)",
        cyanGlow: "0 0 50px -5px rgba(6, 182, 212, 0.25)",
        successGlow: "0 0 40px -5px rgba(16, 185, 129, 0.2)",
      },
      backgroundImage: {
        grid: "radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.03) 1px, transparent 0)",
        glowGradient: "radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.15) 0%, transparent 60%)",
        cyanGradient: "radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.12) 0%, transparent 60%)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      }
    },
  },
  plugins: [],
};
