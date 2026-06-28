/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#05080f",
          900: "#0a101d",
          800: "#111827",
          700: "#1f2937",
        },
        accent: {
          DEFAULT: "#10b981",
          soft: "#34d399",
          glow: "#6ee7b7",
        },
      },
      boxShadow: {
        panel: "0 20px 60px rgba(0,0,0,0.35)",
        glow: "0 0 40px rgba(16,185,129,0.15)",
      },
      backgroundImage: {
        grid: "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0)",
      },
    },
  },
  plugins: [],
};
