import { useEffect, useRef } from "react";

type Props = {
  message: string;
  submessage?: string;
};

export function WaveLoader({ message, submessage }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let phase = 0;

    const resize = () => {
      if (canvas.parentElement) {
        canvas.width = canvas.parentElement.clientWidth || 600;
      } else {
        canvas.width = 600;
      }
      canvas.height = 150;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const width = canvas.width;
      const height = canvas.height;
      const centerY = height / 2;

      phase += 0.025; // Speed of propagation

      const waveCount = 15; // Number of parallel contour waves
      for (let w = 0; w < waveCount; w++) {
        const progress = w / waveCount;
        ctx.beginPath();
        ctx.lineWidth = 1.2;
        
        // Edge fade envelope so waves naturally taper off at edges
        // Combined with phase shifting per wave to create ribbon contours
        for (let x = 0; x < width; x++) {
          const edgeFade = Math.sin((x / width) * Math.PI);
          
          // Primary slow wave + secondary fast wave
          const freq1 = 0.006 + (w * 0.0002);
          const freq2 = 0.015 - (w * 0.0001);
          
          const amp1 = 35 * Math.sin(phase * 0.4 + progress * Math.PI);
          const amp2 = 12 * Math.cos(phase * 0.9 - progress * Math.PI * 0.5);

          const y = centerY + 
            (amp1 * Math.sin(x * freq1 + phase) + amp2 * Math.sin(x * freq2 - phase)) * edgeFade;

          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        // Draw glowing violet-blue-cyan gradient lines
        const grad = ctx.createLinearGradient(0, 0, width, 0);
        const opacity = (1.0 - Math.abs(progress - 0.5) * 1.6) * 0.35; // Brightest in the center
        grad.addColorStop(0, `rgba(139, 92, 246, ${opacity * 0.2})`);   // Violet start fade
        grad.addColorStop(0.3, `rgba(167, 139, 250, ${opacity})`);       // Bright purple
        grad.addColorStop(0.5, `rgba(34, 211, 238, ${opacity})`);        // Glowing cyan
        grad.addColorStop(0.75, `rgba(99, 102, 241, ${opacity})`);       // Deep indigo
        grad.addColorStop(1, `rgba(99, 102, 241, ${opacity * 0.2})`);    // Indigo end fade

        ctx.strokeStyle = grad;
        ctx.stroke();
      }

      animationId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <div className="glass-panel flex flex-col items-center justify-center p-6 text-center relative overflow-hidden">
      <div className="absolute inset-0 bg-brand-dark/40" />
      <div className="absolute inset-0 bg-glowGradient opacity-10 pointer-events-none" />
      
      <div className="w-full h-36 relative z-10 flex items-center justify-center">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
      
      <div className="relative z-10 mt-2">
        <div className="text-base font-extrabold tracking-wide text-white animate-pulse">
          {message}
        </div>
        {submessage && (
          <div className="text-xs text-white/45 mt-1 font-semibold">{submessage}</div>
        )}
      </div>
    </div>
  );
}
