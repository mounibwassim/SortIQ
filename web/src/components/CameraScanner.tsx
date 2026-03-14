import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, AlertCircle } from 'lucide-react';
import { useSettings } from '../context/SettingsContext';

export interface Detection {
  label: string;
  confidence: number;
  box: number[];
  is_waste: boolean;
  location: string;
  message?: string;
  bin_color?: string;
  color_hex?: string;
  tip?: string;
  box_color: string;
  box_color_hex: string;
  raw_label?: string;
  interaction_type?: string;
}

interface CameraScannerProps {
  onFrame: (base64: string) => void;
  onCapture: (base64: string) => void;
  detections: Detection[];
  frozen: boolean;
  isCapturing?: boolean;
  trackedObjects?: Array<{
    id: string;
    label: string;
    bbox: number[];
    count: number;
    stable?: boolean;
  }>;
}

const CameraScanner: React.FC<CameraScannerProps> = ({
  onFrame,
  onCapture,
  detections,
  frozen,
  isCapturing,
  trackedObjects,
}) => {
  const { colors } = useSettings();
  const webcamRef   = useRef<Webcam>(null);
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive]               = useState(false);
  const [permissionDenied, setPermissionDenied] = useState(false);

  // ── LIVE PREVIEW LOOP ──────────────────────────────────────────────
  // Sends frames to /predict-realtime every 800ms for circle drawing.
  // NEVER calls onCapture(). NEVER saves anything.
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isActive && !frozen) {
      interval = setInterval(() => {
        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();
          if (imageSrc) {
            onFrame(imageSrc); // preview only — no save
          }
        }
      }, 800);
    }
    return () => clearInterval(interval);
  }, [isActive, frozen, onFrame]);
  // ── END LIVE PREVIEW LOOP ──────────────────────────────────────────

  // ── DRAW CIRCLES ON CANVAS ────────────────────────────────────────
  useEffect(() => {
    const video  = webcamRef.current?.video;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0 || !isActive) return;

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.box;
      const cx     = x1 + (x2 - x1) / 2;
      const cy     = y1 + (y2 - y1) / 2;
      const radius = Math.max(x2 - x1, y2 - y1) / 2;

      const formattedLabel = det.label
        ? det.label.charAt(0).toUpperCase() + det.label.slice(1)
        : '';
      const color =
        (det.is_waste && colors[formattedLabel as keyof typeof colors])
          ? colors[formattedLabel as keyof typeof colors]
          : det.color_hex || det.box_color_hex || '#22c55e';

      // Waste circle — dotted glow
      if (det.is_waste) {
        ctx.save();
        ctx.shadowBlur  = 15;
        ctx.shadowColor = color;
        ctx.strokeStyle = color;
        ctx.setLineDash([8, 4]);
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      } else {
        // Non-waste circle — solid
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Pulse fill for stable waste items (visual only — no save triggered)
      const match = (trackedObjects || []).find(
        t => t.label.toLowerCase() === (det.raw_label || '').toLowerCase()
      );
      if (match?.stable && det.is_waste) {
        const pulse = 1 + 0.1 * Math.sin(Date.now() / 200);
        ctx.save();
        ctx.globalAlpha = 0.25;
        ctx.fillStyle   = color;
        ctx.beginPath();
        ctx.arc(cx, cy, radius * pulse, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // Label tag above circle
      ctx.setLineDash([]);
      const labelText = `${det.label} ${Math.round(det.confidence * 100)}%`;
      ctx.font = 'bold 13px Inter, sans-serif';
      const metrics = ctx.measureText(labelText);
      const tagW = metrics.width + 20;
      const tagH = 24;
      const tagX = cx - tagW / 2;
      const tagY = cy - radius - 34;

      ctx.fillStyle = color;
      ctx.beginPath();
      (ctx as any).roundRect?.(tagX, tagY, tagW, tagH, 10) ||
        ctx.rect(tagX, tagY, tagW, tagH);
      ctx.fill();

      ctx.fillStyle   = '#ffffff';
      ctx.textAlign   = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, cx, tagY + tagH / 2);
      ctx.textBaseline = 'alphabetic';
    });
  }, [detections, isActive, frozen, trackedObjects, colors]);
  // ── END DRAW CIRCLES ──────────────────────────────────────────────

  // ── SHUTTER BUTTON HANDLER ────────────────────────────────────────
  // This is the ONLY place onCapture() is called.
  // Captures from the WEBCAM (real photo), not the canvas (circles only).
  const handleShutterClick = () => {
    if (isCapturing) return;
    if (!webcamRef.current) return;

    // Get full-quality screenshot from the actual webcam feed
    const imageSrc = webcamRef.current.getScreenshot({
      width:  1280,
      height: 720,
    });

    if (!imageSrc) {
      console.warn('[SHUTTER] Could not capture screenshot');
      return;
    }

    console.log('SHUTTER CLICKED BY USER'); // verification log
    onCapture(imageSrc); // ← THE ONLY PLACE onCapture IS CALLED
  };
  // ── END SHUTTER HANDLER ───────────────────────────────────────────

  const handleUserMediaError = (err: string | DOMException) => {
    console.error('Camera Error:', err);
    setPermissionDenied(true);
    setIsActive(false);
  };

  // ── RENDER ────────────────────────────────────────────────────────
  return (
    <div className="bg-black rounded-3xl shadow-2xl overflow-hidden relative border-4 border-slate-900 group aspect-video">
      {permissionDenied ? (
        <div className="absolute inset-0 bg-red-50 flex flex-col items-center justify-center p-6 text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mb-3" />
          <h4 className="font-bold text-red-700 mb-1">Camera Access Needed</h4>
          <p className="text-red-600 text-[10px] uppercase tracking-widest mb-4">
            Please enable camera to start Robot Team
          </p>
          <button
            onClick={() => { setPermissionDenied(false); setIsActive(true); }}
            className="px-6 py-2 bg-red-600 text-white rounded-full text-xs font-black uppercase tracking-widest"
          >
            Try Again
          </button>
        </div>

      ) : isActive ? (
        <>
          {/* Live camera feed */}
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            screenshotQuality={0.92}
            className={`w-full h-full object-cover transition-transform duration-1000 ${
              frozen ? 'scale-105 saturate-150' : 'scale-100'
            }`}
            videoConstraints={{ facingMode: 'environment' }}
            onUserMediaError={handleUserMediaError}
          />

          {/* Circle overlay canvas — visual only, never used for capture */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
          />

          {/* Stop button */}
          <button
            onClick={() => setIsActive(false)}
            className="absolute top-4 right-4 bg-black/50 hover:bg-black/80 text-white/70 p-2 rounded-full backdrop-blur-md transition-all opacity-0 group-hover:opacity-100"
          >
            <StopCircle />
          </button>

          {/* Tap to save hint */}
          <div className="absolute top-4 left-4 bg-black/40 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/10 flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
            <span className="text-[10px] font-black text-white uppercase tracking-widest">
              Tap shutter to save
            </span>
          </div>

          {/* ── SHUTTER BUTTON ───────────────────────────────────── */}
          {/* onCapture() is called ONLY here — never anywhere else   */}
          <div className="absolute bottom-6 left-0 right-0 flex justify-center">
            <button
              onClick={handleShutterClick}
              disabled={isCapturing}
              style={{
                width:        72,
                height:       72,
                borderRadius: '50%',
                border:       '4px solid white',
                background:   isCapturing ? '#22c55e' : 'white',
                cursor:       isCapturing ? 'not-allowed' : 'pointer',
                boxShadow:    '0 4px 20px rgba(0,0,0,0.4)',
                display:      'flex',
                alignItems:   'center',
                justifyContent: 'center',
                transition:   'all 0.15s',
              }}
              className="active:scale-90"
            >
              <div
                style={{
                  width:        54,
                  height:       54,
                  borderRadius: '50%',
                  background:   isCapturing ? '#16a34a' : 'white',
                  border:       '2px solid rgba(0,0,0,0.1)',
                }}
              />
            </button>
          </div>
          {/* ── END SHUTTER BUTTON ──────────────────────────────── */}
        </>

      ) : (
        /* Start screen */
        <div
          className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 cursor-pointer hover:bg-slate-900/50 transition-all"
          onClick={() => setIsActive(true)}
        >
          <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
            <Camera className="w-8 h-8 text-white" />
          </div>
          <p className="font-black uppercase tracking-[0.2em] text-xs text-white">
            Start Robot Scanner
          </p>
        </div>
      )}
    </div>
  );
};

const StopCircle = () => (
  <svg
    width="24" height="24" viewBox="0 0 24 24"
    fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
  >
    <circle cx="12" cy="12" r="10" />
    <rect x="9" y="9" width="6" height="6" />
  </svg>
);

export default CameraScanner;
