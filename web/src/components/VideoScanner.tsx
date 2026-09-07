import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Play, Pause, Upload, RotateCcw, Sparkles, CheckCircle2, Film, FileVideo } from 'lucide-react';
import { MATERIAL_SAMPLES, renderSampleFrame, type MaterialSample } from '../lib/sampleVideos';
import type { Detection } from './CameraScanner';
import { useSettings } from '../context/SettingsContext';

interface VideoScannerProps {
  onFrame: (base64: string, materialHint?: string) => void;
  onCapture: (base64: string) => void;
  detections: Detection[];
  isCapturing?: boolean;
}

const VideoScanner: React.FC<VideoScannerProps> = ({
  onFrame,
  onCapture,
  detections,
  isCapturing = false,
}) => {
  const { colors } = useSettings();
  const [selectedSample, setSelectedSample] = useState<MaterialSample>(MATERIAL_SAMPLES[0]);
  const [customVideoUrl, setCustomVideoUrl] = useState<string | null>(null);
  const [customFileName, setCustomFileName] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [useUploadMode, setUseUploadMode] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const sampleCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);

  // ── 1. CUSTOM FILE UPLOAD HANDLER ──────────────────────────────────────
  const handleFileSelect = (file: File) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setCustomVideoUrl(url);
    setCustomFileName(file.name);
    setUseUploadMode(true);
    setIsPlaying(true);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  // ── 2. SAMPLE ANIMATION LOOP & FRAME EXTRACTION ─────────────────────────
  const processFrameAndSend = useCallback(() => {
    let sourceCanvas: HTMLCanvasElement | null = null;

    if (useUploadMode && videoRef.current) {
      const v = videoRef.current;
      if (v.videoWidth > 0 && v.videoHeight > 0) {
        const offscreen = document.createElement('canvas');
        offscreen.width = v.videoWidth;
        offscreen.height = v.videoHeight;
        const ctx = offscreen.getContext('2d');
        if (ctx) {
          ctx.drawImage(v, 0, 0, offscreen.width, offscreen.height);
          sourceCanvas = offscreen;
        }
      }
    } else if (sampleCanvasRef.current) {
      sourceCanvas = sampleCanvasRef.current;
    }

    if (sourceCanvas) {
      const base64 = sourceCanvas.toDataURL('image/jpeg', 0.85);
      const hint = useUploadMode ? undefined : selectedSample.material;
      onFrame(base64, hint);
    }
  }, [useUploadMode, selectedSample, onFrame]);

  // Animation Loop for Procedural Sample Video
  useEffect(() => {
    if (useUploadMode || !isPlaying) return;

    let startTime = performance.now();
    const animate = (now: number) => {
      const elapsed = now - startTime;
      const cvs = sampleCanvasRef.current;
      if (cvs) {
        const ctx = cvs.getContext('2d');
        if (ctx) {
          renderSampleFrame(ctx, cvs.width, cvs.height, selectedSample.id, elapsed);
        }
      }

      // Send frame every 500ms to backend for fast responsive scanning
      if (now - lastFrameTimeRef.current > 500) {
        lastFrameTimeRef.current = now;
        processFrameAndSend();
      }

      animFrameRef.current = requestAnimationFrame(animate);
    };

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [selectedSample, isPlaying, useUploadMode, processFrameAndSend]);

  // Video Time Update Interval for Uploaded Custom Videos
  useEffect(() => {
    if (!useUploadMode || !isPlaying) return;

    const interval = setInterval(() => {
      processFrameAndSend();
    }, 500);

    return () => clearInterval(interval);
  }, [useUploadMode, isPlaying, processFrameAndSend]);

  // ── 3. DRAW DETECTIONS OVERLAY ──────────────────────────────────────────
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, w, h);

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.box;
      const cx = x1 + (x2 - x1) / 2;
      const cy = y1 + (y2 - y1) / 2;
      const radius = Math.max(x2 - x1, y2 - y1) / 2;

      const formattedLabel = det.label
        ? det.label.charAt(0).toUpperCase() + det.label.slice(1)
        : '';
      const color =
        det.is_waste && colors[formattedLabel as keyof typeof colors]
          ? colors[formattedLabel as keyof typeof colors]
          : det.color_hex || det.box_color_hex || '#22c55e';

      // Ring overlay
      ctx.save();
      ctx.shadowBlur = 15;
      ctx.shadowColor = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3.5;
      if (det.is_waste) ctx.setLineDash([8, 4]);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Label Tag
      const labelText = `${det.label} ${Math.round(det.confidence * 100)}%`;
      ctx.font = 'bold 12px Inter, sans-serif';
      const metrics = ctx.measureText(labelText);
      const tagW = metrics.width + 20;
      const tagH = 24;
      const tagX = cx - tagW / 2;
      const tagY = cy - radius - 30;

      ctx.fillStyle = color;
      ctx.beginPath();
      (ctx as any).roundRect?.(tagX, tagY, tagW, tagH, 8) || ctx.rect(tagX, tagY, tagW, tagH);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, cx, tagY + tagH / 2);
    });
  }, [detections, colors]);

  // ── 4. SNAPSHOT CAPTURE ────────────────────────────────────────────────
  const handleSnapshotCapture = () => {
    let sourceCanvas: HTMLCanvasElement | null = null;

    if (useUploadMode && videoRef.current) {
      const v = videoRef.current;
      const offscreen = document.createElement('canvas');
      offscreen.width = v.videoWidth || 640;
      offscreen.height = v.videoHeight || 480;
      const ctx = offscreen.getContext('2d');
      if (ctx) {
        ctx.drawImage(v, 0, 0, offscreen.width, offscreen.height);
        sourceCanvas = offscreen;
      }
    } else if (sampleCanvasRef.current) {
      sourceCanvas = sampleCanvasRef.current;
    }

    if (sourceCanvas) {
      const base64 = sourceCanvas.toDataURL('image/jpeg', 0.95);
      onCapture(base64);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInputChange}
        accept="video/*,image/*"
        className="hidden"
      />

      {/* DRAG & DROP UPLOAD BANNER / SWITCHER */}
      <div 
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`p-4 rounded-2xl border-2 border-dashed transition-all flex flex-col sm:flex-row items-center justify-between gap-3 ${
          isDragging 
            ? 'bg-indigo-500/20 border-indigo-500 scale-[1.01]' 
            : useUploadMode 
            ? 'bg-slate-900 border-indigo-500/50 text-white' 
            : 'bg-white border-slate-200 text-slate-700 hover:border-indigo-400'
        }`}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center shrink-0 shadow-inner">
            <FileVideo className="w-5 h-5" />
          </div>
          <div>
            <h4 className="font-extrabold text-sm text-slate-800 dark:text-white">
              {useUploadMode ? `Active Video: ${customFileName}` : 'Upload Conveyor Machine Video'}
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Drag & drop your machine video file (.mp4, .webm, .mov) or test preset feeds
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-black text-xs uppercase tracking-wider shadow-md transition-all active:scale-95 flex items-center gap-2"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Video File</span>
          </button>
        </div>
      </div>

      {/* 🎬 MATERIAL PRESET TABS */}
      <div className="flex items-center justify-between gap-2 bg-slate-900/95 p-3 rounded-2xl border border-slate-800 shadow-lg">
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0">
          <span className="text-[10px] font-black text-slate-400 uppercase tracking-wider px-2 flex items-center gap-1">
            <Film className="w-3.5 h-3.5 text-indigo-400" />
            Preset Feeds:
          </span>
          {MATERIAL_SAMPLES.map((sample) => {
            const isSelected = !useUploadMode && selectedSample.id === sample.id;
            return (
              <button
                key={sample.id}
                onClick={() => {
                  setUseUploadMode(false);
                  setSelectedSample(sample);
                  setIsPlaying(true);
                }}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 shrink-0 ${
                  isSelected
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20 scale-105'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white'
                }`}
              >
                <span>{sample.icon}</span>
                <span>{sample.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 📹 VIDEO PLAYER CONTAINER */}
      <div className="bg-black rounded-3xl shadow-2xl overflow-hidden relative border-4 border-slate-900 aspect-video group">
        {/* Render Custom Video */}
        {useUploadMode && customVideoUrl ? (
          <video
            ref={videoRef}
            src={customVideoUrl}
            autoPlay
            loop
            muted
            playsInline
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            className="w-full h-full object-cover"
          />
        ) : (
          /* Render Animated Sample Canvas Feed */
          <canvas
            ref={sampleCanvasRef}
            width={640}
            height={360}
            className="w-full h-full object-cover"
          />
        )}

        {/* AI Detection Overlay Canvas */}
        <canvas
          ref={overlayCanvasRef}
          width={640}
          height={360}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* TOP STATUS BAR OVERLAY */}
        <div className="absolute top-4 left-4 right-4 flex justify-between items-center pointer-events-none">
          <div className="bg-black/70 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
            <span className="text-[10px] font-black text-white uppercase tracking-widest flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              {useUploadMode ? 'Custom Video Machine Stream' : `${selectedSample.material} Material Feed`}
            </span>
          </div>

          <div className="bg-black/70 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10 text-[10px] font-black text-slate-300 uppercase tracking-widest">
            {detections.length > 0 ? `Detected ${detections.length} item(s)` : 'Scanning Stream...'}
          </div>
        </div>

        {/* BOTTOM CONTROLS & SHUTTER OVERLAY */}
        <div className="absolute bottom-6 left-0 right-0 flex items-center justify-between px-6">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="bg-black/60 hover:bg-black/90 text-white p-3 rounded-2xl backdrop-blur-md transition-all border border-white/10 active:scale-95"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 text-emerald-400" />}
          </button>

          {/* Snapshot Save Button */}
          <button
            onClick={handleSnapshotCapture}
            disabled={isCapturing}
            className="px-5 py-3 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white font-black text-xs uppercase tracking-widest shadow-xl flex items-center gap-2 border border-emerald-300/30 active:scale-95 transition-all"
          >
            <CheckCircle2 className="w-4 h-4 text-emerald-100" />
            {isCapturing ? 'Saving Scan...' : 'Save Scan Frame'}
          </button>

          <button
            onClick={() => {
              setUseUploadMode(false);
              setSelectedSample(MATERIAL_SAMPLES[0]);
              setIsPlaying(true);
            }}
            className="bg-black/60 hover:bg-black/90 text-white p-3 rounded-2xl backdrop-blur-md transition-all border border-white/10 active:scale-95"
            title="Reset Stream"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoScanner;
