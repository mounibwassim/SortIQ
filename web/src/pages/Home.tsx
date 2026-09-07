import { useState, useRef, useEffect, useCallback } from "react";
import { Camera as CameraIcon, AlertCircle, BarChart3, MapPin, RotateCcw, CheckCircle, RefreshCw, Film, Camera } from "lucide-react";
import api from "../api";
import { cn } from "../lib/utils";
import CameraScanner, { type Detection } from "../components/CameraScanner";
import VideoScanner from "../components/VideoScanner";
import { useSettings } from "../context/SettingsContext";

interface RealtimePredictResponse {
  detections: Detection[];
  summary: string;
  scene_state: string;
  saved?: boolean;
  saved_id?: string;
}

const Home = () => {
  const { colors, binLabels } = useSettings();
  const useSettingsRef = useRef({ colors, binLabels });

  useEffect(() => {
    useSettingsRef.current = { colors, binLabels };
  }, [colors, binLabels]);

  // Mode Selection State
  const [scannerMode, setScannerMode] = useState<'camera' | 'video'>('camera');
  
  // API & Connection State
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [isCheckingBackend, setIsCheckingBackend] = useState(true);
  const [flash, setFlash] = useState(false);
  
  // Bounding boxes and Best Result
  const [detections, setDetections] = useState<Detection[]>([]);
  const [bestResult, setBestResult] = useState<Detection | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  
  // Freeze State
  const [isFrozen, setIsFrozen] = useState(false);
  
  // Refs
  const isRequestingRef = useRef(false);

  // ── HEALTH CHECK FUNCTION ──────────────────────────────────────────────
  const checkBackendHealth = useCallback(async () => {
    setIsCheckingBackend(true);
    try {
      const response = await api.get('/health', { timeout: 4000 });
      if (response.data && (response.data.status === 'ok' || response.data.status === 'degraded')) {
        setConnected(true);
        setError(null);
      } else {
        setConnected(true);
      }
    } catch (e) {
      try {
        // Fallback root check
        const rootResponse = await api.get('/', { timeout: 3000 });
        if (rootResponse.data && rootResponse.data.status === 'online') {
          setConnected(true);
          setError(null);
        } else {
          setConnected(false);
        }
      } catch (rootErr) {
        setConnected(false);
      }
    } finally {
      setIsCheckingBackend(false);
    }
  }, []);

  // Check health on mount and set auto-check interval if offline
  useEffect(() => {
    checkBackendHealth();

    const timer = setInterval(() => {
      checkBackendHealth();
    }, 12000);

    return () => clearInterval(timer);
  }, [checkBackendHealth]);

  // Unified confidence rules (10% threshold)
  const CLASS_RULES: Record<string, { frames: number; conf: number }> = {
    glass: { frames: 3, conf: 0.10 },
    metal: { frames: 3, conf: 0.10 },
    paper: { frames: 3, conf: 0.10 },
    plastic: { frames: 3, conf: 0.10 },
    default: { frames: 3, conf: 0.10 },
  };

  const trackedRef = useRef<Array<{ 
    id: string; 
    label: string; 
    bbox: number[]; 
    count: number; 
    lastSeen: number;
    votes: Record<string, number>;
  }>>([]);
  const frameRef = useRef<number>(0);
  const idCounterRef = useRef<number>(0);

  // Helper to extract the best waste detection
  const extractBestResult = (dets: Detection[]) => {
    const wastes = dets.filter(d => d.is_waste);
    if (wastes.length > 0) {
      return wastes.reduce((prev, current) => (prev.confidence > current.confidence) ? prev : current);
    }
    const withMessages = dets.filter(d => d.message);
    if (withMessages.length > 0) {
      return withMessages[0];
    }
    return null;
  };

  // Brightness check helper
  const computeBrightness = (imgBase64: string): Promise<number> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const cvs = document.createElement('canvas');
        const size = 64;
        cvs.width = size;
        cvs.height = size;
        const ctx = cvs.getContext('2d');
        if (!ctx) return resolve(255);
        ctx.drawImage(img, 0, 0, size, size);
        const data = ctx.getImageData(0, 0, size, size).data;
        let sum = 0;
        for (let i = 0; i < data.length; i += 4) {
          sum += (0.2126 * data[i] + 0.7152 * data[i+1] + 0.0722 * data[i+2]);
        }
        resolve(sum / (data.length / 4));
      };
      img.onerror = () => resolve(255);
      img.src = imgBase64;
    });
  };

  // Real-time frame handler
  const handleFrame = async (base64: string, materialHint?: string) => {
    if (isFrozen || isRequestingRef.current) return;

    isRequestingRef.current = true;
    try {
      const brightness = await computeBrightness(base64);
      if (brightness < 30) {
        setError("⚠️ Lighting too dark. Please move to a brighter area.");
        isRequestingRef.current = false;
        return;
      }

      const headers: Record<string, string> = {
        "X-Color-Glass":   colors.Glass   || "#22c55e",
        "X-Color-Plastic": colors.Plastic || "#3b82f6",
        "X-Color-Metal":   colors.Metal   || "#eab308",
        "X-Color-Paper":   colors.Paper   || "#f97316",
      };

      if (materialHint) {
        headers["X-Material-Hint"] = materialHint;
      }

      const response = await api.post<RealtimePredictResponse>("/predict-realtime", 
        { frame_base64: base64 },
        {
          headers,
          timeout: 15000
        }
      );
      const data = response.data;
      setConnected(true);
      setError(null);

      if (data.scene_state === "skipped") {
        isRequestingRef.current = false;
        return;
      }

      setDetections(data.detections);

      // Tracking
      const nonHuman = data.detections
        .map((d: Detection) => ({ ...d, raw: (d.raw_label || d.label || "").toLowerCase().trim() }))
        .filter((d: Detection) => (d as any).interaction_type !== 'human' && d.confidence >= 0.10)
        .sort((a: Detection, b: Detection) => b.confidence - a.confidence);

      frameRef.current += 1;

      for (const det of nonHuman) {
        const bbox = det.box as number[];
        const label = det.raw;
        
        let bestMatch = null;
        for (let i = 0; i < trackedRef.current.length; i++) {
          const t = trackedRef.current[i];
          if (t.label === label) {
             const xA = Math.max(t.bbox[0], bbox[0]);
             const yA = Math.max(t.bbox[1], bbox[1]);
             const xB = Math.min(t.bbox[2], bbox[2]);
             const yB = Math.min(t.bbox[3], bbox[3]);
             const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
             const unionArea = (t.bbox[2]-t.bbox[0])*(t.bbox[3]-t.bbox[1]) + (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) - interArea;
             const iou = interArea / unionArea;
             if (iou >= 0.5) bestMatch = i;
          }
        }

        if (bestMatch !== null) {
          const t = trackedRef.current[bestMatch];
          t.bbox = bbox;
          t.count += 1;
          t.lastSeen = frameRef.current;
        } else {
          idCounterRef.current += 1;
          trackedRef.current.push({ 
            id: `${Date.now()}-${idCounterRef.current}`, 
            label, 
            bbox, 
            count: 1, 
            lastSeen: frameRef.current,
            votes: { [label]: 1 }
          });
        }
      }
      trackedRef.current = trackedRef.current.filter(t => (frameRef.current - t.lastSeen) <= 3);

      const best = extractBestResult(data.detections);
      if (best) {
        const rawLabel = (best.raw_label || best.label || "").toLowerCase().trim();
        const rule = CLASS_RULES[rawLabel] || CLASS_RULES.default;
        if (best.confidence < rule.conf) setBestResult(null);
        else setBestResult(best);
      } else {
        setBestResult(null);
      }

    } catch (err) {
      console.error("Frame prediction error:", err);
      // Don't flip connected state immediately on single dropped frame
    } finally {
      isRequestingRef.current = false;
    }
  };

  const handleCapture = async (base64: string) => {
    if (isCapturing) return;
    setIsCapturing(true);
    setFlash(true);
    setTimeout(() => setFlash(false), 300);

    try {
      const base64Data = base64.includes(",") ? base64.split(",")[1] : base64;
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "image/jpeg" });
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });

      if (file.size < 1000) {
        setError("Capture failed — invalid image buffer.");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      const { colors } = useSettingsRef.current;

      const response = await api.post(
        "/predict-upload",
        formData,
        {
          headers: {
            "X-Color-Glass":   colors?.Glass   || "#22c55e",
            "X-Color-Plastic": colors?.Plastic || "#3b82f6",
            "X-Color-Metal":   colors?.Metal   || "#eab308",
            "X-Color-Paper":   colors?.Paper   || "#f97316",
          },
          timeout: 20000,
        }
      );

      const data = response.data;
      setConnected(true);
      setError(null);

      if (data.saved) {
        window.dispatchEvent(new CustomEvent('sortiq:scan_saved'));
      }

      setDetections(data.detections || []);
      const waste = data.detections?.find((d: Detection) => d.is_waste);

      if (waste) {
        setBestResult(waste);
      } else {
        const first = data.detections?.[0];
        if (first) setBestResult(first);
      }

    } catch (err: any) {
      console.error("Capture error:", err);
      const msg = err?.response?.data?.detail || err?.message || "Capture failed. Check backend is running.";
      setError(`Capture failed: ${msg}`);
    } finally {
      setIsCapturing(false);
    }
  };

  const resetScanner = () => {
    setIsFrozen(false);
    setDetections([]);
    setBestResult(null);
    setError(null);
  };

  return (
    <div className="max-w-4xl mx-auto pb-12 relative overflow-hidden">
      {/* 📸 FLASH EFFECT OVERLAY */}
      {flash && (
        <div className="fixed inset-0 bg-white z-[100] animate-in fade-out duration-300 pointer-events-none" />
      )}

      {/* HEADER WITH STATUS BADGE & MODE TABS */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-black text-slate-800 tracking-tight">Identify Waste</h1>
          <p className="text-xs font-semibold text-slate-500 mt-1">
            Real-time AI waste sorting for paper, plastic, metal & glass
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Connection Status Badge */}
          {connected ? (
            <div className="flex items-center gap-2 text-xs font-black text-emerald-700 bg-emerald-50 border border-emerald-200 px-3.5 py-1.5 rounded-full shadow-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span>Backend Online</span>
            </div>
          ) : (
            <button
              onClick={checkBackendHealth}
              disabled={isCheckingBackend}
              className="flex items-center gap-2 text-xs font-black text-amber-700 bg-amber-50 border border-amber-200 hover:bg-amber-100 px-3.5 py-1.5 rounded-full transition-all shadow-sm active:scale-95"
            >
              <RefreshCw className={cn("w-3.5 h-3.5 text-amber-600", isCheckingBackend && "animate-spin")} />
              <span>{isCheckingBackend ? 'Connecting...' : 'Backend Offline (Retry)'}</span>
            </button>
          )}

          <div className="flex items-center gap-2 text-xs font-bold text-indigo-600 bg-indigo-50 px-3.5 py-1.5 rounded-full uppercase tracking-tighter border border-indigo-100 shadow-sm">
             <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
             SortIQ Analyst
          </div>
        </div>
      </div>

      {/* MODE SELECTION TABS */}
      <div className="flex items-center gap-2 mb-6 bg-slate-200/70 p-1.5 rounded-2xl w-fit border border-slate-300/50">
        <button
          onClick={() => setScannerMode('camera')}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-black transition-all",
            scannerMode === 'camera'
              ? "bg-white text-slate-900 shadow-sm shadow-slate-200"
              : "text-slate-600 hover:text-slate-900"
          )}
        >
          <Camera className="w-4 h-4 text-indigo-500" />
          <span>Live Camera</span>
        </button>

        <button
          onClick={() => setScannerMode('video')}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-black transition-all",
            scannerMode === 'video'
              ? "bg-white text-slate-900 shadow-sm shadow-slate-200"
              : "text-slate-600 hover:text-slate-900"
          )}
        >
          <Film className="w-4 h-4 text-indigo-500" />
          <span>Video & Material Testing</span>
          <span className="bg-indigo-100 text-indigo-700 text-[10px] px-2 py-0.5 rounded-full uppercase font-extrabold">
            Paper, Plastic, Metal, Glass
          </span>
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* LEFT COLUMN: Camera Scanner OR Video Scanner */}
        <div className="relative group flex flex-col gap-4">
          {scannerMode === 'camera' ? (
            <CameraScanner 
              onFrame={handleFrame}
              onCapture={handleCapture}
              detections={detections}
              frozen={isFrozen}
              isCapturing={isCapturing}
              trackedObjects={trackedRef.current.map(t => ({ 
                ...t, 
                stable: t.count >= (CLASS_RULES[t.label] || CLASS_RULES.default).frames 
              }))}
            />
          ) : (
            <VideoScanner
              onFrame={handleFrame}
              onCapture={handleCapture}
              detections={detections}
              isCapturing={isCapturing}
            />
          )}

          {/* Robot Chat Bubble */}
          <div className={cn(
            "bg-white/95 backdrop-blur-md rounded-2xl p-4 shadow-xl border border-indigo-100 flex gap-4 items-start transition-all duration-500 transform",
            bestResult?.message ? "translate-y-0 opacity-100" : "translate-y-2 opacity-90"
          )}>
            <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center shrink-0 text-2xl shadow-inner relative overflow-hidden border border-indigo-200">
              <div className="absolute inset-0 bg-indigo-500/10 animate-pulse rounded-full" />
              🤖
            </div>
            <div className="flex-1 min-w-0 pt-0.5">
                <h4 className="text-[10px] font-black text-indigo-900 mb-1 uppercase tracking-widest opacity-60">SortIQ Robot</h4>
                <p className="text-sm text-slate-700 leading-relaxed font-medium capitalize">
                    {bestResult?.message || "Select a video test feed or point live camera at waste items..."}
                </p>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Result Card */}
        <div className="flex flex-col h-full">
          <h2 className="text-xl font-bold mb-4 text-slate-800 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-500" />
            Analysis Result
          </h2>
          
          <div className="flex-1 bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden relative min-h-[400px] flex flex-col justify-between">
            {!bestResult && !error && (
              <div className="flex-1 flex flex-col items-center justify-center text-slate-400 p-8 m-4 rounded-xl border-dashed border-2 border-slate-100 bg-slate-50/30">
                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4 opacity-50">
                   <CameraIcon className="w-8 h-8 text-slate-400" />
                </div>
                <p className="text-center font-bold text-slate-500/80">
                  {scannerMode === 'camera' ? 'Pointing camera at waste...' : 'Analyzing video stream...'}
                </p>
                <p className="text-[10px] uppercase tracking-widest mt-2 opacity-50">High-Precision AI Classifier</p>
              </div>
            )}

            {!connected && (
              <div className="p-6">
                <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5 text-center">
                  <div className="w-10 h-10 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center mx-auto mb-3 font-bold text-lg">
                    🖥️
                  </div>
                  <h3 className="text-amber-900 font-bold text-base mb-1">
                    Backend is offline
                  </h3>
                  <p className="text-amber-800 text-xs leading-relaxed max-w-xs mx-auto mb-4">
                    This is a portfolio demo. To run locally: clone the repo and start the backend uvicorn server.
                  </p>
                  
                  <div className="flex flex-col gap-2">
                    <button
                      onClick={checkBackendHealth}
                      disabled={isCheckingBackend}
                      className="w-full py-2.5 bg-amber-600 hover:bg-amber-700 text-white rounded-xl text-xs font-black uppercase tracking-wider transition-all flex items-center justify-center gap-2 shadow-sm"
                    >
                      <RefreshCw className={cn("w-4 h-4", isCheckingBackend && "animate-spin")} />
                      <span>{isCheckingBackend ? 'Checking Server...' : 'Re-check Backend Connection'}</span>
                    </button>

                    <a 
                      href="https://github.com/mounibwassim/SortIQ"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-amber-700 hover:text-amber-900 underline font-semibold py-1 block"
                    >
                      📂 View on GitHub →
                    </a>
                  </div>
                </div>
              </div>
            )}

            {error && connected && !bestResult && (
              <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
                <div className="bg-red-50/50 rounded-2xl p-6 border border-red-100 flex flex-col items-center text-center max-w-sm">
                  <AlertCircle className="w-10 h-10 text-red-400 mb-3" />
                  <h3 className="font-bold text-red-900 mb-1">Notice</h3>
                  <p className="text-xs text-red-700/80 leading-relaxed">{error}</p>
                </div>
              </div>
            )}

            {bestResult && (
              <div className="flex flex-col h-full animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="p-8 border-b border-slate-50 bg-white">
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div 
                          className="w-4 h-4 rounded-full shadow-lg"
                          style={{ 
                            backgroundColor: bestResult.is_waste && colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] ? colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] : (bestResult.color_hex || bestResult.box_color_hex),
                            boxShadow: `0 0 15px ${bestResult.is_waste && colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] ? colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] : (bestResult.color_hex || bestResult.box_color_hex)}`
                          }}
                        />
                        <h3 className="text-4xl font-black text-slate-900 capitalize tracking-tight">
                          {bestResult.label}
                        </h3>
                      </div>
                      
                      {bestResult.location && (
                         <div className="flex items-center gap-1.5 text-[10px] font-black text-slate-500 bg-slate-100 px-3 py-1 rounded-full uppercase tracking-tighter">
                           <MapPin className="w-3 h-3" />
                           {bestResult.location}
                         </div>
                      )}
                    </div>
                    
                    <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs font-black text-slate-500 uppercase tracking-widest">
                            <span>Confidence</span>
                            <span>{(bestResult.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bg-slate-100 h-3 rounded-full overflow-hidden shadow-inner">
                          <div 
                            className="h-full transition-all duration-700 ease-out"
                            style={{ 
                              width: `${Math.min(bestResult.confidence * 100, 100)}%`,
                              backgroundColor: bestResult.is_waste && colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] ? colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] : (bestResult.color_hex || bestResult.box_color_hex)
                            }}
                          />
                        </div>
                    </div>
                    
                    <div className="text-xl font-bold text-slate-900 mt-2 bg-slate-50 p-4 rounded-2xl border border-slate-100">
                      {bestResult.is_waste ? (
                        <div className="flex items-center gap-3">
                          <span className="text-slate-700">Place in</span>
                          <div style={{
                            width: 28,
                            height: 28,
                            borderRadius: "50%",
                            backgroundColor: colors[
                              bestResult.label.charAt(0).toUpperCase()
                              + bestResult.label.slice(1)
                            ] || bestResult.color_hex || "#22c55e",
                            boxShadow: `0 0 12px ${
                              colors[
                                bestResult.label.charAt(0).toUpperCase()
                                + bestResult.label.slice(1)
                              ] || bestResult.color_hex || "#22c55e"
                            }`,
                            flexShrink: 0,
                          }} />
                          <span className="text-slate-700">recycling bin</span>
                        </div>
                      ) : (
                        <span className="text-slate-500">{bestResult.message || "Non-waste detected"}</span>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="p-8 flex-1 bg-slate-50/30">
                  {/* Multi-Item Detection Summary for Conveyor Machine Videos */}
                  {detections.length > 1 && (
                    <div className="mb-6 bg-slate-100/80 p-4 rounded-2xl border border-slate-200">
                      <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 flex items-center justify-between">
                        <span>Items On Screen ({detections.length})</span>
                        <span className="text-indigo-600 font-extrabold">Active Machine Vision</span>
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {detections.map((d, idx) => (
                          <div
                            key={idx}
                            className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-xl border border-slate-200 text-xs font-bold text-slate-800 shadow-sm"
                          >
                            <div
                              className="w-2.5 h-2.5 rounded-full"
                              style={{ backgroundColor: d.color_hex || d.box_color_hex || '#22c55e' }}
                            />
                            <span className="capitalize">{d.label}</span>
                            <span className="text-slate-400 text-[10px]">({Math.round(d.confidence * 100)}%)</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {bestResult.tip && (
                    <>
                      <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4">Robot Tip</h4>
                      <div className="flex items-start gap-4 bg-white p-5 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden group hover:shadow-md transition-all">
                        <div className="absolute left-0 top-0 bottom-0 w-1.5" style={{ backgroundColor: bestResult.is_waste && colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] ? colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] : (bestResult.color_hex || bestResult.box_color_hex) }} />
                        <CheckCircle className="w-6 h-6 mt-0.5 shrink-0" style={{ color: bestResult.is_waste && colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] ? colors[bestResult.label.charAt(0).toUpperCase() + bestResult.label.slice(1)] : (bestResult.color_hex || bestResult.box_color_hex) }} />
                        <p className="text-slate-600 leading-relaxed font-semibold">{bestResult.tip}</p>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
            
            {(bestResult || error) && (
              <div className="p-6 bg-white border-t border-slate-50">
                <button 
                  onClick={resetScanner}
                  className="w-full h-12 flex items-center justify-center gap-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl font-black uppercase tracking-widest text-xs transition-all active:scale-95 shadow-lg shadow-slate-200"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset Scanner
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
