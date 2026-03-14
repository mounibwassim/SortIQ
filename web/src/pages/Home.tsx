import { useState, useRef, useEffect } from "react";
import { Camera as CameraIcon, AlertCircle, BarChart3, MapPin, RotateCcw, CheckCircle } from "lucide-react";
import api from "../api";
import { cn } from "../lib/utils";
import CameraScanner, { type Detection } from "../components/CameraScanner";
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
  
  // API State
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState(false);
  
  // Bounding boxes and Best Result
  const [detections, setDetections] = useState<Detection[]>([]);
  const [bestResult, setBestResult] = useState<Detection | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  
  // Freeze State
  const [isFrozen, setIsFrozen] = useState(false);
  
  // Refs
  const isRequestingRef = useRef(false);

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

  // 1. Brightness check helper
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

  // 2. Real-time frame handler
  const handleFrame = async (base64: string) => {
    console.log("HANDLE FRAME CALLED");
    if (isFrozen || isRequestingRef.current) return;

    isRequestingRef.current = true;
    try {
      const brightness = await computeBrightness(base64);
      if (brightness < 40) {
        setError("⚠️ Lighting too dark. Please move to a brighter area.");
        isRequestingRef.current = false;
        return;
      }

      const response = await api.post<RealtimePredictResponse>("/predict-realtime", 
        { frame_base64: base64 },
        {
          headers: {
            "X-Color-Glass":   colors.Glass   || "#22c55e",
            "X-Color-Plastic": colors.Plastic || "#3b82f6",
            "X-Color-Metal":   colors.Metal   || "#eab308",
            "X-Color-Paper":   colors.Paper   || "#f97316",
          }
        }
      );
      const data = response.data;

      if (data.scene_state === "skipped") {
        isRequestingRef.current = false;
        return;
      }

      setDetections(data.detections);

      // Universal Tracking
      const nonHuman = data.detections
        .map(d => ({ ...d, raw: (d.raw_label || d.label || "").toLowerCase().trim() }))
        .filter(d => (d as any).interaction_type !== 'human' && d.confidence >= 0.10)
        .sort((a, b) => b.confidence - a.confidence);

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
      console.error(err);
      setError("⚠️ Connection lost. Retrying backend...");
    } finally {
      isRequestingRef.current = false;
    }
  };

  const handleCapture = async (base64: string) => {
    console.log("HANDLE CAPTURE CALLED");
    if (isCapturing) return;
    setIsCapturing(true);
    setFlash(true);
    setTimeout(() => setFlash(false), 300);

    try {
      // Convert base64 to blob correctly
      const base64Data = base64.includes(",")
        ? base64.split(",")[1]
        : base64;

      const byteCharacters = atob(base64Data);
      const byteNumbers    = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob      = new Blob([byteArray], { type: "image/jpeg" });
      const file      = new File([blob], "capture.jpg", {
        type: "image/jpeg"
      });

      // Verify file was created correctly
      console.log("Capture file size:", file.size, "bytes");
      if (file.size < 1000) {
        console.error("Capture file too small — bad screenshot");
        setError("Capture failed — try again.");
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
            "Content-Type":    "multipart/form-data",
            "X-Color-Glass":   colors?.Glass   || "#22c55e",
            "X-Color-Plastic": colors?.Plastic || "#3b82f6",
            "X-Color-Metal":   colors?.Metal   || "#eab308",
            "X-Color-Paper":   colors?.Paper   || "#f97316",
          },
          timeout: 20000,
        }
      );

      const data = response.data;
      console.log("Upload response:", data);

      if (data.saved) {
        window.dispatchEvent(new CustomEvent('sortiq:scan_saved'));
      }

      setDetections(data.detections || []);
      const waste = data.detections?.find(
        (d: Detection) => d.is_waste
      );

      if (waste) {
        setBestResult(waste);
        console.log("Waste saved:", waste.label);
      } else {
        const first = data.detections?.[0];
        if (first) setBestResult(first);
      }

    } catch (err: any) {
      console.error("Capture error:", err?.response?.data || err);
      setError("Capture failed. Check backend is running.");
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

      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Identify Waste</h1>
        <div className="flex items-center gap-2 text-xs font-bold text-indigo-600 bg-indigo-50 px-3 py-1.5 rounded-full uppercase tracking-tighter border border-indigo-100 shadow-sm">
           <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
           Live Robot Analyst
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* LEFT COLUMN: Camera Source */}
        <div className="relative group">
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

          <div className="absolute top-4 left-4 bg-black/40 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/10 flex items-center gap-2">
             <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
             <span className="text-[10px] font-black text-white uppercase tracking-widest">Tap shutter to save</span>
          </div>

          {/* Robot Chat Bubble */}
          <div className={cn(
            "mt-4 bg-white/95 backdrop-blur-md rounded-2xl p-4 shadow-xl border border-indigo-100 flex gap-4 items-start transition-all duration-500 transform",
            bestResult?.message ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0 pointer-events-none"
          )}>
            <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center shrink-0 text-2xl shadow-inner relative overflow-hidden border border-indigo-200">
              <div className="absolute inset-0 bg-indigo-500/10 animate-pulse rounded-full" />
              🤖
            </div>
            <div className="flex-1 min-w-0 pt-0.5">
                <h4 className="text-[10px] font-black text-indigo-900 mb-1 uppercase tracking-widest opacity-60">SortIQ Robot</h4>
                <p className="text-sm text-slate-700 leading-relaxed font-medium capitalize">
                    {bestResult?.message || "Hold the object steady..."}
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
                <p className="text-center font-bold text-slate-500/80">Pointing camera at waste...</p>
                <p className="text-[10px] uppercase tracking-widest mt-2 opacity-50">Detection sensitive to 10%</p>
              </div>
            )}

            {error && !bestResult && (
              <div className="flex-1 flex items-center justify-center p-6">
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
