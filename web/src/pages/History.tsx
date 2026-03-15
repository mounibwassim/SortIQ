import { useState, useEffect, useRef } from "react";
import { Loader2, Calendar, FileImage, RefreshCw, Trash2, X, AlertTriangle, ChevronRight } from "lucide-react";
import api, { BASE_URL } from "../api";
import { useSettings } from "../context/SettingsContext";

interface ScanHistory {
  id: string;
  timestamp: string;
  predicted_class: string;
  confidence: number;
  image_thumbnail_url: string | null;
  interaction_type: string;
  robot_message: string | null;
}

// -------------------------------------------------------------
// ConfirmModal Component
// -------------------------------------------------------------
const ConfirmModal = ({ 
  isOpen, 
  onClose, 
  onConfirm, 
  title, 
  message 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  onConfirm: () => void; 
  title: string; 
  message: string;
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl max-w-sm w-full p-6 animate-in zoom-in-95 duration-200">
        <div className="flex items-center gap-3 text-red-600 mb-4">
          <div className="p-2 bg-red-50 rounded-full">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h3 className="text-xl font-bold">{title}</h3>
        </div>
        <p className="text-slate-600 mb-6 font-medium">{message}</p>
        <div className="flex gap-3 justify-end">
          <button 
            onClick={onClose}
            className="px-4 py-2 text-slate-600 font-bold hover:bg-slate-50 rounded-xl transition-colors"
          >
            Cancel
          </button>
          <button 
            onClick={() => { onConfirm(); onClose(); }}
            className="px-6 py-2 bg-red-600 hover:bg-red-700 text-white font-bold rounded-xl shadow-md transition-colors"
          >
            Yes, Delete
          </button>
        </div>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// DetailDrawer Component
// -------------------------------------------------------------
const DetailDrawer = ({ 
  scan, 
  isOpen, 
  onClose,
  binColorHex
}: { 
  scan: ScanHistory | null; 
  isOpen: boolean; 
  onClose: () => void;
  binColorHex: string;
}) => {
  if (!isOpen || !scan) return null;

  const isWaste = scan.interaction_type === 'waste';

  return (
    <>
      <div 
        className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 animate-in fade-in duration-300" 
        onClick={onClose}
      />
      <div className="fixed top-0 right-0 h-full w-full max-w-md bg-white shadow-2xl z-50 transform transition-transform duration-300 ease-in-out border-l border-slate-100 flex flex-col slide-in-from-right">
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-100 bg-slate-50/50">
          <h2 className="text-xl font-bold text-slate-800">Scan Details</h2>
          <button 
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-full transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="w-full aspect-square bg-slate-100 rounded-2xl mb-6 overflow-hidden relative shadow-inner border border-slate-200">
            {scan.image_thumbnail_url ? (
              <img 
                src={BASE_URL + scan.image_thumbnail_url} 
                alt="Scan Thumbnail" 
                className="w-full h-full object-cover" 
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-slate-400 flex-col gap-2">
                <FileImage className="w-12 h-12 opacity-50" />
                <span className="text-sm font-medium">No Image Saved</span>
              </div>
            )}
            
            {/* Tag Overlay */}
            <div className="absolute top-4 right-4 bg-white/95 backdrop-blur px-3 py-1.5 rounded-lg text-sm font-bold text-slate-700 shadow-lg border border-slate-100 flex items-center gap-2">
              <div 
                className="w-2.5 h-2.5 rounded-full" 
                style={{ backgroundColor: isWaste ? binColorHex : '#6366f1' }}
              />
              <span className="capitalize">{scan.predicted_class}</span>
            </div>
          </div>

          <div className="space-y-6">
            <div>
              <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Time Captured</p>
              <div className="flex items-center gap-2 text-slate-800 font-medium bg-slate-50 p-3 rounded-xl border border-slate-100">
                <Calendar className="w-4 h-4 text-slate-400" />
                {new Date(scan.timestamp).toLocaleString(undefined, {
                  dateStyle: "full", timeStyle: "medium"
                })}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Confidence</p>
                <div className={`text-2xl font-black ${scan.confidence > 0.70 ? 'text-green-600' : 'text-yellow-600'}`}>
                  {(scan.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Type</p>
                <div className="text-lg font-bold text-slate-700 capitalize">
                  {scan.interaction_type}
                </div>
              </div>
            </div>

            {scan.robot_message && (
              <div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Robot Analysis</p>
                <div className="bg-indigo-50/80 rounded-xl p-4 border border-indigo-100 relative before:absolute before:-top-2 before:left-6 before:border-8 before:border-transparent before:border-b-indigo-100">
                  <p className="text-indigo-900 italic font-medium leading-relaxed">
                    "{scan.robot_message.split('\n').join(' ')}"
                  </p>
                </div>
              </div>
            )}
            
            <div className="pt-4 border-t border-slate-100">
                <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">System ID</p>
                <p className="text-sm font-mono text-slate-500">{scan.id}</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

// -------------------------------------------------------------
// History Page Main Component
// -------------------------------------------------------------
const HistoryPage = () => {
  const [history, setHistory] = useState<ScanHistory[]>([]);
  const [filterMode, setFilterMode] = useState<'all' | 'waste' | 'interactions'>('all');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  
  // Drawer state
  const [selectedScan, setSelectedScan] = useState<ScanHistory | null>(null);
  
  // Modal states
  const [deleteModal, setDeleteModal] = useState<{ isOpen: boolean; scanId: string | null }>({ isOpen: false, scanId: null });
  const [clearModal, setClearModal] = useState(false);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const { colors } = useSettings();

  const fetchHistory = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    try {
      const res = await api.get<ScanHistory[]>("/history?limit=50");
      setHistory(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      if (isRefresh) setRefreshing(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
    // Auto refresh every 5 seconds for continuous scanning updates
    intervalRef.current = setInterval(() => {
      fetchHistory(true);
    }, 5000);

    const handler = () => fetchHistory(true);
    window.addEventListener('sortiq:scan_saved', handler as EventListener);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      window.removeEventListener('sortiq:scan_saved', handler as EventListener);
    };
  }, []);

  const executeDelete = async () => {
    if (!deleteModal.scanId) return;
    try {
      await api.delete(`/history/${deleteModal.scanId}`);
      setHistory(prev => prev.filter(scan => scan.id !== deleteModal.scanId));
      window.dispatchEvent(new CustomEvent('sortiq:scan_saved'));
    } catch (err) {
      console.error(err);
    }
  };

  const executeClearAll = async () => {
    try {
      setLoading(true);
      await api.delete("/history");
      setHistory([]);
      window.dispatchEvent(new CustomEvent('sortiq:scan_saved'));
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getBinColor = (className: string) => {
    const formattedName = className.charAt(0).toUpperCase() + className.slice(1);
    return colors[formattedName] || '#e5e7eb';
  };

  if (loading && history.length === 0) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  const filteredHistory = Array.isArray(history) ? history.filter(scan => {
    if (filterMode === 'all') return true;
    if (filterMode === 'waste') return scan.interaction_type === 'waste';
    return scan.interaction_type !== 'waste';
  }) : [];

  return (
    <div className="max-w-4xl mx-auto pb-12">
      
      {/* Modals & Drawers */}
      <ConfirmModal 
        isOpen={deleteModal.isOpen} 
        onClose={() => setDeleteModal({ isOpen: false, scanId: null })}
        onConfirm={executeDelete}
        title="Delete Scan"
        message="Are you sure you want to permanently delete this scan? This action cannot be undone."
      />
      
      <ConfirmModal 
        isOpen={clearModal} 
        onClose={() => setClearModal(false)}
        onConfirm={executeClearAll}
        title="Clear All History"
        message="Are you sure you want to delete ALL scan history? This action cannot be undone."
      />

      <DetailDrawer 
        scan={selectedScan} 
        isOpen={selectedScan !== null} 
        onClose={() => setSelectedScan(null)}
        binColorHex={selectedScan ? getBinColor(selectedScan.predicted_class) : '#e5e7eb'}
      />

      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Scan History</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-slate-500 bg-white px-3 py-1.5 rounded-full shadow-sm border border-slate-200">
             <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin text-primary" : ""}`} />
             Live Updates
          </div>
          {history.length > 0 && (
            <button 
              onClick={() => setClearModal(true)}
              className="px-4 py-1.5 bg-red-50 text-red-600 hover:bg-red-100 rounded-lg text-sm font-bold transition-colors border border-red-100 shadow-sm"
            >
              Clear All
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2 mb-6 border-b border-slate-200 pb-2">
        {(['all', 'waste', 'interactions'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilterMode(f)}
            className={`px-5 py-2.5 rounded-t-xl font-bold text-sm transition-all border-b-2 -mb-2.5 ${
              filterMode === f 
                ? 'text-indigo-600 border-indigo-600 bg-indigo-50/50' 
                : 'text-slate-500 border-transparent hover:text-slate-800 hover:bg-slate-50'
            }`}
          >
            {f === 'all' ? 'All Scans' : f === 'waste' ? 'Waste Only' : 'Interactions'}
          </button>
        ))}
      </div>

      {/* History List */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        {filteredHistory.length === 0 ? (
          <div className="p-16 text-center text-slate-500 flex flex-col items-center">
            <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4 border border-slate-100">
                <FileImage className="w-8 h-8 text-slate-300" />
            </div>
            <p className="font-bold text-lg text-slate-600">No {filterMode !== 'all' ? filterMode : ''} scans found</p>
            <p className="text-sm mt-1 mb-4">Keep scanning to populate your history dashboard!</p>
            <div className="text-[10px] text-slate-400 bg-slate-50 px-2 py-1 rounded border border-slate-100">
              API: {BASE_URL}
            </div>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {filteredHistory.map((scan) => {
              const isWaste = scan.interaction_type === 'waste';
              const dotColor = isWaste ? getBinColor(scan.predicted_class) : '#6366f1'; // Indigo for interactions
              
              return (
                <div 
                  key={scan.id} 
                  className="p-5 hover:bg-slate-50 transition-colors flex items-center justify-between group cursor-pointer relative"
                  onClick={() => setSelectedScan(scan)}
                >
                  <div className="flex items-center gap-5 w-full">
                    {/* Thumbnail with Color Dot Overlay */}
                    <div className="relative">
                        <div
                          className={`w-16 h-16 rounded-xl flex shrink-0 items-center justify-center overflow-hidden border shadow-sm ${isWaste ? 'bg-slate-100' : 'bg-indigo-50 border-indigo-100'}`}
                          style={isWaste ? { borderColor: dotColor } : undefined}
                        >
                          {scan.image_thumbnail_url ? (
                            <img src={BASE_URL + scan.image_thumbnail_url} alt="thumbnail" className="w-full h-full object-cover" />
                          ) : (
                            <FileImage className={`w-6 h-6 ${isWaste ? 'text-slate-400' : 'text-indigo-400'}`} />
                          )}
                        </div>
                        {/* Dot indicator */}
                        <div 
                            className="absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full border-2 border-white shadow-sm" 
                            style={{ backgroundColor: dotColor }}
                        />
                    </div>

                    <div className="w-full max-w-lg">
                      <div className="flex items-center gap-2">
                        <h3 className="font-bold text-lg text-slate-800 leading-tight capitalize">
                          {scan.predicted_class}
                        </h3>
                        {!isWaste && (
                          <span className="text-[10px] font-black px-2 py-0.5 rounded-md bg-indigo-100 text-indigo-700 uppercase tracking-widest border border-indigo-200">
                            Interaction
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 text-sm font-medium text-slate-400 mt-1">
                        <Calendar className="w-3.5 h-3.5" />
                        {new Date(scan.timestamp).toLocaleString(undefined, {
                          dateStyle: "medium", timeStyle: "short"
                        })}
                      </div>
                    </div>
                  </div>

                  {/* Right Actions */}
                  <div className="flex items-center gap-4 text-right shrink-0">
                    <div className="hidden sm:block">
                      {isWaste ? (
                          <div className={`text-sm font-black ${scan.confidence > 0.70 ? 'text-green-600' : 'text-yellow-600'}`}>
                            {(scan.confidence * 100).toFixed(0)}% <span className="text-slate-400 text-xs font-bold uppercase tracking-wider ml-1">Conf</span>
                          </div>
                      ) : (
                          <div className="text-xs text-indigo-400 font-bold uppercase tracking-wider bg-indigo-50 border border-indigo-100 px-2.5 py-1 rounded-lg">
                            Robot Ping
                          </div>
                      )}
                    </div>
                    
                    {/* Hover actions */}
                    <div className="flex items-center gap-2 transition-all">
                        <button 
                          onClick={(e) => { 
                              e.stopPropagation(); 
                              setDeleteModal({ isOpen: true, scanId: scan.id }); 
                          }}
                          className="p-2.5 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus:opacity-100"
                          title="Delete Scan"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                        <div className="p-2 text-slate-300 sm:group-hover:text-indigo-500 transition-colors">
                            <ChevronRight className="w-5 h-5" />
                        </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default HistoryPage;
