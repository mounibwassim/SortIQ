import { useState, useEffect, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Loader2, Activity, RefreshCw } from "lucide-react";
import api, { BASE_URL } from "../api";
import { useSettings } from "../context/SettingsContext";

interface StatsResponse {
  total_scans: number;
  model_accuracy: number;
  average_confidence: number;
  class_distribution: Record<string, number>;
  interaction_stats: Record<string, number>;
  waste_by_class: Record<string, number>;
  total_waste_scans: number;
}

interface ScanHistory {
  id: string;
  timestamp: string;
  predicted_class: string;
  confidence: number;
  image_thumbnail_url: string | null;
  interaction_type: string;
  robot_message: string | null;
}

const Analytics = () => {
  const { colors } = useSettings();
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [history, setHistory] = useState<ScanHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStats = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    try {
      const [statsRes, historyRes] = await Promise.all([
        api.get<StatsResponse>("/stats"),
        api.get<ScanHistory[]>("/history?limit=100")
      ]);
      setStats(statsRes.data);
      setHistory(historyRes.data);
    } catch (err) {
      console.error(err);
    } finally {
      if (isRefresh) setRefreshing(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    
    // Auto refresh every 5 seconds for continuous scanning updates
    intervalRef.current = setInterval(() => {
      fetchStats(true);
    }, 5000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Listen for saved scans and refresh immediately
  useEffect(() => {
    const handler = () => fetchStats(true);
    window.addEventListener('sortiq:scan_saved', handler as EventListener);
    return () => window.removeEventListener('sortiq:scan_saved', handler as EventListener);
  }, []);

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-slate-400 gap-3">
        <Activity className="w-10 h-10" />
        <p className="font-medium">Error loading analytics data.</p>
        <p className="text-xs">Connecting to: <code className="bg-slate-100 px-1 rounded">{BASE_URL}</code></p>
        <p className="text-xs text-slate-500 mt-2">Check if the backend is running and the URL is correct.</p>
      </div>
    );
  }

  // TASK 1: Fix crash (null safety)
  const safe = {
    waste_by_class:    stats?.waste_by_class    ?? {},
    class_distribution:stats?.class_distribution?? {},
    total_scans:       stats?.total_scans       ?? 0,
    total_waste:       stats?.total_waste_scans ?? 0,
  };

  const safeInteractionStats = stats.interaction_stats ?? {};

  // TASK 4: Fix Scan Distribution + Material Breakdown (Use safe.waste_by_class explicitly)
  // Ensure we map standard capitalized names expected by waste_by_class
  const chartData = Object.entries(safe.waste_by_class)
    .map(([name, value]) => {
      const displayName = name.charAt(0).toUpperCase() + name.slice(1);
      return {
        name: displayName,
        value,
        fill: colors[displayName] || '#6b7280'
      };
    });

  // TASK 3: Filter History computation
  const filteredHistory = (selectedCategory && Array.isArray(history))
    ? history.filter(s => s.predicted_class && s.predicted_class.toLowerCase() === selectedCategory.toLowerCase() && s.interaction_type === 'waste')
    : [];

  const getRecyclingTip = (material: string) => {
    switch (material) {
      case 'Glass': return 'Rinse carefully. Remove lids if metal.';
      case 'Plastic': return 'Empty liquids. Crush to save space.';
      case 'Metal': return 'Rinse cans. Do not crush aluminum if prohibited.';
      case 'Paper': return 'Most paper products are highly recyclable!';
      default: return 'Recycle responsibly.';
    }
  };

  return (
    <div className="max-w-6xl mx-auto pb-12">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Analytics Dashboard</h1>
        <div className="flex items-center gap-2 text-sm text-slate-500 bg-white px-3 py-1.5 rounded-full shadow-sm border border-slate-200">
           <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin text-primary" : ""}`} />
           Auto-updating
        </div>
      </div>
      
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex items-center gap-5">
          <div className="p-4 bg-primary/10 text-primary rounded-lg border border-primary/20 shadow-inner">
            <Activity className="w-8 h-8" />
          </div>
          <div>
            <p className="text-slate-500 font-medium uppercase tracking-wider text-sm mb-1">Total Scans</p>
            <h2 className="text-4xl font-black text-slate-800">{safe.total_scans}</h2>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex items-center gap-5">
          <div className="p-4 bg-green-500/10 text-green-600 rounded-lg border border-green-500/20 shadow-inner">
            <Activity className="w-8 h-8" />
          </div>
          <div>
            <p className="text-slate-500 font-medium uppercase tracking-wider text-sm mb-1">Total Waste Items</p>
            <h2 className="text-4xl font-black text-slate-800">{safe.total_waste}</h2>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-8 border-l-4 border-l-indigo-500">
        <h3 className="text-lg font-bold mb-4 text-slate-700 flex items-center gap-2">
            🤖 Robot Interactions
            <span className="text-sm font-normal text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full ml-auto">Non-Waste Objects</span>
        </h3>
        <div className="flex flex-wrap gap-3">
            {Object.entries(safeInteractionStats).map(([k, v]) => (
                <div key={k} className="px-3 py-1.5 bg-indigo-50 text-indigo-700 rounded-lg text-sm font-bold border border-indigo-100 flex items-center gap-2 capitalize">
                    {k}
                    <span className="bg-white px-2 rounded disabled text-indigo-500">{v}</span>
                </div>
            ))}
            {Object.keys(safeInteractionStats).length === 0 && (
                <div className="text-sm text-slate-400 italic">No fun interactions recorded yet.</div>
            )}
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-10">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold mb-4 text-slate-700">Scan Distribution</h3>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} margin={{ top: 4, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12 }} allowDecimals={false} />
                <Tooltip cursor={{ fill: '#f1f5f9' }} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[280px] flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-200 rounded-lg">
              <p className="font-medium text-sm">No scans yet</p>
              <p className="text-xs mt-1">Scan items to see data here</p>
            </div>
          )}
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold mb-4 text-slate-700">Material Breakdown</h3>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={65}
                  outerRadius={105}
                  paddingAngle={4}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${((percent || 0) * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [Number(value), 'Scans']} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[280px] flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-200 rounded-lg">
              <p className="font-medium text-sm">No scans yet</p>
              <p className="text-xs mt-1">Scan items to see data here</p>
            </div>
          )}
        </div>
      </div>

      <div className="mb-4 text-center">
        <h3 className="text-xl font-bold text-slate-700 mb-6">Explore Scan Database</h3>
        <div className="flex flex-wrap justify-center gap-4 mb-6">
          {[
            { key: 'Glass', icon: '🟢' },
            { key: 'Plastic', icon: '🔵' },
            { key: 'Paper', icon: '📄' },
            { key: 'Metal', icon: '⚡' }
          ].map(cat => (
            <button
              key={cat.key}
              onClick={() => setSelectedCategory(selectedCategory === cat.key ? null : cat.key)}
              className={`px-6 py-3 rounded-xl border-2 transition-all font-bold flex items-center gap-2 shadow-sm ${
                selectedCategory === cat.key 
                  ? 'bg-white'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
              style={selectedCategory === cat.key ? { borderColor: colors[cat.key], color: colors[cat.key] } : {}}
            >
              <span>{cat.icon}</span> 
              {cat.key}
            </button>
          ))}
        </div>
      </div>

      {selectedCategory && (
        <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200 mb-8 animate-in fade-in slide-in-from-top-4 duration-300">
          <h3 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
            Recent {selectedCategory} Scans
            <span className="text-sm font-semibold text-slate-700 bg-white border border-slate-200 px-2.5 py-0.5 rounded-full shadow-sm">
              {filteredHistory.length}
            </span>
          </h3>
          
          {filteredHistory.length === 0 ? (
            <div className="text-center py-12 text-slate-500">
              <p className="font-medium text-lg">No {selectedCategory} scans yet — start scanning!</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredHistory.map(scan => (
                <div key={scan.id} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden flex flex-col group hover:shadow-md transition-all">
                  <div className="h-48 bg-slate-100 relative overflow-hidden">
                    {scan.image_thumbnail_url ? (
                      <img src={BASE_URL + scan.image_thumbnail_url} alt="thumbnail" className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-slate-400">
                        No Image
                      </div>
                    )}
                    <div className="absolute top-3 right-3 bg-white/95 backdrop-blur px-2.5 py-1 rounded text-xs font-bold text-slate-700 shadow-sm border border-slate-100">
                      {(scan.confidence * 100).toFixed(1)}% Match
                    </div>
                  </div>
                  <div className="p-5 flex-1 flex flex-col">
                    <div className="text-xs text-slate-400 mb-3 font-medium uppercase tracking-wider">
                      {new Date(scan.timestamp).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                    </div>
                    {scan.robot_message && (
                      <div className="mt-auto mb-4 bg-indigo-50/80 rounded-lg p-3.5 border border-indigo-100 relative before:absolute before:-top-2 before:left-4 before:border-8 before:border-transparent before:border-b-indigo-100">
                        <p className="text-sm italic font-medium" style={{ color: colors[selectedCategory] || '#312e81' }}>
                          "{scan.robot_message.split('\n').join(' ')}"
                        </p>
                      </div>
                    )}
                    <div className="mt-auto text-xs font-semibold px-3 py-2.5 rounded-lg border flex items-start gap-2"
                         style={{ backgroundColor: `${colors[selectedCategory]}10`, borderColor: `${colors[selectedCategory]}30`, color: colors[selectedCategory] }}>
                      <span className="shrink-0 text-sm">♻️</span>
                      {getRecyclingTip(selectedCategory)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

    </div>
  );
};

export default Analytics;
