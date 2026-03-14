import { RotateCcw } from "lucide-react";
import { useSettings } from "../context/SettingsContext";

const Settings = () => {
  const { colors, binLabels, setColor, setBinLabel } = useSettings();

  const handleReset = () => {
    if (window.confirm("Are you sure you want to reset all colors and bin labels back to their defaults?")) {
      setColor('Glass', '#22c55e');
      setColor('Plastic', '#3b82f6');
      setColor('Metal', '#eab308');
      setColor('Paper', '#f97316');
      setBinLabel('Glass', 'Green Bin');
      setBinLabel('Plastic', 'Blue Bin');
      setBinLabel('Metal', 'Yellow Bin');
      setBinLabel('Paper', 'Orange Bin');
    }
  };

  return (
    <div className="max-w-3xl mx-auto pb-12">
      <h1 className="text-3xl font-bold mb-8 text-slate-800">System Settings</h1>


      {/* Bin Color Mapping Setting */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="p-6 border-b border-slate-100 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-slate-800">Bin Class Mapping</h2>
            <p className="text-sm text-slate-500 mt-1">Tells the system which recycling bin color each material belongs to. Updates automatically save and sync.</p>
          </div>
          <button 
            onClick={handleReset}
            className="flex items-center gap-2 text-sm font-bold text-slate-500 bg-slate-100 hover:bg-slate-200 hover:text-slate-700 px-4 py-2 rounded-lg transition-colors border border-slate-200"
          >
            <RotateCcw className="w-4 h-4" />
            Reset Defaults
          </button>
        </div>
        
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Create dynamic boxes for the base 4 materials */}
            {(Object.keys(colors) as Array<keyof typeof colors>).map((category) => (
              <div key={category} className="p-4 border border-slate-200 rounded-xl flex flex-col gap-3 bg-slate-50 shadow-sm transition-all hover:border-slate-300">
                <div className="flex justify-between items-center w-full">
                   <span className="font-bold text-slate-700 text-lg">{category as string}</span>
                   
                   <div className="flex items-center gap-3">
                     <span className="text-xs text-slate-500 font-medium">Color:</span>
                     <label className="cursor-pointer flex items-center gap-2 group">
                        <div 
                          className="w-8 h-8 rounded-full shrink-0 shadow-sm border-2 border-slate-200 group-hover:border-slate-300 transition-all shadow-inner relative overflow-hidden"
                          style={{ backgroundColor: colors[category] }}
                        >
                          <input 
                            type="color" 
                            className="absolute opacity-0 -inset-4 w-[200%] h-[200%] cursor-pointer"
                            value={colors[category]}
                            onChange={(e) => setColor(category, e.target.value)}
                          />
                        </div>
                     </label>
                   </div>
                </div>
                
                <div className="flex items-center gap-3 bg-white p-2 rounded-lg border border-slate-200 shadow-sm input-group">
                   <span className="text-xs text-slate-500 font-medium whitespace-nowrap px-1">Bin Label:</span>
                   <input 
                     type="text"
                     className="text-sm font-medium text-slate-700 w-full outline-none bg-transparent"
                     value={binLabels[category] || ''}
                     onChange={(e) => setBinLabel(category, e.target.value)}
                     placeholder="e.g. Green Bin"
                   />
                </div>
              </div>
            ))}

          </div>
        </div>
        
        {/* Save button removed because settings auto-sync via Context useEffect */}
      </div>
    </div>
  );
};

export default Settings;
