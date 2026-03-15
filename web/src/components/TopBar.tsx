import { Link } from "react-router-dom";
import { Menu } from "lucide-react";
import { useState } from "react";
import { Home, BarChart2, History, Settings } from "lucide-react";

const TopBar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const links = [
    { name: "Home", href: "/", icon: Home },
    { name: "Analytics", href: "/analytics", icon: BarChart2 },
    { name: "History", href: "/history", icon: History },
    { name: "Settings", href: "/settings", icon: Settings },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-slate-900 border-b border-slate-800 md:hidden">
      <div className="flex items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center gap-3">
          <img
            src="/logo.png"
            alt="SortIQ Logo"
            className="h-10 w-10 object-contain"
          />
          <span className="text-2xl font-black text-white tracking-tighter">
            SortIQ
          </span>
        </Link>
        
        <button 
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="p-2 text-slate-300 hover:text-white"
        >
          <Menu className="w-6 h-6" />
        </button>
      </div>

      {/* Mobile Menu Dropdown */}
      {isMenuOpen && (
        <div className="bg-slate-900 border-b border-slate-800 animate-in slide-in-from-top duration-300">
          <div className="px-4 py-4 flex flex-col gap-2">
            {links.map((link) => (
              <Link
                key={link.name}
                to={link.href}
                onClick={() => setIsMenuOpen(false)}
                className="flex items-center gap-3 px-4 py-3 rounded-md text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
              >
                <link.icon className="w-5 h-5" />
                <span className="font-bold">{link.name}</span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </header>
  );
};

export default TopBar;
