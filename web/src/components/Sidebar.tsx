import { Link, useLocation } from "react-router-dom";
import { Home, BarChart2, History, Settings } from "lucide-react";
import { cn } from "../lib/utils";

const Sidebar = () => {
  const location = useLocation();
  const path = location.pathname;

  const links = [
    { name: "Home", href: "/", icon: Home },
    { name: "Analytics", href: "/analytics", icon: BarChart2 },
    { name: "History", href: "/history", icon: History },
    { name: "Settings", href: "/settings", icon: Settings },
  ];

  return (
    <div className="w-64 bg-slate-900 h-screen text-slate-100 flex flex-col hidden md:flex fixed top-0 left-0 border-r border-slate-800 pt-20">
      <div className="flex-1 py-6 flex flex-col gap-2 px-4">
        {links.map((link) => (
          <Link
            key={link.name}
            to={link.href}
            className={cn(
              "flex items-center gap-3 px-4 py-3 rounded-md transition-colors",
              path === link.href
                ? "bg-primary text-primary-foreground font-medium"
                : "hover:bg-slate-800 hover:text-white text-slate-400"
            )}
          >
            <link.icon className="w-5 h-5" />
            {link.name}
          </Link>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;
