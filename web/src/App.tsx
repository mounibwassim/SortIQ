import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import Home from "./pages/Home";
import Analytics from "./pages/Analytics";
import HistoryPage from "./pages/History";
import Settings from "./pages/Settings";
import { SettingsProvider } from "./context/SettingsContext";

function App() {
  return (
    <SettingsProvider>
      <Router>
      <div className="flex min-h-screen bg-slate-50">
        <Sidebar />
        <TopBar />
        <main className="flex-1 md:ml-64 p-8 pt-20">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </Router>
    </SettingsProvider>
  );
}

export default App;
