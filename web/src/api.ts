import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: BASE_URL,
});

// Configure API to always attach color settings if they exist in local storage
api.interceptors.request.use((config) => {
  if (config.url === '/predict-realtime' || config.url === '/predict') {
    try {
      const stored = localStorage.getItem('sortiq_settings_v2');
      if (stored) {
        const settings = JSON.parse(stored);
        if (settings.colors) {
           config.headers['X-Color-Glass'] = settings.colors.Glass || '#22c55e';
           config.headers['X-Color-Plastic'] = settings.colors.Plastic || '#3b82f6';
           config.headers['X-Color-Metal'] = settings.colors.Metal || '#eab308';
           config.headers['X-Color-Paper'] = settings.colors.Paper || '#f97316';
        }
      }
    } catch (e) {
      // Ignore parse errors
    }
  }
  return config;
});

export const syncSettingsWithBackend = async (threshold: number, binMapping?: any) => {
  try {
    await api.post('/settings', { threshold, binMapping });
  } catch (err) {
    console.error("Failed to sync settings with backend:", err);
  }
};

export { BASE_URL };
export default api;
