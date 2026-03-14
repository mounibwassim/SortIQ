import React, { createContext, useContext, useState, useEffect } from 'react';
import { syncSettingsWithBackend } from '../api';

export interface ColorsState {
  Glass: string;
  Plastic: string;
  Metal: string;
  Paper: string;
  [key: string]: string; // Fallback
}

export interface BinLabelsState {
  Glass: string;
  Plastic: string;
  Metal: string;
  Paper: string;
  [key: string]: string; // Fallback
}

interface SettingsState {
  threshold: number;
  colors: ColorsState;
  binLabels: BinLabelsState;
}

interface SettingsContextType extends SettingsState {
  setThreshold: (val: number) => void;
  setColor: (category: keyof ColorsState, colorHex: string) => void;
  setBinLabel: (category: keyof BinLabelsState, label: string) => void;
}

const defaultSettings: SettingsState = {
  threshold: 0.30,
  colors: {
    Glass: "#22c55e",
    Plastic: "#3b82f6",
    Metal: "#eab308",
    Paper: "#f97316"
  },
  binLabels: {
    Glass: "Green Bin",
    Plastic: "Blue Bin",
    Metal: "Yellow Bin",
    Paper: "Orange Bin"
  }
};

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export const SettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<SettingsState>(() => {
    const stored = localStorage.getItem('sortiq_settings_v2');
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch (e) {
         return defaultSettings;
      }
    }
    return defaultSettings;
  });

  // Sync to localStorage and Backend whenever settings change
  useEffect(() => {
    localStorage.setItem('sortiq_settings_v2', JSON.stringify(settings));
    syncSettingsWithBackend(settings.threshold, { colors: settings.colors, binLabels: settings.binLabels });
  }, [settings]);

  const setThreshold = (val: number) => {
    setSettings(prev => ({ ...prev, threshold: val }));
  };

  const setColor = (category: keyof ColorsState, colorHex: string) => {
    setSettings(prev => ({
      ...prev,
      colors: { ...prev.colors, [category]: colorHex }
    }));
  };

  const setBinLabel = (category: keyof BinLabelsState, label: string) => {
    setSettings(prev => ({
      ...prev,
      binLabels: { ...prev.binLabels, [category]: label }
    }));
  };

  return (
    <SettingsContext.Provider value={{ ...settings, setThreshold, setColor, setBinLabel }}>
      {children}
    </SettingsContext.Provider>
  );
};

export const useSettings = () => {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
};
