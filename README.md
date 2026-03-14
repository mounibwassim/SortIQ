# ♻️ SortIQ - AI-Powered Universal Waste Sorting Platform

**SortIQ** is a high-precision, multi-platform waste classification system that combines real-time computer vision with a rich analytics dashboard. It empowers users to sort waste accurately using AI "Robots" that specialize in different materials.

![SortIQ Preview](logo.png)

## 🧠 The 7-Robot Engine
SortIQ doesn't just run a model; it uses a multi-stage **7-Robot Architecture** to ensure precision and privacy:
1.  **Scout Robot (YOLOv8)**: Rapidly identifies object locations and boundaries.
2.  **Analyst Robot (MobileNetV2)**: Performs deep material classification using TTA (Test-Time Augmentation).
3.  **Gatekeeper Robot**: Protects privacy by blocking classification if a face or human skin is detected.
4.  **Material Specialist Robots**: Hardcoded secondary checks for Glass, Metal, Paper, and Plastic.
5.  **Separator Robot**: Breaks "ties" between similar looking materials (like Glass vs. Metal).
6.  **Narrator Robot**: Generates personalized recycling guidance based on the item and user settings.
7.  **Recorder Robot**: Persists data to history and synchronizes analytics in real-time.

## 🚀 Features
-   **Real-time Detection**: Zero-latency item identification via WebCam or Mobile Camera.
-   **Mobile App (v1.1.0)**: Native Android APK with local network sync and shutter-save functionality.
-   **Analytics Dashboard**: Visual breakdown of recycling habits, Material Distribution, and Success Rates.
-   **Dynamic Personalization**: Users can customize recycling bin colors in settings, and the AI adapts its Narrator messages instantly.
-   **Privacy-First**: Automatic facial detection blocks and hides sensitive data before it's processed.

## 🛠 Tech Stack
-   **Backend**: Python 3.10 | FastAPI | TensorFlow (tf-keras) | SQLite | SQLAlchemy
-   **AI Engine**: YOLOv8s | MobileNetV2 (Legacy Keras 2) | OpenCV
-   **Web**: React 18 | Vite | TailwindCSS | Recharts | Lucide Icons
-   **Mobile**: React Native | Expo 51 | Axios

## 🏁 Quick Start

### 1. Unified Backend
SortIQ requires a dual-model setup (YOLO + MobileNet).
```bash
cd backend
# Create virtual env
python -m venv .venv
.venv\Scripts\activate

# Install dependencies (ensure tf-keras is included for legacy models)
pip install -r requirements.txt
pip install tf-keras

# Start the AI Server
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 2. Web Dashboard
```bash
cd web
npm install
npm run dev
```

### 3. Mobile App (v1.1.0)
```bash
cd SortIQ-Mobile
npm install
# To run locally:
npx expo start
# Or use the pre-built APK in the latest repo releases.
```

## 📋 APK Setup
The mobile app communicates with the laptop via the local IP. 
1. Find your IP (`ipconfig`).
2. Update `SortIQ-Mobile/src/api/index.ts` with your IP.
3. Both must be on the same WiFi network.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
