# ♻️ SortIQ — AI-Powered Waste Sorting Platform

> 

![SortIQ](logo.png)

---

## 🌐 Live Demo

| Platform | URL |
|----------|-----|
| Web App  | https://sortiq-web.vercel.app |
| API      | https://sortiq-backend.onrender.com |
| API Docs | https://sortiq-backend.onrender.com/docs |

---

## ✨ Key Features

### ♻️ Real-Time Waste Detection
- Live camera feed with colored circles drawn on detected items
- Detects Glass, Plastic, Metal and Paper with high accuracy
- Shows confidence percentage for every detection
- Draws circles in the category color set by user in Settings
- Unknown items shown as "❓ Unknown" — never wrong guesses

### 🤖 7-Robot AI Team
Each robot has a specific role and they work together:

| Robot | Role |
|-------|------|
| 🔍 Scout Robot | YOLOv8 scans every camera frame and finds all objects |
| 🧪 Analyst Robot | MobileNetV2 classifies the material type of each object |
| ✅ Verifier Robot | Cross-checks results using HSV color analysis and texture |
| 🚧 Gatekeeper Robot | Blocks faces, people, animals and furniture from being classified as waste |
| 💬 Narrator Robot | Generates smart, friendly messages for every detection |
| 📸 Capturer Robot | Saves scan only when user clicks the shutter button |
| 📊 Recorder Robot | Stores confirmed scans to history and analytics database |

### 📷 Manual Capture System
- Camera shows live circles on detected objects automatically
- Nothing is saved automatically — user is in full control
- User taps the shutter button to capture and save
- White flash + confirmation when item is saved
- Every saved scan appears in History and Analytics

### 🎨 Customizable Bin Colors
- Go to Settings to change the color of each category
- Glass, Plastic, Metal and Paper each have their own color
- Color changes apply instantly across the entire system:
  - Camera circles use the settings color
  - Analytics charts use the settings color
  - History dots use the settings color
  - Result card dot uses the settings color
- Default colors:
  - 🟢 Glass → Green (#22c55e)
  - 🔵 Plastic → Blue (#3b82f6)
  - 🟡 Metal → Yellow (#eab308)
  - 🟠 Paper → Orange (#f97316)

### 📊 Analytics Dashboard
- Total scans counter
- Scan distribution pie chart (all 4 categories)
- Material breakdown bar chart
- Filter buttons: Glass / Plastic / Metal / Paper
- Click any filter to see all scans for that category
- Each scan card shows: thumbnail, time, confidence, robot message

### 📋 History Dashboard
- Complete list of all saved scans
- Click any scan to open full detail drawer
- Shows: image, confidence bar, bin color, robot tip, timestamp
- Delete single scan with confirmation modal
- Clear all history with confirmation
- Filter by: All / Waste Only / Interactions
- Auto-refreshes every 5 seconds

### ⚙️ Settings
- Change bin color for each category (Glass/Plastic/Metal/Paper)
- Change bin label for each category
- Reset to defaults button
- Settings saved to browser localStorage (persist after refresh)

---

## 🗂️ Waste Categories & Items

### 🟢 Glass
Items detected and placed in Glass bin:
Glass bottle, wine bottle, beer bottle, glass cup,
drinking glass, wine glass, champagne glass, shot glass,
glass jar, jam jar, pickle jar, honey jar, mason jar,
glass bowl, glass vase, glass candle holder, glass plate,
perfume bottle, medicine bottle, sauce bottle, light bulb

### 🔵 Plastic
Items detected and placed in Plastic bin:
Water bottle, plastic bottle, soda bottle, juice bottle,
milk jug, shampoo bottle, detergent bottle, plastic bag,
grocery bag, ziplock bag, plastic container, takeaway box,
yogurt cup, plastic cup, plastic lid, plastic tray,
foam cup, styrofoam box, bottle cap, straw, plastic straw,
plastic spoon, plastic fork, plastic wrap, food packaging

### 🟡 Metal
Items detected and placed in Metal bin:
Aluminum can, soda can, beer can, energy drink can,
food tin, canned food, paint can, aerosol can,
metal fork, metal knife, metal spoon, metal straw,
metal bowl, aluminum foil, foil tray, metal bottle cap,
scissors, metal key, metal coin, metal ruler,
paper clip, staple, metal wire, metal pipe

### 🟠 Paper
Items detected and placed in Paper bin:
Paper sheet, white paper, lined paper, newspaper,
magazine, notebook, notepad, book, textbook,
comic book, paperback, diary, cardboard box,
cardboard sheet, pizza box, cereal box, shoe box,
delivery box, egg carton, envelope, paper bag,
paper cup, paper plate, tissue paper, receipt,
brochure, flyer, poster, sticky note, greeting card

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |
| Backend | FastAPI + Python 3.10 |
| Object Detection | YOLOv8n (Lightweight) |
| Waste Classification | MobileNetV2 (TensorFlow) |
| Image Processing | OpenCV + Pillow |
| Database | SQLite + SQLAlchemy |
| Hosting | Vercel (Frontend Only) |

---

## 🚀 Run Locally

### Requirements
- Python 3.10+
- Node.js 18+
- Your trained model file: `backend/model/sortiq_model.h5`
- Your classes file: `backend/model/classes.json`

### Step 1 — Start the Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Backend runs at: http://localhost:8001
API docs at: http://localhost:8001/docs

### Step 2 — Start the Frontend

```bash
cd web

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs at: http://localhost:5173

### Step 3 — Open the App

Open your browser and go to:
http://localhost:5173

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check + model status |
| POST | /predict-upload | Manual capture — saves to DB |
| POST | /predict-realtime | Live preview only — no save |
| GET | /history | Get scan history |
| DELETE | /history/{id} | Delete a scan |
| DELETE | /history | Clear all history |
| GET | /stats | Analytics data |
| POST | /settings | Update bin settings |

---

## 🧠 How Detection Works

```
User opens camera
      ↓
Scout Robot (YOLO) scans frame every 800ms
      ↓
Gatekeeper blocks: faces, people, animals,
furniture, vehicles, food items
      ↓
Analyst Robot (MobileNetV2) classifies
each detected object → Glass/Plastic/Metal/Paper
      ↓
Verifier Robot cross-checks using:
  - HSV color analysis (glass vs metal)
  - Texture analysis (paper vs plastic)
  - Edge detection (glass sharp edges)
  - Skin tone detection (block faces)
      ↓
Narrator Robot generates friendly message
based on detected material
      ↓
Painter Robot draws colored circles on screen
      ↓
User sees circles + robot message
      ↓
User taps shutter button
      ↓
Capturer Robot takes high quality photo
      ↓
Recorder Robot saves to database
      ↓
History + Analytics updated
```

---

## 🎨 Customizing Colors

1. Open the web app
2. Click **Settings** in the navigation
3. Click the color picker next to each category
4. Choose any color you want
5. Changes apply instantly everywhere:
   - Camera circles
   - Analytics charts
   - History dots
   - Result cards
6. Colors saved automatically — persist after refresh
7. Click **Reset to Defaults** to restore original colors

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — built for recycling education and sustainability.

---

## 👨💻 Author

Built with ♻️ by the SortIQ team.
