# ♻️ SortIQ — AI-Powered Waste Sorting Platform

SortIQ is a high-precision waste classification 
system that uses computer vision and AI to help 
users sort waste correctly into Glass, Plastic, 
Metal and Paper categories.

## 🌐 Live Demo
- **Web App:** https://sortiq-web.vercel.app
- **API:** https://sortiq-backend.onrender.com

## 🤖 How It Works
SortIQ uses a team of AI robots working together:
- **Scout Robot** — YOLOv8 detects all objects
- **Analyst Robot** — MobileNetV2 classifies waste
- **Verifier Robot** — Corrects misclassifications
- **Narrator Robot** — Generates smart messages
- **Recorder Robot** — Saves scan history

## ♻️ Waste Categories
| Category | Examples |
|----------|---------|
| 🟢 Glass | Bottles, cups, jars, vases |
| 🔵 Plastic | Water bottles, bags, containers |
| 🟡 Metal | Aluminum cans, tins, cutlery |
| 🟠 Paper | Cardboard, newspaper, books |

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript + Tailwind CSS |
| Backend | FastAPI + Python |
| Detection | YOLOv8 + MobileNetV2 |
| Database | SQLite |
| Hosting | Vercel (web) + Render (API) |

## 🚀 Local Development

### Backend
  cd backend
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8001

### Frontend
  cd web
  npm install
  npm run dev

## 📁 Project Structure
  SortIQ/
    backend/     → FastAPI + AI models
    web/         → React frontend
    .github/     → CI/CD workflows

## 📄 License
MIT License — built for recycling education.
