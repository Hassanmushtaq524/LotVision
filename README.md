# 🚗 LOTVISION: AI-powered Parking Lot Management System

## 📌 Overview
This project is a **Parking Lot Management System** with:
- **Frontend:** React + TailwindCSS
- **Backend:** FastAPI (Python)
- **AI Integration:** Google Gemini for suspicious vehicle analysis

The system logs vehicle entries, flags unauthorized cars, and uses **Gemini AI** to detect suspicious activity based on timestamps and entry patterns.

---

## 📂 Project Structure
```
./
├── backend/       # FastAPI backend
│   ├── main.py    # API entry point
│   ├── models.py  # Database models
│   ├── routes/    # API routes
│   ├── db.py      # Database connection
│   ├── .env       # Environment variables
│   └── ...
├── frontend/      # React + Tailwind frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── App.js
│   │   ├── index.js
│   │   └── ...
│   ├── .env       # Environment variables
│   └── package.json
├── README.md      # Project documentation
└── ...
```

---

## 🚀 Setup & Installation

### 1️⃣ **Clone the repository**
```sh
git clone https://github.com/HassanMushtaq524/lotvision.git
cd lotvision
```

### 2️⃣ **Backend Setup (FastAPI)**
#### 📌 Install dependencies
```sh
cd backend
python -m venv venv  # Create virtual environment
source venv/bin/activate  # Activate venv (Mac/Linux)
venv\Scripts\activate  # Activate venv (Windows)
pip install -r requirements.txt
```

#### 📌 Set up environment variables (Create a `.env` file)
```ini
DATABASE_URL=your-sqlite-db
GEMINI_API_KEY=your-google-gemini-api-key
```

#### 📌 Run the FastAPI server
```sh
uvicorn main:app --reload
```
FastAPI will run at **http://127.0.0.1:8000** 🚀

### 3️⃣ **Frontend Setup (React + TailwindCSS)**
#### 📌 Install dependencies
```sh
cd ../frontend
npm install
```

#### 📌 Set up environment variables (Create a `.env` file)
```ini
REACT_APP_BACKEND_URL=http://127.0.0.1:8000
REACT_APP_GEMINI_API_KEY=your-google-gemini-api-key
```

#### 📌 Start the frontend
```sh
npm start
```
The React app will run at **http://localhost:3000** 🎨

---

## 🔥 Features
✅ **Vehicle Entry & Exit Logging** - Tracks cars entering and exiting the parking lot.
✅ **Unauthorized Vehicle Detection** - Flags unregistered or misplaced vehicles.
✅ **AI-Powered Suspicious Vehicle Analysis** - Uses Google Gemini AI to analyze entry patterns.
✅ **Real-time Logs & Alerts** - Displays flagged vehicles in an easy-to-read dashboard.

---

## ⚡ Tech Stack
- **Frontend:** React, TailwindCSS
- **Backend:** FastAPI, SQLAlchemy
- **Database:** PostgreSQL
- **AI Integration:** Google Gemini API

---

## 🛠️ Future Improvements
🔹 **User Authentication** - Admin login for monitoring vehicle logs.  
🔹 **Automated Alerts** - Send notifications for flagged vehicles through SMS.

---

## 💡 Contributing
Want to improve the project? Feel free to open an issue or submit a pull request!

🚀 Happy Coding! 😃

