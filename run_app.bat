@echo off
cd /d "%~dp0"
echo =====================================================================
echo   SatAssist AI - Satellite Image QA & Visual Grounding Chatbot
echo =====================================================================
echo.
echo Starting FastAPI Backend (http://127.0.0.1:8001)...
start "SatAssist Backend" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe run_backend.py"

timeout /t 2 >nul

echo Starting React + Vite Frontend (http://localhost:5173)...
start "SatAssist Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo =====================================================================
echo   System is running!
echo   Frontend Dashboard: http://localhost:5173
echo   Backend API Docs:   http://127.0.0.1:8001/docs
echo =====================================================================
