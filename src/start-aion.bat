@echo off
echo =============================
echo Launching AION Consciousness + Cloudflare Tunnel
echo =============================

:: Start backend
cd C:\Users\riyar\AION\aion_backend
start cmd /k "python server.py"

:: Wait for backend to start
timeout /t 15 >nul

:: Start Cloudflare Tunnel with permanent name
cd C:\Users\riyar\AION\aion_interface
start cmd /k "cloudflared.exe tunnel --name aion --url http://localhost:5000"
