@echo off
REM ======================================================
REM Auto-restart Cloudflare tunnel for AION
REM Always-on, silent, and auto-copies public URL
REM ======================================================

set CLOUD="C:\Users\riyar\AION\aion_interface\cloudflared.exe"
set PORT=3000
set LOG="%TEMP%\aion_tunnel.log"

:RESTART_TUNNEL
REM Clear previous log
if exist %LOG% del %LOG%

REM Start tunnel in background
start "" /b cmd /c "%CLOUD% tunnel run aion --url http://localhost:%PORT% > %LOG% 2>&1"

REM Wait for tunnel to initialize
timeout /t 5 > nul

REM Grab the public URL from log
for /f "tokens=*" %%i in ('findstr /r /c:"https://.*trycloudflare.com" %LOG%') do set URL=%%i

REM Copy URL to clipboard
if defined URL (
    echo %URL% | clip
    echo Tunnel URL copied to clipboard: %URL%
) else (
    echo Waiting for tunnel URL...
)

REM Monitor tunnel connection every 10 seconds
:MONITOR
timeout /t 10 > nul
findstr /i "Connection terminated" %LOG% > nul
if %errorlevel%==0 (
    echo Tunnel disconnected, restarting...
    goto RESTART_TUNNEL
)
goto MONITOR
