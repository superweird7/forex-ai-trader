@echo off
title Gold AI Signal Dashboard
echo ============================================
echo   GOLD AI SIGNAL SCORER - Dashboard
echo   Starting...
echo ============================================
echo.
cd /d D:\FOREX\dashboard
start http://localhost:5000
python app.py
pause
