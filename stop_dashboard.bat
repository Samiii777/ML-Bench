@echo off
echo 🛑 Stopping ML-Bench Dashboard...
echo.

:: Kill all python processes running streamlit
tasklist | find "python.exe" >nul
if %errorlevel% == 0 (
    echo 🔍 Found Python processes, stopping Streamlit...
    taskkill /f /im python.exe >nul 2>&1
    echo ✅ Dashboard stopped successfully!
) else (
    echo ℹ️  No Python processes found running.
)

:: Also try to kill by port if netstat is available
netstat -ano | find ":8501" >nul
if %errorlevel% == 0 (
    echo 🔍 Found process on port 8501, stopping...
    for /f "tokens=5" %%a in ('netstat -ano ^| find ":8501"') do taskkill /f /PID %%a >nul 2>&1
)

echo.
echo 🎯 If the dashboard is still running:
echo    1. Open Task Manager (Ctrl+Shift+Esc)
echo    2. Look for 'python.exe' processes
echo    3. End those processes manually
echo.
pause 