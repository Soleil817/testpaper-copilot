@echo off
cd /d "%~dp0"
echo ============================================
echo  Installing Python packages for TestPaper Copilot
echo  This may take 2-5 minutes.
echo ============================================
where py >nul 2>nul
if %errorlevel%==0 (
  py -m pip install -r requirements.txt
) else (
  python -m pip install -r requirements.txt
)
echo.
echo Done. If you see errors, copy them and ask for help.
pause
