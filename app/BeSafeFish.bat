@echo off
REM Launcher BeSafeFish — uruchamia GUI z uzyciem Python Launchera (py).
REM Uruchamiac z rootu repo lub z folderu app/ (skrypt sam ustawia CWD = app/).
cd /d "%~dp0"
py besafefish.py
if errorlevel 1 pause
