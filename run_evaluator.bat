@echo off
chcp 65001 > nul
echo モデル評価ツールを起動します...
cd /d "%~dp0"
py app\evaluate_model.py
pause
