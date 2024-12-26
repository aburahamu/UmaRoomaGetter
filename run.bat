@echo off
chcp 65001

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

set ERROR_REPORTING=FALSE

dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 goto :activate_venv

echo.
echo [CREATE VENV]
for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%"
if %ERRORLEVEL% == 0 goto :activate_venv
echo Unable to create venv in directory "%VENV_DIR%"

:activate_venv
echo.
echo [ACTIVATE VENV]
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%

echo.
echo [UPGRADE PIP]
python -m pip install --upgrade pip

echo.
echo [INSTALL REQUIREMENTS]
pip install --upgrade -r requirements.txt

echo.
echo [LAUCH APP]
%PYTHON% app.py
pause
exit /b