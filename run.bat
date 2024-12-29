@echo off
chcp 65001

if not defined PYTHON set PYTHON=python
if not defined VENV_DIR set "VENV_DIR=%~dp0venv"

set ERROR_REPORTING=FALSE

dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 goto :activate_venv

echo.
echo [CREATE VENV]
for /f "delims=" %%i in ('%PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME=%%i
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%"
if %ERRORLEVEL% neq 0 (
    echo Unable to create venv in directory "%VENV_DIR%"
    exit /b
)

:activate_venv
echo.
echo [ACTIVATE VENV]
set PYTHON=%VENV_DIR%\Scripts\Python.exe
echo venv %PYTHON%

echo.
echo [UPGRADE PIP]
%PYTHON% -m pip install --upgrade pip

echo.
echo [INSTALL REQUIREMENTS]
%PYTHON% -m pip install --upgrade -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Unable to install requirements.
    exit /b
)

echo.
echo [LAUNCH APP]
%PYTHON% app.py
if %ERRORLEVEL% neq 0 (
    echo Application failed to launch.
    exit /b
)

pause
exit /b
