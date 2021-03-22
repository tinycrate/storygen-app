@echo off

python --version
if errorlevel 1 GOTO ERROR_NO_PYTHON
echo,

:CREATE_VENV
if EXIST .\venv GOTO ACTIVATE_VENV
echo Creating virtual environment...
python -m venv venv

:ACTIVATE_VENV
CALL .\venv\Scripts\activate.bat
if errorlevel 1 GOTO ERROR_ACTIVATE

:INSTALL_DEPENDENCIES
python -m pip install -r requirements.txt
if errorlevel 1 GOTO ERROR_INSTALL

:RUN_SHELL
cmd /K ".\venv\Scripts\activate.bat"
EXIT

:ERROR_NO_PYTHON
echo Error: Python is not installed or not in PATH.
PAUSE
EXIT

:ERROR_ACTIVATE
echo Error: Cannot activate virtual environment. Please delete folder venv and try again.
PAUSE
EXIT

:ERROR_INSTALL
echo An error has occured while installing packages.
PAUSE
EXIT