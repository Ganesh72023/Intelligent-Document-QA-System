@echo off
setlocal
echo =======================================================
echo   The Relay Project - 1-Click GitHub Repository Push
echo =======================================================
echo.
echo Make sure you have created an empty repository on GitHub.
echo (e.g., https://github.com/your-username/your-repo-name.git)
echo.
set /p REPO_URL="Enter your GitHub Repository URL: "

if "%REPO_URL%"=="" (
    echo [ERROR] No repository URL provided. Exiting.
    pause
    exit /b 1
)

cd /d "C:\Users\GANESH\Downloads\Vaani_ML_Engineering_Intern_Assignment"

echo.
echo [1/3] Setting remote repository origin...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

echo [2/3] Setting branch to main...
git branch -M main

echo [3/3] Pushing all files to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Your repository has been pushed to GitHub!
    echo Check it at: %REPO_URL%
    echo =======================================================
) else (
    echo.
    echo [NOTE] If git asked for credentials or rejected due to existing files,
    echo you can force push to an empty repo with: git push -u origin main --force
)

echo.
pause

