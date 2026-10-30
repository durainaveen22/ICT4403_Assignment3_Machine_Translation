@echo off
echo ======================================================================
echo ICT4403 Assignment 3 - GitHub Upload Setup Script
echo ======================================================================
echo.

:: Initialize git repository if not already initialized
if not exist ".git" (
    echo [1/4] Initializing local git repository...
    git init
) else (
    echo [1/4] Git repository already initialized.
)

:: Stage all files
echo [2/4] Staging all files...
git add .

:: Commit files
echo [3/4] Creating initial commit...
git commit -m "Complete Assignment 3: Machine Translation Systems (Basic Transformer, mT5, LLaMA, Gemma)"

echo.
echo ======================================================================
echo Next Steps to Push to GitHub:
echo 1. Go to https://github.com/new and create a repository named:
echo    ICT4403_Assignment3_Machine_Translation
echo.
echo 2. Run the following commands (replace YOUR_USERNAME with your GitHub username):
echo    git remote add origin https://github.com/YOUR_USERNAME/ICT4403_Assignment3_Machine_Translation.git
echo    git branch -M main
echo    git push -u origin main
echo ======================================================================
echo.
pause
