# How to Upload QwiSense to GitHub (WSL Guide)

## Step 1 — Install Git in WSL
```bash
sudo apt update
sudo apt install git -y
```

## Step 2 — Configure Git with your details
```bash
git config --global user.name "Nemin Orniel"
git config --global user.email "your_email@karunya.edu"
```

## Step 3 — Create repo on GitHub
1. Go to https://github.com/new
2. Repository name: `qwisense`
3. Description: `Quantum-Enhanced WiFi Human Sensing System`
4. Set to **Public**
5. Do NOT initialize with README (we already have one)
6. Click **Create repository**

## Step 4 — Initialize git in your project folder
```bash

cd ~/qwisense          # or wherever you placed the project
git init
git add .
git commit -m "Initial commit: QwiSense quantum WiFi sensing system"
```

## Step 5 — Connect and push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/qwisense.git
git branch -M main
git push -u origin main
```

## Step 6 — Add a GitHub Actions badge (optional, looks professional)
Add this to the top of README.md:
```markdown
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-purple)
![License](https://img.shields.io/badge/License-MIT-green)
```

## Future pushes (after making changes)
```bash
git add .
git commit -m "Add: quantum VQC training results"
git push
```

## Recommended repo structure on GitHub
- Pin the repo on your profile
- Add topics: `quantum-computing`, `wifi-sensing`, `machine-learning`, `python`, `pennylane`
- Link to your paper/report in the About section
