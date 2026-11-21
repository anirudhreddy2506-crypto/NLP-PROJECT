# VisionCaption AI - Complete Deployment Guide

This guide details **four practical deployment options** for VisionCaption AI, ranked from easiest and best suited for an ML capstone portfolio to containerized cloud production.

---

## 🌟 Comparison of Deployment Options

| Deployment Target | Cost | Hardware / RAM | Setup Complexity | Best For |
| :--- | :---: | :---: | :---: | :--- |
| **1. Hugging Face Spaces** | **Free** | 2 vCPU, **16 GB RAM** | Low (Git push) | **Recommended:** Capstone viva, portfolios, public demos |
| **2. Docker Container** | Variable | Any (AWS, GCP, Render) | Medium (Dockerfile) | Production, VM hosting, isolated testing |
| **3. Streamlit Cloud** | Free | 1 GB RAM (Limited) | Very Low | Light UI demos (Requires memory care) |
| **4. Cloudflare / ngrok Tunnel** | Free | Uses your MacBook M-series | Minimal (1 CLI command) | Live viva presentations directly from local Mac |

---

## Option 1: Hugging Face Spaces (Recommended)

Hugging Face Spaces provides a **free 16 GB RAM / 2 vCPU environment** with native Streamlit support, which easily accommodates BLIP and PyTorch vision-language models without running out of memory.

### Step-by-Step Instructions:

1. **Create a New Space on Hugging Face:**
   - Log in to [huggingface.co](https://huggingface.co).
   - Click your profile icon $\to$ **New Space**.
   - **Space Name:** `visioncaption-ai`
   - **License:** `mit` or `apache-2.0`
   - **Space SDK:** Choose **Streamlit**
   - **Hardware:** Select **CPU basic (2 vCPU, 16GB RAM, Free)**.
   - Click **Create Space**.

2. **Clone your Space Repository:**
   ```bash
   git clone https://huggingface.co/spaces/<YOUR_USERNAME>/visioncaption-ai /tmp/hf-space
   ```

3. **Copy Project Files into the Space:**
   ```bash
   # Copy application files into the cloned space
   cp -r app.py requirements.txt configs/ src/ data/ outputs/ /tmp/hf-space/
   ```

4. **Add Hugging Face Space Metadata to `README.md` in the Space:**
   Create or edit `/tmp/hf-space/README.md` with this frontmatter:
   ```yaml
   ---
   title: VisionCaption AI
   emoji: 👁️
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   sdk_version: "1.33.0"
   app_file: app.py
   pinned: false
   ---
   ```

5. **Push and Deploy:**
   ```bash
   cd /tmp/hf-space
   git add .
   git commit -m "Deploy VisionCaption AI to Hugging Face Spaces"
   git push
   ```
   Hugging Face will automatically build the environment, install `requirements.txt`, and launch your app with a permanent public link:  
   `https://huggingface.co/spaces/<YOUR_USERNAME>/visioncaption-ai`

---

## Option 2: Docker Container (AWS, GCP, Render, Fly.io, or Local)

The project includes a production-ready `Dockerfile` and `.dockerignore`.

### 1. Build the Docker Image:
```bash
cd "/Users/anirudh/Desktop/NLP PROJECT"
docker build -t visioncaption-ai:latest .
```

### 2. Run the Container Locally:
```bash
docker run -p 8501:8501 --name visioncaption visioncaption-ai:latest
```
Open your browser at `http://localhost:8501`.

### 3. Deploy Container to Cloud Platforms:
- **Render.com / Fly.io / Railway:**
  - Connect your GitHub repository.
  - Choose Docker deployment.
  - Set the internal port to `8501`.
- **AWS Elastic Beanstalk / ECS / GCP Cloud Run:**
  - Push image to AWS ECR or GCP Artifact Registry:
    ```bash
    docker tag visioncaption-ai:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/visioncaption-ai:latest
    docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/visioncaption-ai:latest
    ```

---

## Option 3: Streamlit Community Cloud

Streamlit Community Cloud is integrated with GitHub for continuous deployment.

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit for VisionCaption AI"
   git remote add origin https://github.com/<YOUR_USERNAME>/VisionCaptionAI.git
   git branch -M main
   git push -u origin main
   ```
   *(Note: Exclude model checkpoint binaries exceeding 100MB using `.gitignore` or use Git LFS)*.

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Click **New app**.
   - Select your repository `VisionCaptionAI`, branch `main`, and main file `app.py`.
   - Click **Deploy**.

---

## Option 4: Instant Public Demo Link from your Mac (Cloudflare / ngrok)

If you are presenting your project live in a viva or college demonstration, you can run the app locally on your MacBook Air (taking advantage of Apple Silicon MPS acceleration) and generate a secure public HTTPS URL in seconds.

### Method A: Cloudflare Tunnel (Free, No Sign-Up Needed)
```bash
# 1. Install cloudflared via Homebrew (if not already installed)
brew install cloudflared

# 2. In Terminal 1, launch VisionCaption AI:
source .venv/bin/activate
streamlit run app.py

# 3. In Terminal 2, create a public tunnel:
cloudflared tunnel --url http://localhost:8501
```
Cloudflare will output a public URL like: `https://xxxx-xxxx.trycloudflare.com` that anyone can open on mobile or laptop.

### Method B: ngrok
```bash
# 1. In Terminal 1, launch VisionCaption AI:
source .venv/bin/activate
streamlit run app.py

# 2. In Terminal 2:
ngrok http 8501
```
Use the `https://xxxx.ngrok-free.app` forwarding address.

