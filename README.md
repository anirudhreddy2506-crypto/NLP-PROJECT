# VisionCaption AI
## Context-Aware, Hallucination-Reduced Multilingual Image Captioning

VisionCaption AI is an advanced multimodal NLP capstone system that transforms images into accurate, grounded, and context-tailored natural language descriptions across English, Hindi, and Telugu. 

The system is optimized for Apple Silicon (macOS Metal Performance Shaders / MPS) and unified memory architectures, with automatic fallback to CPU.

---

## 🚀 Project Milestones Roadmap

- [x] **Milestone 1:** Python 3.12 environment setup, modular project structure, configuration system, and Apple Silicon MPS device detection with graceful CPU fallback.
- [x] **Milestone 2:** Pretrained `Salesforce/blip-image-captioning-base` baseline, robust image preprocessing (JPEG/PNG/WEBP, size checks), latency profiling, and modern Streamlit web interface.
- [ ] **Milestone 3:** Reproducible BLIP fine-tuning pipeline on MS COCO / Flickr8k splits (preventing leakage by image ID), checkpointing, and gradient accumulation.
- [ ] **Milestone 4:** Context-aware controlled generation pipeline (detail, accessibility, actions, child-friendly).
- [ ] **Milestone 5:** Visual grounding & hallucination reduction pipeline with factual claim extraction and verification.
- [ ] **Milestone 6:** Multilingual caption generation across English, Telugu, and Hindi via neural machine translation.
- [ ] **Milestone 7:** Quantitative evaluation suite (BLEU, METEOR, CIDEr, factual consistency, unsupported claim rate, latency) & research dashboard.

---

## 🛠️ Architecture & Module Organization

```
VisionCaptionAI/
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # System dependencies
├── configs/
│   └── config.yaml             # System hyperparameters, devices, and paths
├── src/
│   ├── captioning/             # BLIP model loading and inference pipeline
│   │   ├── blip_model.py       # BLIP model wrapper with MPS memory management
│   │   └── generation.py       # Image validation and caption generation
│   ├── context/                # Context-aware refinement (Milestone 4)
│   ├── grounding/              # Claim extraction and visual verification (Milestone 5)
│   ├── translation/            # Multilingual IndicTrans2 integration (Milestone 6)
│   ├── training/               # Fine-tuning and evaluation pipelines (Milestone 3)
│   └── utils/
│       ├── device.py           # Apple MPS detection & CPU fallback
│       ├── logger.py           # Structured dual-output logger
│       └── generate_samples.py # Benchmark synthetic image generator
├── data/
│   └── sample_images/          # Built-in demo benchmark images
├── models/                     # Model weights & fine-tuned checkpoints
├── outputs/                    # Evaluation logs, predictions, and metrics
└── tests/
    └── test_captioning.py      # Automated unit test suite
```

---

## 💻 Environment Setup & Quick Start

### 1. Prerequisites
- macOS (Apple Silicon M1/M2/M3/M4 or Intel)
- Python 3.12
- 16 GB Unified Memory (recommended)

### 2. Create and Activate Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```bash
# Run validation and device tests
.venv/bin/pytest -v -k "not slow"

# Run full test suite including BLIP model generation
.venv/bin/pytest -v
```

### 5. Launch the Streamlit Web Application
```bash
.venv/bin/streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Hardware Acceleration & Diagnostics

VisionCaption AI natively queries `torch.backends.mps.is_available()` to offload vision-language tensor computations to Apple Silicon GPUs. If an MPS kernel limitation or memory pressure arises, the pipeline catches the exception and falls back to CPU without crashing the application.
