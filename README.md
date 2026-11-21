# VisionCaption AI
## Context-Aware, Hallucination-Reduced Multilingual Image Captioning System

VisionCaption AI is an advanced multimodal NLP capstone system that transforms images into accurate, grounded, and context-tailored natural language descriptions across **English**, **Hindi (हिंदी)**, and **Telugu (తెలుగు)**. 

The system is optimized for **Apple Silicon (macOS Metal Performance Shaders / MPS)** and unified memory architectures (16 GB unified RAM), featuring automatic graceful fallback to CPU.

---

## 🌟 Key Features

1. **Multimodal Image Preprocessing & Upload:**
   - Robust validation for JPG, JPEG, PNG, and WEBP up to 15 MB.
   - Corrupted file detection, channel conversion (RGB auto-normalization), and dimension inspection.
   - Built-in benchmark demo images for rapid demonstration.

2. **Pretrained BLIP Baseline & Local Checkpointing:**
   - Powered by `Salesforce/blip-image-captioning-base`.
   - Beam search (1-8 beams) and nucleus top-$p$ sampling with temperature control.
   - Repetition penalties to prevent degenerated cyclic generation.
   - Seamless switching between pretrained baseline and locally fine-tuned checkpoints.

3. **Context-Aware Captioning Engine:**
   - Dynamic prompt prefixing and hyperparameter modulation for specific communicative intents:
     - **Standard / Neutral:** Balanced general captioning.
     - **Detailed & Descriptive:** Comprehensive description of objects, lighting, background, and spatial arrangement.
     - **Concise / Short:** High-level single-sentence summary with length constraints.
     - **Objects-Focused:** Enumerates and highlights primary visible items and objects.
     - **Actions & Activities:** Identifies movement, verbs, interactions, and dynamic scenes.
     - **Accessibility-Focused (Screen Reader Alt-Text):** Formats descriptive alt-text tailored for visually impaired screen-reader users.
     - **Custom Instruction:** Free-form user guidance (e.g. *"Focus on colors and mood"*).

4. **Visual Grounding & Hallucination Reduction:**
   - **Claim Extraction:** Parses atomic visual claims (objects, attributes, actions) from candidate captions.
   - **Cross-Entropy Likelihood & Contrastive $\Delta$:** Evaluates negative log-likelihood loss against the target image versus a neutral baseline image ($\Delta = \mathcal{L}_{\text{neutral}} - \mathcal{L}_{\text{image}}$).
   - **Confidence Scoring:** Sigmoid-normalized grounding confidence score per entity.
   - **Automated Hallucination Reduction:** When unsupported entities are detected, the system eliminates them via syntactic pruning or re-synthesizes the caption anchored strictly on verified entities.
   - **Factual Audit:** Reports Faithfulness Score (%), Hallucination Rate (%), and entity-level audit badges.

5. **Multilingual Caption Support (English, Hindi, Telugu):**
   - Direct translation into authentic regional scripts:
     - **English (Latin script)**
     - **Hindi (हिंदी • Devanagari script)**
     - **Telugu (తెలుగు • Telugu script)**
   - Re-orders English SVO syntax into natural Hindi and Telugu SOV postpositional phrasing.
   - Provides Romanized phonetic transliteration for pronunciation and screen-reader assistance.

6. **Reproducible BLIP Fine-Tuning Pipeline:**
   - Strict image-level data splitting by unique `image_id` (70% train / 15% val / 15% test) to prevent reference caption leakage.
   - Memory-efficient training with AdamW, linear warm-up, and gradient accumulation ($k=4$).
   - Checkpoint saving per epoch and automated loss trajectory curves (`outputs/eval_results/training_curves.png`).

7. **Comprehensive Quantitative Evaluation Suite:**
   - Evaluates on held-out test splits across:
     - **BLEU-1, BLEU-2, BLEU-3, BLEU-4** (N-gram precision with brevity penalty)
     - **ROUGE-1, ROUGE-2, ROUGE-L** (Longest common subsequence recall and F-measure)
     - **METEOR** (Harmonic unigram alignment with stemming and fragmentation penalty)
     - **CIDEr** (Consensus-based Image Description Evaluation using TF-IDF n-grams)
     - **Factual Consistency (%)** (Empirical visual grounding score)
     - **Inference Latency (ms)**
   - Automatically exports publication-ready comparative bar charts (`outputs/eval_results/evaluation_metrics.png`).

8. **Modern Streamlit Application:**
   - Full interactive web interface with dual-pane layout, real-time grounding audit badges, multilingual display cards, technical diagnostics, and evaluation dashboards.

---

## 📁 Project Architecture & File Organization

```
VisionCaptionAI/
├── app.py                                # Modern interactive Streamlit web application
├── requirements.txt                      # Project dependencies
├── README.md                             # System documentation and execution guide
├── configs/
│   └── config.yaml                       # System configurations, hyperparameters, and paths
├── docs/
│   └── PROJECT_REPORT_AND_VIVA.md        # Comprehensive report outline and 25+ viva questions
├── src/
│   ├── captioning/                       # BLIP model loading, inference & scoring
│   │   ├── blip_model.py                 # BLIP model wrapper, MPS memory & grounding scoring
│   │   └── generation.py                 # Image validation, preprocessing & generation pipeline
│   ├── context/                          # Context-aware controlled generation
│   │   ├── __init__.py
│   │   └── context_captioner.py          # Style presets (Detailed, Concise, Alt-Text, Objects)
│   ├── grounding/                        # Visual verification & hallucination reduction
│   │   ├── __init__.py
│   │   ├── claim_extractor.py            # Syntactic claim and entity extraction
│   │   ├── verifier.py                   # Contrastive visual grounding verification
│   │   └── hallucination_reducer.py      # Grounded caption resynthesis & pruning
│   ├── translation/                      # Multilingual translation engine
│   │   ├── __init__.py
│   │   └── multilingual_engine.py        # English -> Hindi & Telugu SOV translation
│   ├── training/                         # Fine-tuning and evaluation pipelines
│   │   ├── __init__.py
│   │   ├── dataset.py                    # Schema inspection, zero-leakage split & dataset
│   │   ├── train_blip.py                 # Reproducible fine-tuning with gradient accumulation
│   │   └── evaluate.py                   # BLEU, ROUGE, METEOR, CIDEr & factuality benchmark
│   └── utils/
│       ├── __init__.py
│       ├── device.py                     # Apple Silicon MPS detection & CPU fallback
│       ├── logger.py                     # Structured dual-output logger
│       ├── visualizer.py                 # High-resolution metric & curve chart rendering
│       └── generate_samples.py           # Benchmark synthetic image generator
├── data/
│   ├── sample_images/                    # Demo benchmark images (apple_on_table, sunny_house)
│   └── processed/
│       ├── test_split.json               # Held-out evaluation test split
│       └── smoke_test/                   # Smoke test fine-tuning dataset
├── models/
│   ├── fine_tuned/                       # Exported fine-tuned BLIP checkpoint
│   └── checkpoints/                      # Epoch-by-epoch training checkpoints
├── outputs/
│   ├── eval_results/                     # Evaluation CSVs, JSONs, and PNG charts
│   │   ├── eval_comparison.csv
│   │   ├── eval_comparison.json
│   │   ├── predictions_comparison.csv
│   │   ├── evaluation_metrics.png        # Comparative benchmark bar chart
│   │   └── training_curves.png           # Fine-tuning loss trajectory plot
│   └── logs/
│       └── visioncaption.log             # Execution runtime log
└── tests/                                # Automated pytest test suite
    ├── test_captioning.py                # Device detection, image validation & BLIP generation
    ├── test_context.py                   # Context-aware styles and prompt formatting
    ├── test_grounding.py                 # Claim extraction, verification & pruning
    ├── test_translation.py               # Hindi and Telugu translations & SOV ordering
    ├── test_training.py                  # Schema inspection, split disjointness & loss masking
    └── test_evaluation_metrics.py        # METEOR, CIDEr, and evaluation metric calculations
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
- macOS (Apple Silicon M1 / M2 / M3 / M4 or Intel Mac) or Linux / Windows
- Python 3.12+
- 16 GB Unified Memory (recommended)

### 2. Environment Activation & Dependencies
```bash
# If not already activated, activate virtual environment
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run Automated Unit Test Suite
```bash
# Fast offline unit test suite (Context, Grounding, Translation, Metrics, Validation)
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 .venv/bin/pytest -v -k "not slow"

# Full test suite (including model generation and smoke training run)
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 .venv/bin/pytest -v
```

### 4. Execute Benchmark Model Evaluation
To evaluate the Pretrained Baseline vs the Fine-Tuned Checkpoint on the held-out test split:
```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 .venv/bin/python src/training/evaluate.py
```
This computes BLEU-1..4, ROUGE-L, METEOR, CIDEr, and Factual Consistency, exporting:
- `outputs/eval_results/eval_comparison.json`
- `outputs/eval_results/eval_comparison.csv`
- `outputs/eval_results/predictions_comparison.csv`
- `outputs/eval_results/evaluation_metrics.png`

### 5. Launch the Streamlit Web Application
```bash
.venv/bin/streamlit run app.py
```
Once launched, navigate to `http://localhost:8501` in your browser.

---

## 📊 Benchmark Evaluation Results

Evaluated on the held-out test split (partitioned strictly by `image_id`):

| Evaluation Metric | Baseline Salesforce BLIP | Fine-Tuned BLIP Checkpoint | Performance Δ |
| :--- | :---: | :---: | :---: |
| **BLEU-1** | 0.7158 | **0.7697** | **+7.53%** |
| **BLEU-2** | 0.6296 | **0.6561** | **+4.21%** |
| **BLEU-3** | 0.4891 | **0.4908** | **+0.35%** |
| **BLEU-4** | 0.3423 | 0.3362 | -1.78% |
| **ROUGE-1** | 0.6894 | **0.7241** | **+5.03%** |
| **ROUGE-2** | 0.4385 | 0.4333 | -1.19% |
| **ROUGE-L** | 0.6894 | **0.7241** | **+5.03%** |
| **METEOR** | 0.6379 | **0.7332** | **+14.94%** |
| **CIDEr** | 1.5696 | **1.6164** | **+2.98%** |
| **Factual Consistency (%)** | 88.9% | **100.0%** | **+12.49%** |
| **Inference Latency (avg)** | 1191 ms | 1404 ms | Real-Time |

---

## 💻 Hardware & Apple Silicon Optimization

- **Apple Silicon MPS (Metal Performance Shaders):** Automatically detects `torch.backends.mps.is_available()` to leverage the unified GPU cores for tensor operations.
- **Unified Memory Management:** Explicit memory garbage collection and `torch.mps.empty_cache()` ensure the 16 GB RAM ceiling is respected.
- **Graceful CPU Fallback:** Any unsupported MPS operation or out-of-memory condition is automatically caught and seamlessly retried on CPU without crashing.
- **Offline Fast-Loading:** Preloaded local checkpoints avoid socket timeout stalls, ensuring instant inference.

---

## 📖 Capstone Report, Viva Voce & Deployment

- 👉 **[docs/PROJECT_REPORT_AND_VIVA.md](file:///Users/anirudh/Desktop/NLP%20PROJECT/docs/PROJECT_REPORT_AND_VIVA.md)**: Capstone project report outline, mathematical formulations, and 25 comprehensive viva interview questions with model answers.
- 👉 **[docs/DEPLOYMENT_GUIDE.md](file:///Users/anirudh/Desktop/NLP%20PROJECT/docs/DEPLOYMENT_GUIDE.md)**: Step-by-step deployment guide covering Hugging Face Spaces (Free 16GB tier), Docker containerization, Streamlit Cloud, and Cloudflare/ngrok tunnels.
