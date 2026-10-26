"""
VisionCaption AI - Streamlit Web Application
Context-Aware, Hallucination-Reduced Multilingual Image Captioning
"""

import os
import sys
import time
import json
from typing import Optional
from PIL import Image
import streamlit as st
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.device import get_optimal_device, get_device_info
from src.utils.logger import setup_logger
from src.captioning.blip_model import BLIPCaptioner
from src.captioning.generation import validate_and_load_image, generate_caption

logger = setup_logger("visioncaption.app")

# Page Configuration
st.set_page_config(
    page_title="VisionCaption AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #E3F2FD;
        color: #1565C0;
        margin-right: 0.4rem;
    }
    .caption-card {
        padding: 1.25rem;
        border-radius: 0.75rem;
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .caption-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1A237E;
        line-height: 1.5;
    }
    .stat-box {
        background-color: #FFFFFF;
        border: 1px solid #ECEFF1;
        border-radius: 0.5rem;
        padding: 0.75rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading BLIP Captioning Model into memory...")
def load_captioner_model(model_name: str, device_preference: str) -> BLIPCaptioner:
    """
    Loads and caches the BLIP captioning model on the optimal device (Apple MPS or CPU).
    """
    logger.info("Initializing cached BLIPCaptioner (%s on %s)...", model_name, device_preference)
    return BLIPCaptioner(model_name_or_path=model_name, device_preference=device_preference)


def load_config() -> dict:
    """Load config.yaml with fallback defaults."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "system": {"device": "auto", "allow_mps_fallback": True},
        "models": {"baseline_captioner": "Salesforce/blip-image-captioning-base"},
        "generation": {"num_beams": 4, "max_new_tokens": 50, "min_length": 5, "repetition_penalty": 1.2},
    }


config = load_config()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### ⚙️ System & Controls")
    
    device_opt = st.selectbox(
        "Compute Device",
        options=["auto", "mps", "cpu"],
        index=0,
        help="Select 'auto' for automatic Apple Silicon MPS detection with CPU fallback.",
    )
    
    device_obj = get_optimal_device(preference=device_opt)
    dev_info = get_device_info(device_obj)
    
    st.info(
        f"**Active Hardware:** `{device_obj.type.upper()}`\n\n"
        + (f"• Apple MPS Acceleration: **Active**\n" if dev_info.get("is_mps") else "• Hardware: Standard CPU\n")
        + f"• Memory Strategy: Unified 16GB"
    )

    st.markdown("---")
    st.markdown("### 🤖 Active Model Checkpoint")
    
    available_models = [
        ("Pretrained Baseline", config["models"]["baseline_captioner"]),
    ]
    fine_tuned_path = os.path.join(PROJECT_ROOT, config["models"].get("fine_tuned_dir", "models/fine_tuned"))
    if os.path.exists(fine_tuned_path):
        available_models.append(("Fine-Tuned Checkpoint", fine_tuned_path))

    selected_model_idx = st.selectbox(
        "Model Selection",
        range(len(available_models)),
        format_func=lambda idx: available_models[idx][0],
        help="Switch between the original pretrained baseline and your locally fine-tuned model checkpoint.",
    )
    active_model_name = available_models[selected_model_idx][1]
    active_model_label = available_models[selected_model_idx][0]

    st.markdown("---")
    st.markdown("### 🎛️ Generation Hyperparameters")

    decoding_strategy = st.radio(
        "Decoding Strategy",
        options=["Beam Search", "Nucleus Sampling"],
        index=0,
        help="Beam search yields deterministic, high-probability descriptions. Nucleus sampling allows stylistic variety.",
    )

    if decoding_strategy == "Beam Search":
        num_beams = st.slider("Beam Count (num_beams)", min_value=1, max_value=8, value=4, step=1)
        do_sample = False
        temperature = 0.7
        top_p = 0.9
    else:
        num_beams = 1
        do_sample = True
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.05)
        top_p = st.slider("Top-p (Nucleus)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

    max_new_tokens = st.slider("Max New Tokens", min_value=10, max_value=100, value=50, step=5)
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

    st.markdown("---")
    st.markdown("### 📁 Model Pipeline Status")
    st.markdown("✅ **M1: Hardware & Device Setup (MPS)**")
    st.markdown("✅ **M2: Pretrained BLIP Baseline**")
    st.markdown("✅ **M3: BLIP Fine-Tuning & Evaluation**")
    st.markdown("⏳ *M4: Context-Aware Refinement*")
    st.markdown("⏳ *M5: Visual Grounding & Factuality*")
    st.markdown("⏳ *M6: IndicTrans2 (Hindi & Telugu)*")
    st.markdown("⏳ *M7: Final Viva & Demo Pack*")


# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-title">VisionCaption AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Multimodal Image Captioning with Apple MPS Acceleration, Grounding, and Multilingual Output</div>',
    unsafe_allow_html=True,
)

col_badges = st.columns([1.5, 1, 1, 3.5])
with col_badges[0]:
    st.markdown(f'<span class="badge">Model: {active_model_label}</span>', unsafe_allow_html=True)
with col_badges[1]:
    st.markdown(f'<span class="badge">Device: {device_obj.type.upper()}</span>', unsafe_allow_html=True)
with col_badges[2]:
    st.markdown('<span class="badge">PyTorch 2.x</span>', unsafe_allow_html=True)

st.markdown("")

# Two-column layout: Input & Output
col_left, col_right = st.columns([1.1, 1.3], gap="large")

with col_left:
    st.subheader("1. Input Image & Settings")
    
    input_source = st.radio(
        "Image Source",
        options=["Upload Custom Image", "Choose Benchmark Sample"],
        horizontal=True,
    )
    
    pil_image = None
    img_meta = {}
    img_caption_label = ""
    
    if input_source == "Upload Custom Image":
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Maximum file size 15MB.",
        )
        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                pil_image, img_meta = validate_and_load_image(raw_bytes)
                img_caption_label = f"Uploaded: {uploaded_file.name} ({img_meta['width']}x{img_meta['height']} px, {img_meta['format']})"
            except Exception as e:
                st.error(f"Image validation error: {e}")
    else:
        sample_options = [
            os.path.join(PROJECT_ROOT, "data/sample_images/sunny_house_landscape.jpg"),
            os.path.join(PROJECT_ROOT, "data/sample_images/apple_on_table.png"),
        ]
        sample_choice = st.selectbox(
            "Select Benchmark Sample Image",
            options=sample_options,
            format_func=lambda x: "Sunny House & Tree (Landscape)" if "sunny" in x else "Apple on Table (Still Life)",
        )
        if os.path.exists(sample_choice):
            pil_image, img_meta = validate_and_load_image(sample_choice)
            img_caption_label = f"Sample: {os.path.basename(sample_choice)} ({img_meta['width']}x{img_meta['height']} px, {img_meta['format']})"

    # Caption style selector (Feature B)
    caption_style = st.selectbox(
        "Caption Focus & Style Preset",
        options=[
            "Standard / Neutral",
            "Detailed & Descriptive",
            "Concise / Short",
            "Actions & Activities",
            "Accessibility-Focused",
            "Custom Instruction",
        ],
        index=0,
    )
    
    context_instruction = ""
    if caption_style == "Custom Instruction":
        context_instruction = st.text_input(
            "Custom Style / Context Instruction",
            placeholder="e.g. Describe colors and lighting, or use simple words for a child",
        )
    elif caption_style == "Detailed & Descriptive":
        context_instruction = "Describe the scene in rich detail."
    elif caption_style == "Concise / Short":
        context_instruction = "Generate a short, concise description."
    elif caption_style == "Actions & Activities":
        context_instruction = "Focus primarily on actions and interactions in the image."
    elif caption_style == "Accessibility-Focused":
        context_instruction = "Generate an accessibility-focused description for screen readers."
    
    # Target Language selector (Feature D interface)
    target_language = st.selectbox(
        "Target Output Language",
        options=["English", "Telugu", "Hindi"],
        index=0,
        help="Select output language. English is generated directly by BLIP. Multilingual neural translation will be active in Milestone 6.",
    )
    
    generate_btn = st.button("🚀 Generate Caption", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Visual Preview & Generated Caption")
    
    if pil_image is not None:
        try:
            st.image(
                pil_image,
                caption=img_caption_label,
                use_container_width=True,
            )
            
            if generate_btn:
                with st.spinner(f"Executing inference on image using {active_model_label}..."):
                    captioner = load_captioner_model(
                        model_name=active_model_name,
                        device_preference=device_opt,
                    )
                    
                    # Prefix context prompt if provided
                    prompt_prefix = None
                    if context_instruction and context_instruction.strip():
                        prompt_prefix = f"a photo of"
                    
                    result = generate_caption(
                        captioner=captioner,
                        image=pil_image,
                        text_prompt=prompt_prefix,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        min_length=config["generation"].get("min_length", 5),
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    
                    if result["success"]:
                        st.markdown(
                            f"""
                            <div class="caption-card">
                                <div style="font-size: 0.85rem; color: #666; font-weight: bold; text-transform: uppercase; margin-bottom: 0.4rem;">
                                    Generated Caption ({active_model_label} • {target_language})
                                </div>
                                <div class="caption-text">
                                    "{result['caption']}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        
                        # Metrics row
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric(label="Inference Latency", value=f"{result['latency_ms']} ms")
                        with m2:
                            st.metric(label="Compute Device", value=result["device"].upper())
                        with m3:
                            st.metric(label="Active Model", value=active_model_label)
                        
                        if target_language != "English":
                            st.info(
                                f"ℹ️ **Note on Multilingual Output ({target_language}):** Direct BLIP generates English captions. Neural translation via IndicTrans2 is scheduled for Milestone 6."
                            )

                        with st.expander("🔍 Technical Diagnostics & Generation Details"):
                            st.json({
                                "model_checkpoint": result["model"],
                                "device": result["device"],
                                "latency_seconds": result["latency_seconds"],
                                "image_dimensions": [img_meta["width"], img_meta["height"]],
                                "image_format": img_meta["format"],
                                "num_beams": num_beams,
                                "max_new_tokens": max_new_tokens,
                                "repetition_penalty": repetition_penalty,
                                "context_instruction": context_instruction or "(none)",
                            })
                    else:
                        st.error(f"Generation error: {result['error']}")
                        
        except Exception as e:
            st.error(f"Image validation error: {e}")
    else:
        st.info("👈 Please upload an image or pick a benchmark sample from the left panel to begin caption generation.")

# ----------------- EVALUATION & RESEARCH DASHBOARD -----------------
st.markdown("---")
with st.expander("📊 Milestone 3 Research & Evaluation Dashboard (Baseline vs Fine-Tuned)", expanded=False):
    st.markdown("### Quantitative Model Comparison on Held-Out Test Set")
    st.markdown(
        "Evaluation results comparing the **Salesforce BLIP Baseline** against the **Fine-Tuned BLIP Checkpoint** "
        "on the held-out test split (partitioned strictly by image ID to avoid data leakage)."
    )

    eval_json_path = os.path.join(PROJECT_ROOT, "outputs/eval_results/eval_comparison.json")
    pred_csv_path = os.path.join(PROJECT_ROOT, "outputs/eval_results/predictions_comparison.csv")

    if os.path.exists(eval_json_path):
        import pandas as pd
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        df_metrics = pd.DataFrame(eval_data.get("comparison_table", []))
        st.dataframe(df_metrics, use_container_width=True)

        st.markdown("#### Sample Predictions Comparison")
        if os.path.exists(pred_csv_path):
            df_preds = pd.read_csv(pred_csv_path)
            st.dataframe(df_preds, use_container_width=True)

        st.caption(
            "**Note on Metrics:** BLEU and ROUGE evaluate lexical and n-gram overlap against human references. "
            "Higher scores denote improved stylistic alignment with the target domain. "
            "Visual factuality checks and hallucination reduction are added in Milestone 5."
        )
    else:
        st.info("Run `python src/training/evaluate.py` to generate the evaluation comparison table.")
