import os
import cv2
import torch
import timm
import faiss
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from sentence_transformers import SentenceTransformer

# ===============================
# 1. DEVICE
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# ===============================
# 2. IMAGE CLASSIFICATION MODEL
# ===============================

CLASS_MAP = {
    0: "Calculus (CaS)",
    1: "Tongue Coating (CoS)",
    2: "Gum Disease (Gingivitis)",
    3: "Mouth Cancer (MC)",
    4: "Oral Candidiasis (OC)",
    5: "Oral Lichen Planus (OLP)",
    6: "Other (OT)"
}

def load_image_model():
    model = timm.create_model(
        "inception_resnet_v2",
        pretrained=False,
        num_classes=7
    )
    model.load_state_dict(
        torch.load("inceptionresnetv2_teeth.pth", map_location=device),
        strict=False
    )
    model.to(device)
    model.eval()
    return model

img_model = load_image_model()
print("✅ Image model load ho gaya")

# ===============================
# 3. RAG SETUP
# ===============================

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("oral_disease.index")
final_chunks = np.load("final_chunks.npy", allow_pickle=True).tolist()
print("✅ RAG system ready")

# ===============================
# 4. MAIN ANALYZE FUNCTION
# ===============================

def analyze_input(file):
    if file is None:
        return "⚠️ Koi file upload nahi ki.", None, ""

    file_path = file
    is_video = file_path.lower().endswith((".mp4", ".avi", ".mov"))

    if is_video:
        cap = cv2.VideoCapture(file_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return "❌ Video frame read nahi hua.", None, ""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        img = Image.open(file_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_t = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = img_model(img_t)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    disease = CLASS_MAP[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # ── Dark Professional Chart ──
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    colors = ["#1e3a5f"] * 7
    colors[pred_idx] = "#00c9a7"
    labels = [c.split("(")[0].strip() for c in CLASS_MAP.values()]

    bars = ax.barh(labels, probs, color=colors, height=0.55, edgecolor="none")

    for i, (bar, p) in enumerate(zip(bars, probs)):
        ax.text(p + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", ha="left",
                color="#00c9a7" if i == pred_idx else "#4a6a8a",
                fontsize=9,
                fontweight="bold" if i == pred_idx else "normal")

    ax.set_xlim(0, 1.2)
    ax.set_title(f"Detected: {disease}", color="#00c9a7",
                 fontsize=12, fontweight="bold", pad=14, loc="left")
    ax.tick_params(axis="y", colors="#8899aa", labelsize=9)
    ax.xaxis.set_visible(False)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    plt.tight_layout(pad=1.5)
    chart_path = "analysis.png"
    plt.savefig(chart_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # ── RAG: Auto info for detected disease ──
    query = f"symptoms treatment medicine care of {disease}"
    q_emb = embed_model.encode([query]).astype("float32")
    _, I = index.search(q_emb, k=3)
    chunks = [final_chunks[i].strip() for i in I[0]]

    formatted_chunks = ""
    for idx, chunk in enumerate(chunks, 1):
        formatted_chunks += f"**{idx}.** {chunk}\n\n"

    # ── Diagnosis result card ──
    diagnosis_md = f"""
### ✅ Diagnosis Complete

| Field | Result |
|-------|--------|
| 🦷 **Disease** | {disease} |
| 📊 **Confidence** | {confidence:.1f}% |
"""

    # ── RAG info ──
    rag_md = f"""### 🩺 Medical Information — {disease}

---

{formatted_chunks}
---
*⚕️ Yeh information sirf educational purposes ke liye hai. Kisi doctor se zaroor milein.*
"""

    return diagnosis_md.strip(), chart_path, rag_md.strip()


# ===============================
# 5. CSS
# ===============================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #070e1a !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #c8dff5 !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 28px 40px !important;
}

.app-header {
    padding: 48px 0 32px;
    text-align: center;
    border-bottom: 1px solid #111e30;
    margin-bottom: 32px;
}
.app-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin-bottom: 10px;
}
.app-header h1 span { color: #00c9a7; }
.app-header p {
    color: #4a6a8a !important;
    font-size: 0.95rem !important;
}

.panel {
    background: #0c1a2e !important;
    border: 1px solid #112240 !important;
    border-radius: 18px !important;
    padding: 26px !important;
}

.section-label {
    color: #4a6a8a;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

label > span, .label-wrap span {
    color: #4a6a8a !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

[data-testid="file"], div[class*="upload"] {
    background: #070e1a !important;
    border: 2px dashed #112240 !important;
    border-radius: 12px !important;
    transition: border-color 0.25s !important;
    min-height: 160px !important;
}
[data-testid="file"]:hover { border-color: #00c9a7 !important; }

.gr-button-primary, button[variant="primary"] {
    background: linear-gradient(135deg, #00c9a7 0%, #0087b3 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 13px 30px !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(0,201,167,0.2) !important;
    width: 100% !important;
    margin-top: 14px !important;
}
.gr-button-primary:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

.gr-prose, .prose, div[data-testid="markdown"] {
    color: #c8dff5 !important;
    font-family: 'DM Sans', sans-serif !important;
    line-height: 1.75 !important;
}
div[data-testid="markdown"] h3 {
    color: #00c9a7 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #112240 !important;
    padding-bottom: 8px !important;
    margin-bottom: 14px !important;
}
div[data-testid="markdown"] table { width: 100% !important; border-collapse: collapse !important; margin: 10px 0 !important; }
div[data-testid="markdown"] td { padding: 9px 14px !important; border: 1px solid #112240 !important; color: #c8dff5 !important; font-size: 0.92rem !important; }
div[data-testid="markdown"] blockquote {
    border-left: 3px solid #00c9a7 !important;
    background: #070e1a !important;
    padding: 10px 16px !important;
    border-radius: 0 8px 8px 0 !important;
    color: #4a6a8a !important;
    font-style: italic !important;
    margin: 14px 0 !important;
}
div[data-testid="markdown"] p { color: #e2f0ff !important; font-size: 0.95rem !important; line-height: 1.8 !important; }
div[data-testid="markdown"] strong { color: #ffffff !important; }
div[data-testid="markdown"] em { color: #7a9abc !important; font-size: 0.82rem !important; }
div[data-testid="markdown"] hr { border: none !important; border-top: 1px solid #112240 !important; margin: 16px 0 !important; }

img { border-radius: 12px !important; }

/* Full width fixes */
.app, #app, main, .main, .wrap, .contain {
    max-width: 100% !important;
    width: 100% !important;
}
.gr-row { width: 100% !important; }
.gr-column { min-width: 0 !important; }

.divider { border: none; border-top: 1px solid #111e30; margin: 28px 0; }

.footer {
    text-align: center;
    color: #1e3a5f !important;
    font-size: 0.78rem !important;
    padding: 24px 0 8px;
    letter-spacing: 0.04em;
}
"""

# ===============================
# 6. PREVIEW FUNCTION
# ===============================

def show_preview(file):
    """File upload hote hi preview dikhao"""
    if file is None:
        return gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    is_video = file.lower().endswith((".mp4", ".avi", ".mov"))
    if is_video:
        return gr.update(visible=False, value=None), gr.update(visible=True, value=file)
    else:
        return gr.update(visible=True, value=file), gr.update(visible=False, value=None)

# ===============================
# 7. GRADIO UI
# ===============================

with gr.Blocks(css=CSS, title="Mouth_and_Oral_Diseases_(MOD)Scan AI") as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>🦷 Oral<span>Scan</span> AI</h1>
        <p>AI-powered Oral Disease Detection &amp; Medical Information System</p>
    </div>
    """)

    # ── Row 1: Upload | Preview | Diagnosis+Chart ──
    with gr.Row(equal_height=True):

        # Column 1 — Upload
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<div class='section-label'>Upload Image or Video</div>")
            file_input = gr.File(
                label="",
                file_types=["image", ".mp4", ".avi", ".mov"],
            )
            analyze_btn = gr.Button("🔍  Run Analysis", variant="primary")

        # Column 2 — Preview
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<div class='section-label'>File Preview</div>")
            image_preview = gr.Image(
                label="", show_label=False, visible=False
            )
            video_preview = gr.Video(
                label="", show_label=False, visible=False
            )

        # Column 3 — Diagnosis + Chart
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML("<div class='section-label'>Diagnosis Result</div>")
            diagnosis_output = gr.Markdown(
                value="*Upload a file and click **Run Analysis** to begin...*"
            )
            chart_output = gr.Image(label="Confidence Distribution", show_label=True)

    gr.HTML("<hr class='divider'>")

    # ── Row 2: Medical Info ──
    gr.HTML("<div class='section-label' style='margin-bottom:16px;'>📋 Medical Information</div>")
    rag_output = gr.Markdown(
        value="*Analysis ke baad yahan medical information automatically ayegi...*"
    )

    gr.HTML("""
    <div class="footer">
        ⚕️ For educational purposes only &nbsp;·&nbsp; Always consult a qualified medical professional
    </div>
    """)

    # ── Events ──

    # File upload hote hi preview dikhao
    file_input.change(
        fn=show_preview,
        inputs=[file_input],
        outputs=[image_preview, video_preview]
    )

    # Analyze button — sab kuch ek saath
    analyze_btn.click(
        fn=analyze_input,
        inputs=[file_input],
        outputs=[diagnosis_output, chart_output, rag_output]
    )

if __name__ == "__main__":
    demo.launch(width="100%")