import streamlit as st
import os
import io
from datetime import datetime
from dotenv import load_dotenv
from TextToImage import (
    ImageGenerator, MODELS, MODEL_INFO,
    validate_api_token, get_example_prompts
)
from PIL import Image

# Load .env
load_dotenv()

# ----------------- Page config -----------------
st.set_page_config(page_title="AI Image Studio — Ultra", page_icon="🎨", layout="wide")

# ----------------- API MODE (token update via URL) -----------------
query_params = st.experimental_get_query_params()

if "token" in query_params:
    new_token = query_params["token"][0]
    st.session_state["hf_token"] = new_token

    st.write({
        "status": "success",
        "message": "HuggingFace token updated",
        "token": new_token
    })

    st.stop()  # return JSON only, stop full UI

# ----------------- Styles -----------------
st.markdown("""
<style>
body { font-family: Inter, sans-serif; background: linear-gradient(135deg,#f3f8ff 0%, #ffffff 100%);} 
.header {background:linear-gradient(90deg,#6e00ff,#00a3ff); color:white; padding:26px; border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,0.12)}
.header h1 {margin:0; font-size:30px}
.card {background:rgba(255,255,255,0.85); border-radius:12px; padding:14px; box-shadow:0 6px 18px rgba(10,10,10,0.06)}
.stButton > button {border-radius:10px; padding:10px 14px}
.result {border-radius:12px; overflow:hidden}
.small-muted {color:#666; font-size:13px}
</style>
""", unsafe_allow_html=True)

# ----------------- Utilities -----------------
def image_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def generate_filename(prompt, model_name, timestamp):
    clean_prompt = "".join(c for c in prompt[:40] if c.isalnum() or c == " ").strip().replace(" ", "_")
    model_clean = model_name.replace("/", "_")
    return f"{model_clean}_{timestamp}_{clean_prompt}.png"

# ----------------- Session -----------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("count", 0)
st.session_state.setdefault("hf_token", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
st.session_state.setdefault("prompt", "")
st.session_state.setdefault("example_prompt", "")

# ----------------- Callback -----------------
def set_example_prompt():
    st.session_state["prompt"] = st.session_state["example_prompt"]

# ----------------- Header -----------------
st.markdown(
    "<div class='header'><h1>AI Image Studio — Ultra</h1>"
    "<div class='small-muted'>Fast • Beautiful • Live token switching • Img2Img</div>"
    "</div>",
    unsafe_allow_html=True
)

# ----------------- Layout -----------------
left, right = st.columns([2.6, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Create an image from text")

    prompt_input = st.text_area(
        "Prompt", key="prompt", height=160,
        placeholder="A cinematic portrait of a warrior, rim light, dramatic"
    )

    with st.expander("Advanced options", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            neg_prompt = st.text_input("Negative prompt", placeholder="low quality, watermark, extra fingers")
            seed = st.number_input("Seed (0=random)", 0, 2**31-1, 0)

        with col2:
            steps = st.slider("Steps", 10, 150, 28)
            guidance = st.slider("Guidance scale", 1.0, 30.0, 7.5, 0.5)

        with col3:
            width = st.selectbox("Width", [512, 640, 768, 1024], 0)
            height = st.selectbox("Height", [512, 640, 768, 1024], 0)

    st.write("Optional: Upload an image (Img2Img)")
    init_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    g1, g2, g3 = st.columns([2, 1, 1])
    generate_btn = g1.button("✨ Generate")
    clear_btn = g2.button("Clear History")
    copy_btn = g3.button("Copy Prompt")

    if copy_btn:
        st.write("Use CTRL/CMD + C to copy the prompt.")

    if clear_btn:
        st.session_state["history"] = []
        st.session_state["count"] = 0

    st.markdown("</div>", unsafe_allow_html=True)

    if generate_btn:
        prompt = st.session_state["prompt"].strip()

        if not prompt:
            st.error("Please enter a prompt.")
        else:
            token = st.session_state["hf_token"]
            if not validate_api_token(token):
                st.error("Invalid HuggingFace token.")
            else:
                model_key = st.session_state.get("model", list(MODELS.keys())[0])
                model_id = MODELS.get(model_key)

                gen = ImageGenerator(token, output_dir="./outputs")

                with st.spinner("Generating..."):
                    success, message, pil_img = gen.generate_image(
                        prompt=prompt,
                        model_id=model_id,
                        negative_prompt=neg_prompt or None,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                        seed=seed if seed != 0 else None,
                        init_image=init_image
                    )

                if not success:
                    st.error(message)
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state["history"].insert(0, {
                        "img": pil_img,
                        "prompt": prompt,
                        "model": model_key,
                        "ts": ts
                    })

                    st.markdown("<div class='card result'>", unsafe_allow_html=True)
                    st.image(pil_img, use_column_width=True)
                    st.write(message)
                    st.download_button(
                        "⬇️ Download PNG",
                        image_to_bytes(pil_img),
                        file_name=generate_filename(prompt, model_key, ts),
                        mime="image/png"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Settings & Controls")

    st.text_input("HuggingFace API token", key="hf_token", type="password")

    st.markdown("---")
    st.write("Model Selection")
    model_selection = st.selectbox("Choose model", list(MODELS.keys()), key="model")

    model_id = MODELS[model_selection]
    info = MODEL_INFO.get(model_id, {})

    st.markdown(f"**Model:** {model_id}")
    st.markdown(
        f"**License:** {info.get('license','-')}  \n"
        f"**Status:** {info.get('status','-')}  \n"
        f"**Description:** {info.get('description','-')}"
    )

    st.markdown("---")
    st.subheader("Example Prompts")

    st.selectbox("Choose example", get_example_prompts(), key="example_prompt")
    st.button("Use example", on_click=set_example_prompt)

    st.markdown("---")
    st.caption("Images generated: " + str(len(st.session_state["history"])))
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Gallery -----------------
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("Your Gallery")

    cols = st.columns(3)
    for i, item in enumerate(st.session_state["history"]):
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(item["img"], use_column_width=True)
            st.caption(item["prompt"][:80] + "...")
            st.download_button(
                "Download",
                image_to_bytes(item["img"]),
                file_name=generate_filename(item["prompt"], item["model"], item["ts"])
            )
            st.markdown("</div>", unsafe_allow_html=True)
