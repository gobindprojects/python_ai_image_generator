import streamlit as st
import os
import io
from datetime import datetime
from dotenv import load_dotenv
from TextToImage import ImageGenerator, MODELS, MODEL_INFO, validate_api_token, get_example_prompts

# Load environment variables
load_dotenv()

# App Config
st.set_page_config(
    page_title="AI Image Studio",
    page_icon="🎨",
    layout="wide",
)

# ------------------------ CUSTOM DESIGN ------------------------
st.markdown("""
<style>

/* GLOBAL -----------------------------------------------------*/
body {
    background: linear-gradient(135deg, #e9efff 0%, #ffffff 100%);
    font-family: 'Inter', sans-serif;
}

/* HEADER -----------------------------------------------------*/
.gradient-header {
    padding: 2.7rem 1rem;
    background: linear-gradient(135deg, #5a00ff 0%, #0099ff 100%);
    color: white;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.20);
    animation: fadeDown 0.6s ease;
}

.gradient-header h1 {
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -1px;
}

@keyframes fadeDown {
    from {opacity:0; transform: translateY(-25px);}
    to   {opacity:1; transform: translateY(0);}
}

/* SIDEBAR -----------------------------------------------------*/
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(18px);
    border-right: 1px solid #e5e5e5;
}

.stSidebar > div {
    padding-top: 30px;
}

/* INPUT BOX ---------------------------------------------------*/
textarea {
    border-radius: 14px !important;
}

/* BUTTONS ------------------------------------------------------*/
.stButton > button {
    border-radius: 12px;
    padding: 0.75rem;
    background: linear-gradient(135deg, #6a00ff, #007bff);
    border: none;
    color: white;
    font-size: 16px;
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

/* RESULT CARD --------------------------------------------------*/
.result-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(14px);
    padding: 1.4rem;
    border-radius: 18px;
    box-shadow: 0 6px 22px rgba(0,0,0,0.08);
    animation: fadeIn 0.4s ease;
}

@keyframes fadeIn {
    from {opacity:0;}
    to   {opacity:1;}
}

/* IMAGE HISTORY CARD ------------------------------------------*/
.history-card {
    background: white;
    padding: 1rem;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.09);
    transition: 0.2s;
}

.history-card:hover {
    transform: translateY(-4px);
}

/* FOOTER -------------------------------------------------------*/
.footer {
    padding: 35px;
    text-align: center;
    color: #666;
}

</style>
""", unsafe_allow_html=True)

# ------------------------ UTILITIES ------------------------
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'count' not in st.session_state:
        st.session_state.count = 0

def image_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()

def generate_filename(prompt, model_name, timestamp):
    clean_prompt = "".join(c for c in prompt[:40] if c.isalnum() or c == " ").strip().replace(" ", "_")
    model_clean = model_name.replace(" ", "_")
    return f"{model_clean}_{timestamp}_{clean_prompt}.png"


# ------------------------ MAIN APP ------------------------
def main():

    initialize_session_state()

    # Header
    st.markdown("""
    <div class="gradient-header">
        <h1>AI Image Studio 🎨</h1>
        <p>Premium text-to-image generator powered by advanced AI models.</p>
    </div>
    """, unsafe_allow_html=True)

    # Layout columns
    left, right = st.columns([2.4, 1])

    # ---------------------------------------------------------
    # LEFT SIDE: PROMPT + GENERATION
    # ---------------------------------------------------------
    with left:

        st.subheader("📝 Describe your image")

        prompt = st.text_area(
            "Enter prompt",
            height=140,
            placeholder="Example: Ultra-realistic portrait of a cyberpunk samurai, neon lights, dramatic lighting"
        )

        g1, g2 = st.columns([3,1])

        generate_btn = g1.button("✨ Generate Image", use_container_width=True)
        clear_btn    = g2.button("❌ Clear", use_container_width=True)

        if clear_btn:
            st.session_state.history = []
            st.rerun()

        # Generation logic
        if generate_btn:
            if not prompt.strip():
                st.error("Please enter a prompt.")
            else:
                st.info("Generating image… please wait")

                api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
                if not validate_api_token(api_token):
                    st.error("Invalid HuggingFace API token")
                    return

                gen = ImageGenerator(api_token, "./outputs")

                selected_model = list(MODELS.keys())[0]  # default model
                model_id = MODELS[selected_model]

                success, msg, image = gen.generate_image(prompt, model_id)

                if not success:
                    st.error(msg)
                else:
                    st.success("Image created!")

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    st.session_state.history.insert(0, {
                        "img": image,
                        "prompt": prompt,
                        "model": selected_model,
                        "ts": timestamp
                    })

                    st.session_state.count += 1

                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Download button
                    st.download_button(
                        "⬇️ Download Image",
                        data=image_to_bytes(image),
                        file_name=generate_filename(prompt, selected_model, timestamp),
                        mime="image/png",
                        use_container_width=True,
                        key=f"main_dl_{timestamp}"
                    )

    # ---------------------------------------------------------
    # RIGHT SIDE: SETTINGS
    # ---------------------------------------------------------
    with right:

        st.subheader("⚙️ Settings")

        api_token = st.text_input(
            "🔑 HuggingFace API Token",
            type="password",
            value=os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        )

        st.write("AI Model")
        model = st.selectbox("Select Model", list(MODELS.keys()))

        st.markdown("---")

        st.write("📌 Example Prompt")
        example = get_example_prompts()[0]  # Only 1 example
        st.info(example)

        if st.button("Use Example Prompt", use_container_width=True):
            st.session_state.temp_prompt = example
            st.rerun()

        if "temp_prompt" in st.session_state:
            prompt = st.session_state.temp_prompt
            del st.session_state.temp_prompt


    # ---------------------------------------------------------
    # IMAGE HISTORY
    # ---------------------------------------------------------
    if st.session_state.history:
        st.subheader("🖼️ Your Generated Images")

        cols = st.columns(2)

        for index, data in enumerate(st.session_state.history):

            with cols[index % 2]:
                st.markdown("<div class='history-card'>", unsafe_allow_html=True)

                st.image(data["img"], use_container_width=True)
                st.caption(f"Prompt: {data['prompt'][:70]}...")
                st.caption(f"Model: {data['model']}")

                st.download_button(
                    "⬇️ Download",
                    data=image_to_bytes(data["img"]),
                    file_name=generate_filename(data["prompt"], data["model"], data["ts"]),
                    mime="image/png",
                    use_container_width=True,
                    key=f"hist_dl_{data['ts']}"
                )

                st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        🚀 AI Image Studio • Crafted with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)


# Run App
if __name__ == "__main__":
    main()
