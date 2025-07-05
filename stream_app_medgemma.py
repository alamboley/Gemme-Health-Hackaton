"""Streamlit IGT Assistant – image crop/zoom + live audio transcription
---------------------------------------------------------------------
* Upload an image → enter a prompt (e.g. "zoom on the left inner thoracic cage")
  → cropped/zoomed result displayed.
* Record audio or upload audio → transcription via Gemini 2.5 Flash.

🔑 WARNING – replace API_KEY with your own Google GenAI key.
🔑 Optional – if you load MedGemma/PaliGemma from Hugging Face, make sure you
            requested access and set an HF token in your environment.
"""

import io
import re
import wave
from pathlib import Path
from typing import Tuple, Union

from transformers import pipeline  # huggingface transformers >= 4.40
from transformers import AutoProcessor, AutoModelForImageTextToText

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ---------- Google GenAI ------------------------------------------------------
import google.generativeai as genai

from huggingface_hub import login
login(new_session=False)

API_KEY = "AIzaSyCUjnELWeJ9dNYlPy4iq9-B2ui1XYS8ZSk"
MODEL_NAME = "gemini-2.5-flash"

pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
result = pipe(text=messages)
print(result[0]['generated_text'])
processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
model = AutoModelForImageTextToText.from_pretrained("google/medgemma-4b-it")

# model.predi

genai.configure(api_key=API_KEY)

# ---------- (Optional) MedGemma / PaliGemma ----------------------------------
# Uncomment + configure if you have access to the gated repo
# from transformers import pipeline  # huggingface transformers >= 4.40
# model_id = "google/paligemma-3b-mix-448"   # or medgemma‑3b‑mix-448
# model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
# model_variant = "3b"       # used only for demo “thinking” print
# is_thinking = False

# -----------------------------------------------------------------------------
# Helper – Gemini transcription
# -----------------------------------------------------------------------------

def transcribe_audio(wav_bytes: bytes) -> str:
    """Send raw WAV bytes to Gemini Flash 2.5 and return the transcript."""
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    contents = [
        {
            "role": "user",
            "parts": [
                {"mime_type": "audio/wav", "data": wav_bytes},
                {
                    "text": (
                        "Generate a verbatim transcription of the spoken words in this audio. "
                        "Ignore music or other background sounds."
                    )
                },
            ],
        }
    ]
    stream = model.generate_content(
        contents,
        generation_config={"temperature": 1.0, "max_output_tokens": 8192},
        stream=True,
    )
    return "".join(chunk.text for chunk in stream if hasattr(chunk, "text"))

# -----------------------------------------------------------------------------
# Helper – simple "smart" cropper (no external model)
# -----------------------------------------------------------------------------

def keyword_crop(img: Image.Image, prompt: str, zoom_factor: float = 2.0) -> Image.Image:
    """Return a cropped & resized image based on simple keyword rules.

    * "left" / "gauche"  → left half
    * "right" / "droite" → right half
    * "top" / "haut"      → top half
    * "bottom" / "bas"    → bottom half
    * if combination (e.g. "top left"), crop quarter.
    * if no keyword found, return the original image.
    """
    w, h = img.size
    x1 = y1 = 0
    x2, y2 = w, h

    p = prompt.lower()
    # Horizontal
    if ("left" in p) or ("gauche" in p):
        x2 = w // 2
    elif ("right" in p) or ("droite" in p):
        x1 = w // 2
    # Vertical
    if ("top" in p) or ("haut" in p):
        y2 = h // 2
    elif ("bottom" in p) or ("bas" in p):
        y1 = h // 2

    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        # fallback – no crop
        return img

    cropped = img.crop((x1, y1, x2, y2))
    if zoom_factor != 1.0:
        new_size = (int(cropped.width * zoom_factor), int(cropped.height * zoom_factor))
        cropped = cropped.resize(new_size, Image.LANCZOS)
    return cropped

# -----------------------------------------------------------------------------
# (Optional) – wrapper that would call MedGemma / PaliGemma in multimodal mode
# -----------------------------------------------------------------------------

def simple_medgemma_agent(
    prompt_text: str,
    image_input: Union[Image.Image, None] = None,
    system_instruction: str = "You are a helpful medical assistant.",
):
    """Demo wrapper.

    * If the prompt includes "crop" or "zoom": perform keyword_crop locally.
    * Otherwise, try to send to a (commented‑out) MedGemma pipeline.
    """

    if image_input is not None and re.search(r"\b(crop|zoom)\b", prompt_text, re.I):
        return keyword_crop(image_input, prompt_text, zoom_factor=2.0)

    # ---  Text or image‑to‑text with MedGemma  --------------------------------
    # Remove the following lines' comments if you have access to the model.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        },
    ]
    if image_input is not None:
        messages[1]["content"].append({"type": "image", "image": image_input})
        pipe = pipeline("image-text-to-text", model=model_id, model_kwargs=model_kwargs)
        pipe.model.generation_config.do_sample = False
        output = pipe(text=messages, max_new_tokens=300)
        return output[0]["generated_text"][-1]["content"]
    else:
        pipe = pipeline("text-generation", model=model_id, model_kwargs=model_kwargs)
        pipe.model.generation_config.do_sample = False
        output = pipe(messages, max_new_tokens=500)
        return output[0]["generated_text"][-1]["content"]
    """

    # Fallback stub
    return "[MedGemma not configured – returning stub answer]"

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="IGT Assistant", layout="wide")

st.sidebar.title("🔧 Mode d'emploi")
st.sidebar.markdown("1️⃣ Chargez une image.\n2️⃣ Entrez un prompt (ex. ‘zoom …’).\n3️⃣ Enregistrez votre voix et transcrivez." )

st.title("📁 IGT Assistant")
col1, col2 = st.columns(2)

# ---------------- Column 1 – Image + prompt ----------------------------------
with col1:
    st.subheader("🖼️ Upload Image")
    img_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "gif"], key="img_upload")

    if img_file:
        pil_in = Image.open(img_file).convert("RGB")
        st.image(pil_in, caption=img_file.name, use_column_width=True)

        prompt_text = st.text_input(
            "👉 Prompt (ex.: ‘zoom on the left inner thoracic cage’)",
            key="img_prompt",
        )

        if st.button("⚡ Process image", key="process_img"):
            with st.spinner("Processing …"):
                try:
                    result = simple_medgemma_agent(prompt_text=prompt_text, image_input=pil_in)
                except Exception as e:
                    st.error(f"Error: {e}")
                else:
                    if isinstance(result, Image.Image):
                        st.success("✅ Cropped / zoomed image:")
                        enhancer = ImageEnhance.Contrast(result)
                        st.image(result, caption="Result", use_column_width=True)
                    else:
                        st.success("✅ Model response:")
                        st.write(result)
    else:
        st.info("No image uploaded yet.")

# ---------------- Column 2 – Audio capture & STT -----------------------------
with col2:
    st.subheader("🎤 Live Audio Recording & Transcription (Gemini)")

    ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
        audio_receiver_size=4096,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    if "audio_frames" not in st.session_state:
        st.session_state.audio_frames = []

    if ctx.audio_receiver:
        frames = ctx.audio_receiver.get_frames()
        if frames:
            st.session_state.audio_frames.extend(frames)

    st.caption(f"Buffered frames: {len(st.session_state.audio_frames)}")

    if st.button("⏹️ Stop & Transcribe", type="primary"):
        frames = st.session_state.audio_frames
        if not frames:
            st.warning("No audio captured.")
        else:
            arrays = [f.to_ndarray() for f in frames]
            audio_np = np.concatenate(arrays, axis=0)
            sr = 48000
            pcm = (
                (audio_np * 32767).astype(np.int16)
                if audio_np.dtype == np.float32
                else audio_np.astype(np.int16)
            )
            if pcm.ndim > 1:
                pcm = pcm[:, 0]

            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
            wav_buf.seek(0)

            st.audio(wav_buf, format="audio/wav")

            with st.spinner("Transcribing with Gemini …"):
                try:
                    transcript = transcribe_audio(wav_buf.getvalue())
                except Exception as e:
                    st.error(f"Gemini request failed: {e}")
                else:
                    st.subheader("📝 Transcript")
                    st.write(transcript or "(no speech detected)")
            st.session_state.audio_frames = []
    else:
        st.info("Click *Start* above to begin recording.")

# ---------------- Audio file uploader fallback -------------------------------

st.markdown("---")
st.subheader("📂 Upload an Audio File (optional)")

aud_file = st.file_uploader("Or choose an audio file", type=["mp3", "wav", "ogg", "flac"], key="aud_upload")

if aud_file and st.button("Transcribe file with Gemini", key="transcribe_file"):
    with st.spinner("Transcribing file …"):
        transcript = transcribe_audio(aud_file.read())
        st.subheader("📝 Transcript")
        st.write(transcript or "(no speech detected)")

# ---------------- Footer ------------------------------------------------------

st.markdown("---")
st.caption("✨ Built with Streamlit · Gemini Flash 2.5 · (Med)Gemma ✨")
