# Streamlit Media Uploader & Gemini Speech‑to‑Text Demo (dev key inline)
# -----------------------------------------------------------------------------
# Quick start (local dev):
#   pip install --upgrade streamlit pillow streamlit-webrtc google-generativeai
#   streamlit run streamlit_media_gemini.py
# NOTE: The Google GenAI API key is hard‑coded below **for development only**.
#       Do NOT commit real secrets to Git in production!

import io
import wave
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

import google.generativeai as genai

# -----------------------------------------------------------------------------
# Config – hard‑coded creds (DEV ONLY!)
# -----------------------------------------------------------------------------
API_KEY = "AIzaSyCUjnELWeJ9dNYlPy4iq9-B2ui1XYS8ZSk"
MODEL_NAME = "gemini-2.5-flash"  # supports audio parts

# One‑liner auth
genai.configure(api_key=API_KEY)

# -----------------------------------------------------------------------------
# Helper – Gemini transcription
# -----------------------------------------------------------------------------

def transcribe_audio(wav_bytes: bytes) -> str:
    """Send raw WAV bytes to Gemini Flash 2.5 and return the transcript."""

    model = genai.GenerativeModel(model_name=MODEL_NAME)

    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "mime_type": "audio/wav",
                    "data": wav_bytes,
                },
                {
                    "text": "Generate a verbatim transcription of the spoken words in this audio. Ignore music or other background sounds.",
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

# prompt: créé un agent qui utilise MedGemma initialisé plus haut

def simple_medgemma_agent(prompt_text, image_input=None, system_instruction="You are a helpful medical assistant.", crop_box=None):
    """
    Runs inference with the initialized MedGemma model or crops an image.

    Args:
        prompt_text: The text prompt for the model (used in text-to-text or image-to-text mode).
        image_input: An optional PIL Image object for multimodal tasks or cropping.
        system_instruction: The system instruction for the model (used in text-to-text or image-to-text mode).
        crop_box: A tuple of (left, upper, right, lower) coordinates for cropping (used in cropping mode).

    Returns:
        The generated response from the MedGemma model (in text-to-text or image-to-text mode)
        or the cropped PIL Image object (in cropping mode).
    """

    if crop_box and image_input:
        # Cropping mode + resize
        cropped = image_input.crop(crop_box).resize((512, 512))
        return cropped

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        }
    ]

    if image_input:
        messages[1]["content"].append({"type": "image", "image": image_input})
        # Use the image-text-to-text pipeline for multimodal inputs
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            model_kwargs=model_kwargs,
        )
        pipe.model.generation_config.do_sample = False
        output = pipe(text=messages, max_new_tokens=300)
        response = output[0]["generated_text"][-1]["content"]

    else:
        # Use the text-generation pipeline for text-only inputs
        pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
        )
        pipe.model.generation_config.do_sample = False
        max_new_tokens = 1500 if "27b" in model_variant and is_thinking else 500
        output = pipe(messages, max_new_tokens=max_new_tokens)
        response = output[0]["generated_text"][-1]["content"]
        if "27b" in model_variant and is_thinking and "<unused95>" in response:
            thought, response = response.split("<unused95>")
            thought = thought.replace("<unused94>thought\n", "")
            print(f"**[ MedGemma thinking ]**\n\n{thought}\n\n---")


    return response

# -----------------------------------------------------------------------------
# Page & sidebar
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Live demo", layout="wide")

st.sidebar.title("🔧 Mode d'emploi")
st.sidebar.write("1️⃣ Chargez une image.\n2️⃣ Enregistrez votre voix.\n3️⃣ Stoppez & transcrivez avec Gemini.")

# -----------------------------------------------------------------------------
# Layout – image uploader | audio recorder
# -----------------------------------------------------------------------------

st.title("📁 Media Uploader & Gemini STT Demo")
col1, col2 = st.columns(2)

# ---- Column 1 : image upload -------------------------------------------------
with col1:
    st.subheader("🖼️ Upload Image")
    img_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "gif"])
    if img_file:
        st.image(Image.open(img_file), caption=img_file.name, use_column_width=True)
    else:
        st.info("No image uploaded yet.")

# ---- Column 2 : live recording & STT ----------------------------------------
# NOTE: If you still hit overflow, you can bump audio_receiver_size again or
#       shorten the recording duration. For very long recordings consider
#       streaming frames to the backend incrementally rather than holding all
#       frames in memory.
with col2:
    st.subheader("🎤 Live Audio Recording & Transcription (Gemini)")

    ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
        audio_receiver_size=4096,  # increased to avoid queue overflow
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    if "audio_frames" not in st.session_state:
        st.session_state.audio_frames = []

    # Always show buffer size for debug
    st.caption(f"Buffered frames: {len(st.session_state.audio_frames)}")

    if ctx.audio_receiver:
        frames = ctx.audio_receiver.get_frames()
        # Debug: show how many frames we get each rerun
        st.write("Frames captured this tick:", len(frames))
        if frames:
            st.session_state.audio_frames.extend(frames)

    if st.button("⏹️ Stop & Transcribe", type="primary"):
        frames = st.session_state.audio_frames
        if not frames:
            st.warning("No audio captured.")
        else:
            # Concatenate frames and convert to int16 mono
            arrays = [f.to_ndarray() for f in frames]
            audio_np = np.concatenate(arrays, axis=0)
            sr = 48000
            pcm = (audio_np * 32767).astype(np.int16) if audio_np.dtype == np.float32 else audio_np.astype(np.int16)
            if pcm.ndim > 1:
                pcm = pcm[:, 0]

            # Encode to WAV (in‑memory)
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
            wav_buf.seek(0)

            # Preview playback
            st.audio(wav_buf, format="audio/wav")

            # --- Transcription ---
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
        st.info("Click *Start* to begin recording.")

# -----------------------------------------------------------------------------
# Audio file uploader fallback -------------------------------------------------

st.markdown("---")
st.subheader("📂 Upload an Audio File (optional)")
aud_file = st.file_uploader("Or choose an audio file", type=["mp3", "wav", "ogg", "flac"])
if aud_file and st.button("Transcribe file with Gemini"):
    with st.spinner("Transcribing file …"):
        transcript = transcribe_audio(aud_file.read())
        st.subheader("📝 Transcript")
        st.write(transcript or "(no speech detected)")

st.markdown("---")
st.caption("✨ Streamlit + Gemini Flash 2.5 (dev key inline) ✨")
