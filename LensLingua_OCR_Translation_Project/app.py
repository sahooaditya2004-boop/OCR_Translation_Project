import os
import io
import time
import hashlib
from typing import Optional

import cv2
import numpy as np
import pytesseract
import requests
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

APP_TITLE = "LensLingua"
APP_SUBTITLE = "Extract text from images. Translate it across languages."
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

LANGUAGES = {
    "English": {"tesseract": "eng", "translation": "English"},
    "Hindi": {"tesseract": "hin", "translation": "Hindi"},
    "Odia": {"tesseract": "ori", "translation": "Odia"},
    "Bengali": {"tesseract": "ben", "translation": "Bengali"},
    "Telugu": {"tesseract": "tel", "translation": "Telugu"},
    "Tamil": {"tesseract": "tam", "translation": "Tamil"},
    "Spanish": {"tesseract": "spa", "translation": "Spanish"},
}
MODELS = {
    "GPT-4o (OpenAI)": "openai/gpt-4o",
    "Mixtral 8x7B (Mistral)": "mistralai/mixtral-8x7b-instruct",
}

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    :root { color-scheme: light; }
    .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 40%); }
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #e2e8f0; border-radius: 22px;
            background: linear-gradient(115deg, #eef2ff, #f0fdfa); margin-bottom: 1.4rem; }
    .hero h1 { margin: 0; letter-spacing: -0.04em; font-size: 2.25rem; color: #172554; }
    .hero p { color: #475569; margin: .45rem 0 0 0; font-size: 1.03rem; }
    .step-card { border: 1px solid #e2e8f0; border-radius: 16px; padding: 1rem 1.1rem; background: white; }
    .muted { color: #64748b; font-size: .9rem; }
    div[data-testid="stMetric"] { background: white; padding: .8rem; border: 1px solid #e2e8f0; border-radius: 14px; }
    .stButton > button[kind="primary"] { border-radius: 10px; font-weight: 650; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>🔎 {APP_TITLE}</h1>
  <p>{APP_SUBTITLE}<br>OpenCV preprocessing · Tesseract OCR · OpenRouter-powered translation</p>
</div>
""", unsafe_allow_html=True)

def get_api_key(sidebar_value: str) -> str:
    return sidebar_value.strip() or os.getenv("OPENROUTER_API_KEY", "").strip()

def image_to_pil(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")

def preprocess_image(image: Image.Image, mode: str = "Balanced", deskew: bool = True) -> Image.Image:
    """Prepare an image for OCR using OpenCV. Returns a PIL image."""
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Upscale small images to help OCR.
    h, w = gray.shape[:2]
    if max(h, w) < 1800:
        scale = min(2.0, 1800 / max(h, w))
        if scale > 1.05:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if mode in ("Balanced", "Strong"):
        gray = cv2.fastNlMeansDenoising(gray, None, h=8 if mode == "Balanced" else 13,
                                        templateWindowSize=7, searchWindowSize=21)
        # Improve local contrast for uneven lighting.
        clahe = cv2.createCLAHE(clipLimit=2.0 if mode == "Balanced" else 3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if mode == "Strong":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 11)
    elif mode == "High contrast":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if deskew:
        # Estimate the dominant angle from foreground pixels, then rotate conservatively.
        try:
            inverted = cv2.bitwise_not(gray)
            coords = np.column_stack(np.where(inverted > 0))
            if len(coords) > 100:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if 0.5 <= abs(angle) <= 12:
                    hh, ww = gray.shape
                    matrix = cv2.getRotationMatrix2D((ww // 2, hh // 2), angle, 1.0)
                    gray = cv2.warpAffine(gray, matrix, (ww, hh), flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass

    return Image.fromarray(gray)

def run_ocr(image: Image.Image, language_code: str, psm: int = 3) -> str:
    config = f"--oem 3 --psm {psm}"
    try:
        return pytesseract.image_to_string(image, lang=language_code, config=config).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed. On Streamlit Cloud, ensure packages.txt is committed; "
            "for local use, install Tesseract and add it to PATH."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(
            f"OCR could not run for language '{language_code}'. The required Tesseract language data "
            "may not be installed in this environment."
        ) from exc

def translate_text(text: str, target_language: str, model: str, api_key: str,
                   source_language: Optional[str] = None) -> str:
    if not api_key:
        raise ValueError("Add your OpenRouter API key in the sidebar or set OPENROUTER_API_KEY.")
    if not text.strip():
        raise ValueError("There is no text to translate.")
    source_hint = f"The source language is {source_language}. " if source_language else "Detect the source language automatically. "
    system_prompt = (
        "You are a careful professional translator. Translate the user's text faithfully. "
        f"{source_hint}Translate it into {target_language}. Preserve meaning, names, numbers, "
        "punctuation, paragraph breaks, lists, and formatting as closely as possible. "
        "Do not summarize, explain, or add commentary. Return only the translation. "
        "If some text is ambiguous or unreadable, make the most conservative translation possible."
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "LensLingua OCR Translator",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text.strip()},
        ],
        "temperature": 0.2,
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=(15, 90))
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"OpenRouter temporarily returned HTTP {response.status_code}."
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            if not response.ok:
                try:
                    detail = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    detail = response.text
                raise RuntimeError(f"OpenRouter error (HTTP {response.status_code}): {detail}")
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("OpenRouter returned no translation choices.")
            content = choices[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            result = str(content).strip()
            if not result:
                raise RuntimeError("The selected model returned an empty translation.")
            return result
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"Could not connect to OpenRouter: {exc}"
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"Translation request failed: {last_error}") from exc
    raise RuntimeError(last_error or "Translation failed after retries.")

def image_bytes(image: Image.Image, fmt="PNG") -> bytes:
    output = io.BytesIO()
    image.save(output, format=fmt)
    return output.getvalue()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Your OpenRouter key stays on the server for this session. Do not commit it to Git.")
    api_key_input = st.text_input("OpenRouter API key", type="password",
                                  value="", placeholder="sk-or-v1-…",
                                  help="Alternatively set OPENROUTER_API_KEY as an environment secret.")
    model_label = st.selectbox("Translation model", list(MODELS.keys()), index=0)
    model_id = MODELS[model_label]
    st.divider()
    st.subheader("OCR options")
    source_language = st.selectbox("Text in image (OCR language)", list(LANGUAGES.keys()), index=0)
    preprocess_mode = st.selectbox("Image enhancement", ["Balanced", "Light", "High contrast", "Strong"], index=0)
    deskew = st.checkbox("Correct slight rotation", value=True)
    psm_label = st.selectbox("Page layout", ["Automatic (recommended)", "Single block of text", "Single line", "Sparse text"],
                             index=0)
    psm = {"Automatic (recommended)": 3, "Single block of text": 6, "Single line": 7, "Sparse text": 11}[psm_label]
    st.divider()
    st.markdown("**Supported OCR languages**")
    st.caption("English · Hindi · Odia · Bengali · Telugu · Tamil · Spanish")
    st.caption("Translation target can be any language supported by the selected model.")

api_key = get_api_key(api_key_input)

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("1. Add an image")
    uploaded = st.file_uploader("Upload a JPG, PNG, WEBP, or TIFF image", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"])
    if uploaded:
        try:
            original = image_to_pil(uploaded)
            st.image(original, caption=f"Original image · {original.width} × {original.height}", use_container_width=True)
            if original.width * original.height > 25_000_000:
                st.warning("This image is very large and may use substantial memory. Consider resizing it before upload.")
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")
            original = None
    else:
        original = None
        st.info("Upload an image containing printed or handwritten text to get started.")

with right:
    st.subheader("2. Extract text")
    if original is not None:
        if st.button("✨ Preprocess & extract text", type="primary", use_container_width=True):
            with st.spinner("Enhancing image and running OCR…"):
                try:
                    processed = preprocess_image(original, preprocess_mode, deskew)
                    extracted = run_ocr(processed, LANGUAGES[source_language]["tesseract"], psm)
                    st.session_state["processed_image"] = image_bytes(processed)
                    st.session_state["ocr_text"] = extracted
                    st.session_state["editable_ocr_text"] = extracted
                    st.session_state["ocr_source_hash"] = hashlib.sha256(image_bytes(original)).hexdigest()
                    st.session_state.pop("translation_text", None)
                except Exception as exc:
                    st.error(str(exc))
        if st.session_state.get("processed_image"):
            st.image(st.session_state["processed_image"], caption="Processed image used for OCR", use_container_width=True)
    else:
        st.markdown('<div class="step-card"><p class="muted">Your extracted text will appear here after you upload an image and run OCR.</p></div>', unsafe_allow_html=True)

st.divider()
st.subheader("3. Review, edit & translate")
default_text = st.session_state.get("ocr_text", "")
edited_text = st.text_area(
    "Extracted text (editable — correct OCR mistakes before translating)",
    value=default_text,
    height=220,
    placeholder="Extracted text will appear here. You can also paste text directly.",
    key="editable_ocr_text",
)
target_language = st.selectbox(
    "Translate into",
    ["Hindi", "English", "Odia", "Bengali", "Telugu", "Tamil", "Spanish", "French", "German",
     "Japanese", "Korean", "Chinese (Simplified)", "Arabic", "Portuguese", "Russian", "Italian",
     "Urdu", "Marathi", "Gujarati", "Punjabi", "Malayalam", "Kannada"],
    index=1 if source_language != "English" else 2,
)
translate_col, reset_col = st.columns([3, 1])
with translate_col:
    translate_clicked = st.button("🌐 Translate text", type="primary", use_container_width=True)
with reset_col:
    if st.button("Clear results", use_container_width=True):
        for key in ["ocr_text", "processed_image", "translation_text", "ocr_source_hash", "editable_ocr_text"]:
            st.session_state.pop(key, None)
        st.rerun()

if translate_clicked:
    if not edited_text.strip():
        st.warning("Extract text from an image or paste some text first.")
    elif not api_key:
        st.error("OpenRouter API key is missing. Add it in the sidebar or set OPENROUTER_API_KEY.")
    else:
        with st.spinner(f"Translating with {model_label}…"):
            try:
                translated = translate_text(
                    edited_text,
                    target_language,
                    model_id,
                    api_key,
                    source_language=source_language if st.session_state.get("ocr_text") else None,
                )
                st.session_state["translation_text"] = translated
                st.session_state["translation_language"] = target_language
            except Exception as exc:
                st.error(str(exc))

translated_text = st.session_state.get("translation_text", "")
if translated_text:
    st.success(f"Translation ready · {st.session_state.get('translation_language', target_language)}")
    st.text_area("Translated text", value=translated_text, height=220, key="translation_display")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download translation (.txt)", data=translated_text.encode("utf-8"),
                           file_name="lenslingua_translation.txt", mime="text/plain", use_container_width=True)
    with c2:
        combined = f"EXTRACTED TEXT\n{'=' * 14}\n{edited_text}\n\nTRANSLATION ({st.session_state.get('translation_language', target_language)})\n{'=' * 35}\n{translated_text}\n"
        st.download_button("⬇️ Download both texts (.txt)", data=combined.encode("utf-8"),
                           file_name="lenslingua_ocr_and_translation.txt", mime="text/plain", use_container_width=True)

st.markdown("---")
with st.expander("Troubleshooting & privacy"):
    st.markdown("""
    - **OCR is inaccurate:** choose a better-lit, sharper image; try *High contrast* or *Strong* enhancement; confirm the OCR language matches the image.
    - **Missing language data:** the deployment includes Tesseract language packages via `packages.txt`. Local installations need the matching language packs.
    - **OpenRouter errors:** verify the key, account credits, selected model availability, and network access.
    - **API costs:** translation is sent to the selected OpenRouter model. Review the model's current pricing before use.
    - **Privacy:** images are processed by this app's runtime. Extracted text is sent to OpenRouter only when you press Translate. Do not upload sensitive documents unless you are comfortable with that processing.
    - **Handwriting:** OCR accuracy varies; this app is best suited to clear printed text.
    """)

st.caption("LensLingua · Built with Streamlit, OpenCV, Tesseract OCR and OpenRouter.")
