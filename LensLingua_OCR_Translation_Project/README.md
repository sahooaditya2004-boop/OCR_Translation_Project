# LensLingua — OCR + Translation

A deployable image-to-text and translation web app built with **OpenCV**, **Tesseract OCR**, **Streamlit**, and the **OpenRouter API**.

## Features

- Upload JPG, PNG, WEBP, and TIFF images.
- OpenCV preprocessing: upscaling, denoising, local contrast enhancement, thresholding, and optional deskew correction.
- Tesseract OCR for English, Hindi, Odia, Bengali, Telugu, Tamil, and Spanish.
- Edit OCR output before translation.
- Translate with OpenAI GPT-4o or Mistral Mixtral 8x7B through OpenRouter.
- Choose a target language (including languages beyond the OCR list).
- Download the translation or both the extracted and translated text.
- Retry handling, timeouts, useful errors, and a responsive Streamlit UI.
- Deploy to Streamlit Community Cloud or run in Docker.

## 1. Requirements

- Python 3.10 or 3.11 recommended.
- Tesseract OCR installed locally with language data for the language(s) you use.
- An OpenRouter account/API key for translation. OCR works without an API key.

## 2. Run locally

### Windows

1. Install Tesseract OCR for Windows and ensure `tesseract.exe` is added to your PATH. Install language data for `eng`, `hin`, `ori`, `ben`, `tel`, `tam`, and `spa` as needed.
2. Unzip this project and open a terminal in the project directory.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`, or enter your key in the app sidebar. The app currently reads environment variables directly; for PowerShell you can set one for the current terminal with:

   ```powershell
   $env:OPENROUTER_API_KEY="your_key_here"
   ```
5. Start the app:

   ```powershell
   streamlit run app.py
   ```

   Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

### macOS

Install Tesseract and the language packs with your package manager, install the Python requirements, then run `streamlit run app.py`.

### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-ori tesseract-ocr-ben tesseract-ocr-tel tesseract-ocr-tam tesseract-ocr-spa
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 3. OpenRouter API key

Create an API key from your OpenRouter account. Never commit a real key to GitHub.

You can either:
- enter the key into the password field in the app sidebar, or
- set `OPENROUTER_API_KEY` in your environment.

The app sends text to OpenRouter only when you press **Translate text**. Model IDs used:
- `openai/gpt-4o`
- `mistralai/mixtral-8x7b-instruct`

Model access and pricing can change; confirm availability in your OpenRouter account.

## 4. Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload the project files (do not upload `.env` or any real API key).
2. Open Streamlit Community Cloud and create a new app from that repository.
3. Set the main file to `app.py`.
4. In app settings, add a secret:

   ```toml
   OPENROUTER_API_KEY = "your_real_key_here"
   ```

5. Deploy. Streamlit will install Python packages from `requirements.txt` and system packages from `packages.txt`.

If a particular Tesseract language package is unavailable on the build image, remove that package from `packages.txt` and install it through a compatible Docker deployment instead.

## 5. Deploy with Docker

Build and run:

```bash
docker build -t lenslingua .
docker run --rm -p 8501:8501 -e OPENROUTER_API_KEY="your_real_key_here" lenslingua
```

Open `http://localhost:8501`.

## How it works

1. **Upload:** load an image into the Streamlit interface.
2. **Preprocess:** OpenCV resizes and enhances the image; deskew is attempted when enabled.
3. **OCR:** Tesseract converts the processed image to editable text.
4. **Review:** correct any OCR mistakes.
5. **Translate:** the app sends the reviewed text to the chosen OpenRouter model.
6. **Export:** download the translated text or a combined OCR + translation file.

## Troubleshooting

- `TesseractNotFoundError`: install Tesseract and add it to PATH; restart the terminal after installation.
- Missing language: verify `tesseract --list-langs` includes the required code (`eng`, `hin`, `ori`, `ben`, `tel`, `tam`, `spa`).
- HTTP 401: check the API key.
- HTTP 402/429: check account credits, rate limits, and model access.
- OCR output is garbled: use a clearer image, choose the correct OCR language, and test the other preprocessing modes.
- The model returns no text: check model access or try the other model.

## Privacy note

This project does not store uploaded images or translations in a database. The uploaded image is processed in the app session. Text is sent to OpenRouter only when translation is requested. Do not use sensitive documents unless the processing is acceptable for your use case.

## Project structure

```text
lenslingua/
├── app.py
├── requirements.txt
├── packages.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```
