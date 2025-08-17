import os
import io
import pdfplumber
import pytesseract
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Utilities ----------
def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)

def classify(content: str):
    """Send text content to GPT-4o-mini for classification"""
    prompt = f"""
    You are a legal document classifier.
    Classify the following document as either 'Judgement' or 'Non-Judgement'.

    Document Content:
    {content[:5000]}  # limit length
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful legal document classifier."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# ---------- Streamlit UI ----------
st.set_page_config(page_title="LLM Legal Document Classifier", page_icon="⚖️")
st.title("⚖️ LLM Agent - Legal Document Classifier")
st.write("Upload a PDF or Image, and classify it as **Judgement** or **Non-Judgement** using GPT-4o Agents.")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    # Determine file type
    if uploaded_file.type == "application/pdf":
        st.info("Extracting text from PDF...")
        content = extract_text_from_pdf(file_bytes)
    else:
        st.info("Extracting text from Image using OCR...")
        content = extract_text_from_image(file_bytes)

    if not content.strip():
        st.warning("No text extracted. Falling back to Vision-based classification...")
        # Send the raw image/PDF page to GPT-4o Vision
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal document classifier."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Classify this document as Judgement or Non-Judgement."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + file_bytes.decode("latin1")}}
                ]}
            ]
        )
        result = response.choices[0].message.content
    else:
        st.text_area("Extracted Text (first 1000 chars)", content[:1000])
        result = classify_with_gpt(content)

    st.success("✅ Classification Result")
    st.write(result)
