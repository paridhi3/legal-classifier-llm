import os
import io
import pdfplumber
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# LangChain + Azure OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor, tool

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_END_POINT"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    model=os.getenv("MODEL_NAME"),
    api_version=os.getenv("API_VERSION"),
    temperature=0
)

# ---------------------------
# Utilities for Text Extraction
# ---------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)

# ---------------------------
# Tool: Classifier
# ---------------------------
@tool
def classifier(document_text: str) -> str:
    """
    Classify a given legal document's text into 'Judgement' or 'Non-Judgement'.
    """
    classification_prompt = f"""
    You are a legal document classifier. 
    Determine if the following document is a **Judgement** issued by a court, 
    or a **Non-Judgement** (such as a contract, agreement, letter, application, etc.).

    Document text:
    {document_text[:5000]}

    Respond ONLY with 'Judgement' or 'Non-Judgement'.
    """
    response = llm.invoke([{"role": "user", "content": classification_prompt}])
    return response.content.strip()

# ---------------------------
# Agent Setup
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a legal document classification agent."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [classifier]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Legal Document Classifier", page_icon="⚖️", layout="centered")
st.title("⚖️ Legal Document Classifier (LLM Agent)")
st.write("Upload one or more legal documents (PDF or image) to classify as **Judgement** or **Non-Judgement** using GPT-powered agents.")

uploaded_files = st.file_uploader(
    "Upload PDF or image files", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Processing documents... Please wait ⏳")

    results = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()

        # Extract text depending on file type
        if uploaded_file.type == "application/pdf":
            content = extract_text_from_pdf(file_bytes)
        else:
            content = extract_text_from_image(file_bytes)

        if not content.strip():
            category = "Unclassified (No text found)"
            confidence = "N/A"
        else:
            try:
                result = agent_executor.invoke({"input": content})
                category = result["output"]
                confidence = "LLM classification (no score)"
            except Exception as e:
                category = "Error"
                confidence = str(e)

        results.append({
            "File Name": uploaded_file.name,
            "Category": category,
            "Confidence": confidence
        })

    # Display results
    st.success("✅ Classification complete!")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
