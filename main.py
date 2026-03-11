from fastapi import FastAPI, UploadFile, File
from groq import Groq
import fitz
import json
import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_bytes

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Resume AutoFill API")


# -----------------------------
# NORMAL PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(file_bytes):

    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""

    for page in pdf:
        text += page.get_text()

    return text


# -----------------------------
# OCR EXTRACTION (for scanned PDFs)
# -----------------------------
def extract_text_with_ocr(file_bytes):

    images = convert_from_bytes(file_bytes)

    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)

    return text


# -----------------------------
# CLEAN LLM OUTPUT
# -----------------------------
def clean_llm_output(text):

    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.replace("json", "").strip()

    return text


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/autofill")
async def autofill_form(resume: UploadFile = File(...)):

    file_bytes = await resume.read()

    # Try normal extraction
    resume_text = extract_text_from_pdf(file_bytes)

    # If empty → use OCR
    if not resume_text.strip():
        print("Using OCR extraction...")
        resume_text = extract_text_with_ocr(file_bytes)

    print("------EXTRACTED TEXT------")
    print(resume_text[:1000])

    if not resume_text.strip():
        return {
            "status": "error",
            "message": "Could not extract any text from the resume."
        }

    resume_text = resume_text[:12000]

    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
You are an expert resume parser.

Extract the following information from the resume text.

If any field is missing return null.

Resume Text:
{resume_text}

Return ONLY valid JSON in this format:

{{
"first_name": string | null,
"last_name": string | null,
"email": string | null,
"phone": string | null,
"skills": list,
"experience_years": number | null,
"current_ctc": string | null,
"expected_ctc": string | null,
"city": string | null,
"state": string | null,
"country": string | null,
"pin_code": string | null,
"address": string | null
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.choices[0].message.content

    result = clean_llm_output(result)

    try:
        parsed_data = json.loads(result)
    except:
        parsed_data = {"raw_output": result}

    return {
        "status": "success",
        "parsed_data": parsed_data
    }


# -----------------------------
# ROOT ENDPOINT
# -----------------------------
@app.get("/")
def home():
    return {"message": "Resume AutoFill API Running"}