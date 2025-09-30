import os
import json
import fitz  # PyMuPDF
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import glob

load_dotenv()  # Load .env file


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")
OCR_JSON_FILE = "ocr_output.json"

print("[DEBUG] Configuring APIs...")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")
client = OpenAI(api_key=OPENAI_API_KEY)

pdf_filename = os.path.basename(PDF_PATH)
OCR_JSON_FILE = f"ocr_output_{os.path.splitext(pdf_filename)[0].replace(' ', '_')}.json"


def cleanup_previous_outputs():
    # Delete old OCR output if it exists
    if os.path.exists(OCR_JSON_FILE):
        print(f"[DEBUG] Removing old OCR file: {OCR_JSON_FILE}")
        os.remove(OCR_JSON_FILE)

    # Delete all page images
    for img_file in glob.glob("page_*.png"):
        print(f"[DEBUG] Removing old image file: {img_file}")
        os.remove(img_file)


def extract_text_from_pdf(pdf_path):
    print(f"[DEBUG] Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page_texts = {}

    for page_num in range(len(doc)):
        print(f"[DEBUG] Processing page {page_num + 1}...")
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_path = f"page_{page_num+1}.png"
        print(f"[DEBUG] Saving page image: {img_path}")
        pix.save(img_path)

        print("[DEBUG] Uploading page to Gemini OCR...")
        uploaded_file = genai.upload_file(path=img_path)

        print("[DEBUG] Generating text content...")
        response = model.generate_content([
            "Extract the text from this scanned deed page.",
            uploaded_file
        ])

        page_texts[str(page_num + 1)] = response.text.strip()
        print(f"[DEBUG] Page {page_num + 1} text extracted.")

    print("[DEBUG] Finished extracting text from PDF.")
    return page_texts


def extract_fields_from_text(page_texts):
    print("[DEBUG] Combining page texts...")
    full_text = "\n\n".join(
        [f"--- Page {p} ---\n{text}" for p, text in page_texts.items()]
    )

    prompt = f"""
You are an expert legal document parser.

Extract the following fields from the deed document.
If a field is missing, return null.

Required fields:
- InstrumentID
- RecordingDate
- LenderName
- BorrowerOrOwnerName
- State
- GrantorName
- LegalDescription
- LoanAmount

Rules:
- InstrumentID is usually at the top right of the first page.
- RecordingDate usually appears as "Recorded on" or "Recording Date".
- LoanAmount often follows "Principal Amount" or "Loan Amount".
- LegalDescription may be long; include the full text.
- Always output valid JSON with keys exactly as above.
- For each field, include also a "source_page".
- GrantorName and LenderName usually appears as Grantor:/Lender:

Document text:
{full_text}
    """

    print("[DEBUG] Sending data to OpenAI GPT-4o-mini for extraction...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a structured data extractor."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    print("[DEBUG] Received response from OpenAI.")
    extracted_data = json.loads(response.choices[0].message.content)

    return extracted_data


if __name__ == "__main__":
    print("[INFO] Starting document parsing...")

    # Cleanup old outputs
    cleanup_previous_outputs()

    # Extract text from PDF
    page_texts = extract_text_from_pdf(PDF_PATH)

    # Save OCR results
    print(f"[INFO] Saving OCR results to '{OCR_JSON_FILE}'...")
    with open(OCR_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump({"page_texts": page_texts}, f, indent=2)

    # Extract structured fields
    print("[INFO] Extracting structured fields...")
    extracted_fields = extract_fields_from_text(page_texts)

    print("[INFO] Extraction complete. Results:")
    print(json.dumps(extracted_fields, indent=2))

    # Cleanup images after processing
    print("[INFO] Cleaning up temporary images...")
    cleanup_previous_outputs()

    print("[INFO] Done.")


