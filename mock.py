# INSTALL REQUIRED PACKAGES FIRST:
# pip install pytesseract pdf2image spacy pandas scikit-learn opencv-python
# https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0
# https://github.com/UB-Mannheim/tesseract/wiki
# python -m spacy download de_core_news_sm


import os
import re
import shutil
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import spacy
import pandas as pd

# Check if Poppler is installed
if not shutil.which("pdftoppm"):
    raise EnvironmentError("Poppler is not installed or not found in PATH. Install it from: https://github.com/oschwartz10612/poppler-windows/releases/")

# OCR & Image Preprocessing
def preprocess_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(img)

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    full_text = ""

    for idx, img in enumerate(images):
        img = preprocess_image(img)
        text = pytesseract.image_to_string(
            img,
            lang='deu',  # make sure 'deu' language is installed for Tesseract
            config='--psm 6 -c preserve_interword_spaces=1'
        )
        if len(text.strip()) < 10:
            print(f"Warning: OCR output is short on page {idx + 1}")
        full_text += text + "\n\n"

    # Postprocessing
    cleaned_text = re.sub(r'[^\w\s.,:;!?§$%&/()=\'\-—\n]', '', full_text)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text

# Rule-based metadata extraction
def extract_metadata_rules(text, glossary):
    metadata = {
        'title': None,
        'year': None,
        'publisher': None
    }

    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        metadata['year'] = year_match.group()

    for line in text.split('\n'):
        if re.match(r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\s\-]{15,}$", line.strip()):
            metadata['title'] = line.strip()
            break

    for term, field in glossary.items():
        if term in text:
            match = re.search(fr"{term}[:\s]*(.{{5,40}})", text)
            if match:
                metadata[field] = match.group(1).strip()

    print("metadata")
    print(metadata)
    return metadata

# Format metadata for ML training (NER prep)
def prepare_training_data(texts, labels):
    training_data = []
    nlp = spacy.blank("de")

    label_map = {
        "title": "LAW_TITLE",
        "year": "YEAR",
        "publisher": "PUBLISHER"
    }

    for text, annots in zip(texts, labels):
        doc = nlp.make_doc(text)
        ents = []
        for label, value in annots.items():
            if value and value in text:
                start = text.find(value)
                end = start + len(value)
                span = doc.char_span(start, end, label=label_map.get(label, label.upper()))
                if span is not None:
                    ents.append(span)
        training_data.append((doc.text, {"entities": [(e.start_char, e.end_char, e.label_) for e in ents]}))

    return training_data

# Glossary for rule-based detection
LEGAL_GLOSSARY = {
    "Herausgeber": "publisher",
    "Verlag": "publisher",
    "Autor": "author",
    "Datum": "date",
    "Gesetz": "title",
    "Amtliche Ausgabe": "official_source"
}

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    sample_pdf = "Bayerisches_Berufsbildungsgesetz_BayBerBiG.pdf"

    if not os.path.exists(sample_pdf):
        raise FileNotFoundError(f"PDF not found: {sample_pdf}")

    # Step 1: OCR extraction
    extracted_text = extract_text_from_pdf(sample_pdf)
    print(f"\n--- Extracted Text Sample ---\n{extracted_text[:500]}...\n")

    # Step 2: Rule-based metadata extraction
    metadata = extract_metadata_rules(extracted_text, LEGAL_GLOSSARY)
    print("\n--- Rule-based Metadata ---")
    for k, v in metadata.items():
        print(f"{k.upper()}: {v or 'Not found'}")

    # Step 3: Save metadata to CSV
    df = pd.DataFrame([metadata])
    df.to_csv("extracted_metadata.csv", index=False)
    print("\nMetadata saved to extracted_metadata.csv")

    # Optional: Prepare for ML training
    training_data = prepare_training_data([extracted_text], [metadata])
    print("\nTraining data preview:", training_data[0])
