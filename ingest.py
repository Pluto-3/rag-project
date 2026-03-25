import fitz

pdf_path = "sample.pdf"

def extract_text(filepath):
    doc = fitz.open(filepath)
    full_text = ""

    for page in doc:
        text = page.get_text()
        text = text.replace("\n\n", "\n")
        text = " ".join(text.split())
        full_text += text + " "

    doc.close()
    return full_text.strip()

text = extract_text(pdf_path)

print(f"Extracted {len(text)} characters")
print("\n--- Preview (first 500 chars) ---")
print(text[:500])