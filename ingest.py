import fitz


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


if __name__ == "__main__":
    text = extract_text("sample.pdf")
    print(f"Extracted {len(text)} characters")
    print("\n--- Preview (first 500 chars) ---")
    print(text[:500])
