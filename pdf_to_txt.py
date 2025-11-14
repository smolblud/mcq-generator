# pdf_to_txt.py
import os
import sys
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract

# ---------- CONFIG ----------
POPPLER_PATH = r"C:\Users\Areesha\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# If Tesseract is not in PATH, set full path below (uncomment and edit)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ---------- END CONFIG ----------

pdf_files = [
    r"C:\Projects\React Projects\mcq-generator\data\physics\v3-11-physics-book-2025.pdf",
    r"C:\Projects\React Projects\mcq-generator\data\physics\v2-12-physics-book-2025.pdf",
    r"C:\Projects\React Projects\mcq-generator\data\math\v3-11-maths-book-2025.pdf",
    r"C:\Projects\React Projects\mcq-generator\data\math\v2-12-maths-book-2025.pdf",
    r"C:\Projects\React Projects\mcq-generator\data\english\v2-12-english-book-2025.pdf",
    r"C:\Projects\React Projects\mcq-generator\data\english\v2-11-english-2025.pdf"
]

# Small test cap — set to None to process all pages
max_pages_for_test = None
# max_pages_for_test = 5

def clean_text(text):
    # remove watermark lines (common pattern you reported)
    lines = [ln for ln in text.splitlines() if "studyplusplus.com" not in ln.strip().lower()]
    # collapse multiple blank lines
    out_lines = []
    prev_blank = False
    for ln in lines:
        if ln.strip() == "":
            if not prev_blank:
                out_lines.append("")
            prev_blank = True
        else:
            out_lines.append(ln)
            prev_blank = False
    return "\n".join(out_lines).strip() + "\n\n"

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def process_pdf(pdf_path):
    subject = os.path.basename(os.path.dirname(pdf_path))  # english / math / physics
    fname = os.path.splitext(os.path.basename(pdf_path))[0]
    out_folder = os.path.join(os.path.dirname(pdf_path))
    ensure_folder(out_folder)
    txt_path = os.path.join(out_folder, f"{fname}.txt")

    print(f"\n=== Processing: {pdf_path}")
    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
        page_count = int(info.get("Pages", info.get("pages", 0)))
    except Exception as e:
        print("ERROR: pdfinfo_from_path failed:", e)
        return

    print("Pages:", page_count)
    if max_pages_for_test:
        page_count = min(page_count, max_pages_for_test)
        print(f"(testing) Processing first {page_count} pages only")

    # open output file for streaming writes
    with open(txt_path, "w", encoding="utf-8") as out_f:
        for p in range(1, page_count + 1):
            try:
                # convert only this page -> returns list with 1 PIL image
                images = convert_from_path(pdf_path, dpi=300,
                                           first_page=p, last_page=p,
                                           poppler_path=POPPLER_PATH)
                if not images:
                    print(f"Page {p}: no image returned")
                    continue
                img = images[0]
                # OCR
                text = pytesseract.image_to_string(img)
                cleaned = clean_text(text)
                out_f.write(f"--- PAGE {p} ---\n")
                out_f.write(cleaned)
                out_f.flush()
                print(f"Page {p} -> OCR length {len(cleaned)} chars")
            except Exception as e:
                print(f"Error on page {p}:", e)
                # optionally continue to next page
                continue

    print(f"Saved OCR text to: {txt_path}")

def main():
    # quick sanity checks
    if not os.path.isdir(POPPLER_PATH):
        print("POPPLER_PATH not found:", POPPLER_PATH)
        print("Set POPPLER_PATH to your Poppler 'bin' folder (where pdftoppm.exe lives).")
        sys.exit(1)

    # process each file
    for pdf in pdf_files:
        if not os.path.isfile(pdf):
            print("Missing file, skipping:", pdf)
            continue
        process_pdf(pdf)

if __name__ == "__main__":
    main()
